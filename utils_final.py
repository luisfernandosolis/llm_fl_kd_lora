import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model, PeftModel
import json
import random
from typing import List, Tuple
from datasets import Dataset
import copy
import math


def initialize_model(model_name, lora_rank = 8, lora = True, sim_name = 'sim_name', round = 1):

    """
    Initialize model and tokenizer from Hugging Face model name
    If lora is True, apply LoRA to the model, else will fine-tune full model
    """
    
    if ('bert' in model_name) or ('Bert' in model_name) or ('BERT' in model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

    if lora:
        lora_config = LoraConfig(
                    r = lora_rank,
                    lora_alpha= lora_rank*2,
                    lora_dropout=0.1,
                )
        
        model = get_peft_model(model, lora_config)

        #save model
    model.save_pretrained(f"./fl-results/{sim_name}/round_0/global_model")

    return model, tokenizer

def load_dataset(path, nrows = 100):
    
    """
    Load huggingface dataset in path
    """
    dataset = DatasetDict.load_from_disk(path, keep_in_memory=True)

    if nrows:
        dataset["train"] = dataset["train"].select(list(range(nrows)))
    
    #create labels
    #dataset["train"] = dataset["train"].map(lambda x: {"labels": x["input_ids"]}, batched=True)
    
    return dataset

def split_data(dataset, num_clients, sim_name, eval_split = 0.1):
    """
    Split dataset into num_clients for federated learning.
    Returns a tuple of lists containing the training and evaluation datasets for each client.
    """
    indices = list(range(len(dataset['train']['text'])))

    clients_indices = []
    for i in range(num_clients):
        client_indices = indices[i::num_clients]
        clients_indices.append(client_indices)

    clients_datasets_train = []
    clients_datasets_eval = []

    for i in range(num_clients):
        client_data_indices = clients_indices[i]

        eval_size = int(len(client_data_indices) * eval_split)
        train_indices = client_data_indices[eval_size:]
        eval_indices = client_data_indices[:eval_size]

        client_dataset_train = dataset['train'].select(train_indices)
        client_dataset_eval = dataset['train'].select(eval_indices)

        clients_datasets_train.append(client_dataset_train)
        clients_datasets_eval.append(client_dataset_eval)
    
    # save datasets
    for i in range(num_clients):
        clients_datasets_train[i].save_to_disk(f"./fl-results/{sim_name}/round_0/client_{i}")

    return clients_datasets_train, clients_datasets_eval


def train_client(client, client_dataset, round, sim_name, tokenizer, 
                 epochs=1, batch_size=2, max_steps=100, lr = 2e-3,
                 model_name = 'HuggingFaceTB/SmolLM-360M', lora = True):
    
    """
    Train model on client dataset
    """

    print("entro a esta funcon")
    client_dataset = client_dataset.shuffle(seed=round)

    quantization_config = {
        'load_in_4bit': True,
        'bnb_4bit_quant_type': 'nf4',  # nested float 4 for better accuracy
        'bnb_4bit_compute_dtype': torch.float16,
        'bnb_4bit_use_double_quant': True,  # nested quantization for further memory savings
    }

    if ('bert' in model_name) or ('Bert' in model_name) or ('BERT' in model_name):
        if lora:
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model = PeftModel.from_pretrained(model, f'fl-results/{sim_name}/round_{round-1}/global_model', is_trainable=True)
        else:
            model = AutoModelForMaskedLM.from_pretrained(f'fl-results/{sim_name}/round_{round-1}/global_model')
    
    else:
        if lora:
            model = AutoModelForCausalLM.from_pretrained(model_name,
            torch_dtype=torch.float16,
            **{k: v for k, v in quantization_config.items() 
                if k != 'bnb_4bit_compute_dtype'}  # exclude dtype from kwargs
            )
            model = PeftModel.from_pretrained(model, f'fl-results/{sim_name}/round_{round-1}/global_model', is_trainable=True,)
        else:
            model = AutoModelForCausalLM.from_pretrained(f'fl-results/{sim_name}/round_{round-1}/global_model'                                          
            )
    
    if ('bert' in model_name) or ('Bert' in model_name) or ('BERT' in model_name):
        data_collator = DataCollatorForLanguageModeling(
                        tokenizer=tokenizer,
                        mlm=True,
                        mlm_probability=0.15,  # 15% tokens will be masked
                    )
        
    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./fl-results",
        logging_dir="./logs",
        logging_steps=max_steps+11,
        learning_rate=lr,
        weight_decay=0.01,
        max_steps=max_steps,
        num_train_epochs=epochs,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=max_steps+1,
        fp16=True,  # Enable mixed precision training
        optim='paged_adamw_8bit',  # Use 8-bit optimizer for memory efficiency,
        #per_device_train_batch_size=8,
        #gradient_accumulation_steps=batch_size // 8,
        lr_scheduler_type = 'constant')
    

    class CustomTrainer(Trainer):
        def __init__(self, client, **kwargs):
            super().__init__(**kwargs)
            self.train_losses = {}
            self.validation_losses = {}
            self.client = client
    
        def log(self, logs):
            # Save client losses
            super().log(logs)
            if "loss" in logs:
                self.train_losses[client] = float(logs["loss"])
                print("log losswseesss:: ",self.train_losses[client])
            if "eval_loss" in logs:
                self.validation_losses[client] = float(logs["eval_loss"])


    if ('bert' in model_name) or ('Bert' in model_name) or ('BERT' in model_name):
        trainer = CustomTrainer(client=client,
                                model=model,
                                args=training_args,
                                train_dataset=client_dataset,
                                tokenizer=tokenizer,
                                eval_dataset=client_dataset,
                                data_collator=data_collator)
    else:
        trainer = CustomTrainer(client=client,
                                model=model,
                                args=training_args,
                                train_dataset=client_dataset,
                                tokenizer=tokenizer,
                                eval_dataset=client_dataset)
    
    trainer.train()

    # Save model
    output_dir = f"./fl-results/{sim_name}/round_{round}/client_{client}"
    os.makedirs(output_dir, exist_ok=True)
    #model.save_pretrained(output_dir)

    # Save losses
    print("Training losses here: ", trainer.train_losses)
    
    with open(f"{output_dir}/training_losses.json", "w") as f:
        json.dump(trainer.train_losses, f)
    
    with open(f"{output_dir}/validation_losses.json", "w") as f:
        json.dump(trainer.validation_losses, f)
    
    return model

def get_adapters(model):
    """
    Extract LoRA adapter weights from the model.
    Assumes that LoRA layers are identifiable by specific names or attributes.
    """
    adapters = {}
    for name, param in model.named_parameters():
        if "lora_" in name:  # Assume LoRA layers are named with "lora_"
            adapters[name] = param.data.clone()  # Clone to avoid modifying the original
    return adapters

def set_adapters(model, aggregated_adapters):
    """
    Update the model with aggregated LoRA adapter weights.
    """
    for name, param in model.named_parameters():
        if name in aggregated_adapters:
            param.data.copy_(aggregated_adapters[name])  # Update the parameter
    
def aggregate_models(models, lora=True):
    """
    Aggregate models by the mean (FedAvg).
    If LoRA, aggregate only the adapters.
    """
    if lora:
        all_adapters = [get_adapters(model) for model in models]
        aggregated_adapters = {}
        
        for key in all_adapters[0].keys():
            aggregated_adapters[key] = torch.mean(
                torch.stack([adapters[key] for adapters in all_adapters]), dim=0
            )
        
        return aggregated_adapters
    
    else:
        # Aggregate the entire model (not just adapters)
        global_model = models[0]
        for name, param in global_model.named_parameters():
            param.data.copy_(torch.mean(
                torch.stack([model.state_dict()[name].data for model in models]), dim=0
            ))
        return global_model

def save_global_model(global_model, round, sim_name):
    output_dir = f"./fl-results/{sim_name}/round_{round}/global_model"
    os.makedirs(output_dir, exist_ok=True)
    global_model.save_pretrained(output_dir)
    return global_model


def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=1e-5):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr