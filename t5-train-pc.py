import neptune.new as neptune
from transformers.integrations import NeptuneCallback
import os
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
from pprint import pprint
import torch
import matplotlib.pyplot as plt
import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from pytorch_lightning import seed_everything
import pickle as pkl
from scipy.stats import spearmanr, pearsonr
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoModel, GPT2LMHeadModel, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments, AdamW, get_scheduler
import pandas as pd
from torch.utils.data import Dataset, random_split, Subset
from datasets import load_dataset, Dataset, load_metric
import random
from torch import cuda
import os
import numpy as np
# os.chdir("/content/drive/MyDrive/iitkgp-mtp-dialog-response-generation")
RANDOM_SEED = 50
seed_everything(RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

class PromptsDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            # Encode the descriptions using the GPT-Neo tokenizer
            encodings_dict = tokenizer("<|startoftext|>" 
                                        + txt +    
                                        "<|endoftext|>",
                                        truncation=True,
                                        max_length=max_length, 
                                        padding='max_length')
            input_ids = torch.tensor(encodings_dict['input_ids'])    
            self.input_ids.append(input_ids)
            mask = torch.tensor(encodings_dict['attention_mask'])
            self.attn_masks.append(mask)
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
    
running = "pc"

for j in range(1, 5):
    
    with open(f"data/{running}-split{j}-train-tcpc.pkl", "rb") as f1:
        train_set = pkl.load(f1)
    with open(f"data/{running}-split{j}-val.pkl", "rb") as f1:
        val_set = pkl.load(f1)
    with open(f"data/{running}-split{j}-test.pkl", "rb") as f1:
        test_set = pkl.load(f1)
        
        
    label2int = {
        "understandable": {"no": 0, "yes": 1},
        "natural" : {"no": 1, "somewhat": 2, "yes": 3},
        "context": {"no": 1, "somewhat": 2, "yes": 3},
        "interesting": {"dull": 1, "somewhat interesting": 2, "interesting": 3},
        "facts": {"no": 0, "yes": 1},
        "overall": {"very bad": 1, "bad": 2, "neutral": 3, "good": 4, "very good": 5},
    }

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large",
                                bos_token="<|startoftext|>",
                                eos_token="<|endoftext|>",
                                pad_token="<|pad|>")

    train_lines = []
    train_labels = []
    train_labels_words = []
    for i, each in enumerate(train_set):
        datapoint = gpt2_tokenizer.decode(train_set[i][0], skip_special_tokens=True)
        unlabelled_here = ".".join(datapoint.split(".")[:-1]) + ". "
        train_labels_words.append(datapoint.replace(unlabelled_here, ""))
        try:
            if "Is the response dull or interesting" in unlabelled_here:
                label_int = label2int["interesting"][datapoint.replace(unlabelled_here, "")]
            elif "overall impression of the quality" in unlabelled_here:
                label_int = label2int["overall"][datapoint.replace(unlabelled_here, "")]
            elif "facts that the response is conditioned" in unlabelled_here:
                label_int = label2int["facts"][datapoint.replace(unlabelled_here, "")]
            elif "understandable given the previous context" in unlabelled_here:
                label_int = label2int["understandable"][datapoint.replace(unlabelled_here, "")]
            elif "valid continuation of the preceding conversation" in unlabelled_here:
                label_int = label2int["context"][datapoint.replace(unlabelled_here, "")]
            elif "something that a person would naturally say" in unlabelled_here:
                label_int = label2int["natural"][datapoint.replace(unlabelled_here, "")]
            train_lines.append(unlabelled_here)
            train_labels.append(label_int)
        except:
            print("error")

    val_lines = []
    val_labels = []
    val_labels_words = []
    for i, each in enumerate(val_set):
        datapoint = gpt2_tokenizer.decode(val_set[i][0], skip_special_tokens=True)
        unlabelled_here = ".".join(datapoint.split(".")[:-1]) + ". "
        val_labels_words.append(datapoint.replace(unlabelled_here, ""))
        try:
            if "Is the response dull or interesting" in unlabelled_here:
                label_int = label2int["interesting"][datapoint.replace(unlabelled_here, "")]
            elif "overall impression of the quality" in unlabelled_here:
                label_int = label2int["overall"][datapoint.replace(unlabelled_here, "")]
            elif "facts that the response is conditioned" in unlabelled_here:
                label_int = label2int["facts"][datapoint.replace(unlabelled_here, "")]
            elif "understandable given the previous context" in unlabelled_here:
                label_int = label2int["understandable"][datapoint.replace(unlabelled_here, "")]
            elif "valid continuation of the preceding conversation" in unlabelled_here:
                label_int = label2int["context"][datapoint.replace(unlabelled_here, "")]
            elif "something that a person would naturally say" in unlabelled_here:
                label_int = label2int["natural"][datapoint.replace(unlabelled_here, "")]
            val_lines.append(unlabelled_here)
            val_labels.append(label_int)
        except:
            print("error")

    test_lines = []
    test_labels = []
    test_labels_words = []
    for i, each in enumerate(test_set):
        datapoint = gpt2_tokenizer.decode(test_set[i][0], skip_special_tokens=True)
        unlabelled_here = ".".join(datapoint.split(".")[:-1]) + ". "
        test_labels_words.append(datapoint.replace(unlabelled_here, ""))
        try:
            if "Is the response dull or interesting" in unlabelled_here:
                label_int = label2int["interesting"][datapoint.replace(unlabelled_here, "")]
            elif "overall impression of the quality" in unlabelled_here:
                label_int = label2int["overall"][datapoint.replace(unlabelled_here, "")]
            elif "facts that the response is conditioned" in unlabelled_here:
                label_int = label2int["facts"][datapoint.replace(unlabelled_here, "")]
            elif "understandable given the previous context" in unlabelled_here:
                label_int = label2int["understandable"][datapoint.replace(unlabelled_here, "")]
            elif "valid continuation of the preceding conversation" in unlabelled_here:
                label_int = label2int["context"][datapoint.replace(unlabelled_here, "")]
            elif "something that a person would naturally say" in unlabelled_here:
                label_int = label2int["natural"][datapoint.replace(unlabelled_here, "")]
            test_lines.append(unlabelled_here)
            test_labels.append(label_int)
        except:
            print("error")

    def train_data_gen():
        for i in range(len(train_lines)):
            yield {"context_and_question": train_lines[i], "answer": train_labels_words[i]}

    def val_data_gen():
        for i in range(len(val_lines)):
            yield {"context_and_question": val_lines[i], "answer": val_labels_words[i]}

    def test_data_gen():
        for i in range(len(test_lines)):
            yield {"context_and_question": test_lines[i], "answer": test_labels_words[i]}


    train_dataset = Dataset.from_generator(train_data_gen)
    val_dataset = Dataset.from_generator(val_data_gen)
    test_dataset = Dataset.from_generator(test_data_gen)

    train_dataset.cleanup_cache_files()
    val_dataset.cleanup_cache_files()
    test_dataset.cleanup_cache_files()

    train_dataset = Dataset.from_generator(train_data_gen)
    val_dataset = Dataset.from_generator(val_data_gen)
    test_dataset = Dataset.from_generator(test_data_gen)

    model_checkpoint = "t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=2000)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

    prefix = "analyze the following dialog context and answer the subsequent question on the response: "
    max_input_length = 2000
    max_target_length = 8

    def preprocess(datapoints):
        inputs = [prefix + context_and_question for context_and_question in datapoints["context_and_question"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(datapoints["answer"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_dataset.map(preprocess, batched=True)
    tokenized_val = val_dataset.map(preprocess, batched=True)
    tokenized_test = test_dataset.map(preprocess, batched=True)

    lr = 2e-5
    bs = 4
    ne = 10

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = ne * (len(tokenized_train) / bs) * 10
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        output_dir=f"{running}-{j}-{model_name}-finetuned-10-epochs-{lr}-lr",
        num_train_epochs=ne,
    #     evaluation_strategy = "steps",
    #     eval_steps=1000,
    #     save_steps=1000,
        learning_rate=lr,
        save_strategy="epoch",
        logging_first_step=True,
        logging_dir=f"{running}-{j}-{model_name}-finetuned-10-epochs-{lr}-lr/LOGS/",
        per_device_train_batch_size=bs,
    #     per_device_eval_batch_size=batch_size,
    #     auto_find_batch_size=True,
    #     weight_decay=0.01,
    #     save_total_limit=3,
    #     num_train_epochs=10,
    #     predict_with_generate=True,
        fp16=False,
        report_to="none",
    #     push_to_hub=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train,
#         eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    )

    trainer.train()

