#!/usr/bin/python
model_checkpoint = f'gpt2-large-finetuned-5-epochs-2.5e-06-lr/checkpoint-2079/'
print(f"Model Checkpoint = {model_checkpoint}\n")

import neptune.new as neptune
from transformers.integrations import NeptuneCallback
import os
import math
import pandas as pd
import numpy as np
from pprint import pprint
import torch
import matplotlib.pyplot as plt
import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from pytorch_lightning import seed_everything
import pickle as pkl
from scipy.stats import spearmanr, pearsonr
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoModel, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import pandas as pd
from torch.utils.data import Dataset, random_split, Subset
import random
from torch import cuda
import os
import numpy as np
# os.chdir("/content/drive/MyDrive/iitkgp-mtp-dialog-response-generation")
RANDOM_SEED = 50
seed_everything(RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large",
                            bos_token="<|startoftext|>",
                            eos_token="<|endoftext|>",
                            pad_token="<|pad|>")
model = GPT2LMHeadModel.from_pretrained(model_checkpoint).to(device)
# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").cuda()
# Resize the token embeddings because we've just added 3 new tokens 
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

label2int = {
    "understandable": {"no": 0, "yes": 1},
    "natural" : {"no": 1, "somewhat": 2, "yes": 3},
    "context": {"no": 1, "somewhat": 2, "yes": 3},
    "interesting": {"dull": 1, "somewhat interesting": 2, "interesting": 3},
    "facts": {"no": 0, "yes": 1},
    "overall": {"very bad": 1, "bad": 2, "neutral": 3, "good": 4, "very good": 5},
}
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

with open("tc_train_dataset-Oct20.pkl", "rb") as f1:
    tc_train_dataset = pkl.load(f1)
with open("tc_val_dataset-Oct20.pkl", "rb") as f1:
    tc_val_dataset = pkl.load(f1)
with open("tc_test_dataset-Oct20.pkl", "rb") as f1:
    tc_test_dataset = pkl.load(f1)
with open("pc_train_dataset-Oct20.pkl", "rb") as f1:
    pc_train_dataset = pkl.load(f1)
with open("pc_val_dataset-Oct20.pkl", "rb") as f1:
    pc_val_dataset = pkl.load(f1)
with open("pc_test_dataset-Oct20.pkl", "rb") as f1:
    pc_test_dataset = pkl.load(f1)

tc_test_lines = []
tc_labels = []
for i, each in enumerate(tc_test_dataset):
    datapoint = tokenizer.decode(tc_test_dataset[i][0], skip_special_tokens=True)
    unlabelled_here = ".".join(datapoint.split(".")[:-1]) + ". "
    tc_test_lines.append(unlabelled_here)
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
    tc_labels.append(label_int)

pc_test_lines = []
pc_labels = []
for i, each in enumerate(pc_test_dataset):
    datapoint = tokenizer.decode(pc_test_dataset[i][0], skip_special_tokens=True)
    unlabelled_here = ".".join(datapoint.split(".")[:-1]) + ". "
    pc_test_lines.append(unlabelled_here)
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
    pc_labels.append(label_int)
    
def get_results(test_lines, test_labels):
    predicted_q1 = []
    predicted_q2 = []
    predicted_q3 = []
    predicted_q4 = []
    predicted_q5 = []
    predicted_q6 = []
    labels_q1 = []
    labels_q2 = []
    labels_q3 = []
    labels_q4 = []
    labels_q5 = []
    labels_q6 = []

    for i, each in enumerate(test_lines):
#         print(i)
        tokenized = tokenizer(each)
        input_ids = torch.tensor([tokenized.input_ids]).to(device)
        unlabelled_here = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if len(input_ids[0]) < 1000:
            generation = model.generate(input_ids, do_sample=False, max_new_tokens=100, num_beams=5)
            ## do_sample=False enables greedy decoding
            predicted_label = tokenizer.batch_decode(generation, skip_special_tokens=True)[0].replace(each, "").lstrip().rstrip()
            if "understandable given the previous context" in unlabelled_here:
                label_int = label2int["understandable"][predicted_label]
                predicted_q1.append(label_int)
                labels_q1.append(test_labels[i])
            elif "something that a person would naturally say" in unlabelled_here:
                label_int = label2int["natural"][predicted_label]
                predicted_q2.append(label_int)
                labels_q2.append(test_labels[i])
            elif "valid continuation of the preceding conversation" in unlabelled_here:
                label_int = label2int["context"][predicted_label]
                predicted_q3.append(label_int)
                labels_q3.append(test_labels[i])
            elif "Is the response dull or interesting" in unlabelled_here:
                label_int = label2int["interesting"][predicted_label]
                predicted_q4.append(label_int)
                labels_q4.append(test_labels[i])
            elif "facts that the response is conditioned" in unlabelled_here:
                label_int = label2int["facts"][predicted_label]
                predicted_q5.append(label_int)
                labels_q5.append(test_labels[i])
            elif "overall impression of the quality" in unlabelled_here:
                label_int = label2int["overall"][predicted_label]
                predicted_q6.append(label_int)
                labels_q6.append(test_labels[i])
    
    print("LENGTHS:\n")
    print(len(predicted_q1))
    print(len(predicted_q2))
    print(len(predicted_q3))
    print(len(predicted_q4))
    print(len(predicted_q5))
    print(len(predicted_q6))
    
    print("\n\nLABELS:\n")
    print(labels_q1, "\n")
    print(labels_q2, "\n")
    print(labels_q3, "\n")
    print(labels_q4, "\n")
    print(labels_q5, "\n")
    print(labels_q6, "\n")
    
    print("\n\nPREDICTIONS:\n")
    print(predicted_q1, "\n")
    print(predicted_q2, "\n")
    print(predicted_q3, "\n")
    print(predicted_q4, "\n")
    print(predicted_q5, "\n")
    print(predicted_q6, "\n")

print("\n\n######## TC ########\n\n")
get_results(tc_test_lines, tc_labels)

print("\n\n######## PC ########\n\n")
get_results(pc_test_lines, pc_labels)