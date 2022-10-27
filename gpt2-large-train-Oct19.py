#!/usr/bin/python

PARAMS = {
    "model_checkpoint": "gpt2-xl",
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "overwrite_output_dir": False,
}
PARAMS["output_dir"] = f"{PARAMS['model_checkpoint']}-finetuned-{PARAMS['num_epochs']}-epochs-{PARAMS['learning_rate']}-lr-no-weight-decay-scheduler"

print(f"Output Directory = {PARAMS['output_dir']}")

import neptune.new as neptune

# neptune.init(project='divyanshusheth/dialog-systems-evaluation', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1N2U3ZjlmNi1iYTFiLTQxZTctYWQ4ZC1iYzFhZDE5M2NmN2MifQ==')
neptune.init()


from transformers.integrations import NeptuneCallback
from transformers import AdamW
from transformers import get_scheduler
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
# os.chdir("/content/drive/MyDrive/iitkgp-mtp-dialog-response-generation")
RANDOM_SEED = 50
seed_everything(RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("tc_all_prompts.pkl", "rb") as f1:
    tc_all_prompts = pkl.load(f1)

with open("tc_all_labels.pkl", "rb") as f1:
    tc_all_labels = pkl.load(f1)

with open("pc_all_prompts.pkl", "rb") as f1:
    pc_all_prompts = pkl.load(f1)

with open("pc_all_labels.pkl", "rb") as f1:
    pc_all_labels = pkl.load(f1)

pc_all_prompts_labelled = [-1] * len(pc_all_prompts)
tc_all_prompts_labelled = [-1] * len(tc_all_prompts)
pc_all_prompts_labelled_understandable = [-1] * len(pc_all_prompts)
tc_all_prompts_labelled_understandable = [-1] * len(tc_all_prompts)
pc_all_prompts_labelled_natural = [-1] * len(pc_all_prompts)
tc_all_prompts_labelled_natural = [-1] * len(tc_all_prompts)
pc_all_prompts_labelled_context = [-1] * len(pc_all_prompts)
tc_all_prompts_labelled_context = [-1] * len(tc_all_prompts)
pc_all_prompts_labelled_interesting = [-1] * len(pc_all_prompts)
tc_all_prompts_labelled_interesting = [-1] * len(tc_all_prompts)
pc_all_prompts_labelled_facts = [-1] * len(pc_all_prompts)
tc_all_prompts_labelled_facts = [-1] * len(tc_all_prompts)
pc_all_prompts_labelled_overall = [-1] * len(pc_all_prompts)
tc_all_prompts_labelled_overall = [-1] * len(tc_all_prompts)

int2word = {
    "understandable": {0: "no", 1: "yes"},
    "natural": {1: "no", 2: "somewhat", 3: "yes"},
    "context": {1: "no", 2: "somewhat", 3: "yes"},
    "interesting": {1: "dull", 2: "somewhat interesting", 3: "interesting"},
    "facts": {0: "no", 1: "yes"},
    "overall": {1: "very bad", 2: "bad", 3: "neutral", 4: "good", 5: "very good"}
}
change_int_to_word = True

def replace_in_q(q):
    q = q.replace("Interesting (1 - 3)", "Interesting (dull/somewhat interesting/interesting)")\
        .replace("(1 - 3)", "(no/somewhat/yes)")\
        .replace("(1 - 5)", "(very bad/bad/neutral/good/very good)")\
        .replace("(0 - 1)", "(no/yes)")
    return q

for i, each in enumerate(pc_all_prompts):
    prompt_here = pc_all_prompts[i]
    label_1 = round(np.mean(np.array(pc_all_labels[i][0])))
    label_2 = round(np.mean(np.array(pc_all_labels[i][1])))
    label_3 = round(np.mean(np.array(pc_all_labels[i][2])))
    label_4 = round(np.mean(np.array(pc_all_labels[i][3])))
    label_5 = round(np.mean(np.array(pc_all_labels[i][4])))
    label_6 = round(np.mean(np.array(pc_all_labels[i][5])))
    # all_prompts[i] = prefix + each + f"\n1. Understandable: {str(label_1)}\n2. Natural: {str(label_2)}\n3. Maintains Context: {str(label_3)}\n4. Interesting: {str(label_4)}\n5. Overall Quality: {str(label_5)}"
    # pc_all_prompts_labelled[i] = each.lstrip() + f"\n1. Understandable: {str(label_1)}\n2. Natural: {str(label_2)}\n3. Maintains Context: {str(label_3)}\n4. Interesting: {str(label_4)}\n5. Uses Knowledge: {str(label_5)}\n6. Overall Quality: {str(label_6)}"
    before_questions = each.lstrip()[:each.find("\n1.")]
    before_questions_nofacts = before_questions[:before_questions.find("Facts:")] + before_questions[before_questions.find("Generated "):]
    if change_int_to_word:
        pc_all_prompts_labelled[i] = replace_in_q(each.lstrip() + f"""\n1. {str(int2word["understandable"][label_1])}\n2. {str(int2word["natural"][label_2])}\n3. {str(int2word["context"][label_3])}\n4. {str(int2word["interesting"][label_4])}\n5. {str(int2word["facts"][label_5])}\n6. {str(int2word["overall"][label_6])}""")
        pc_all_prompts_labelled_understandable[i] = replace_in_q(before_questions_nofacts + each.lstrip()[each.find("\n1."):each.find("\n2.")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["understandable"][label_1])}""")
        pc_all_prompts_labelled_natural[i] = replace_in_q(before_questions_nofacts + each.lstrip()[each.find("\n2."):each.find("\n3.")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["natural"][label_2])}""")
        pc_all_prompts_labelled_context[i] = replace_in_q(before_questions_nofacts + each.lstrip()[each.find("\n3."):each.find("\n4.")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["context"][label_3])}""")
        pc_all_prompts_labelled_interesting[i] = replace_in_q(before_questions_nofacts + each.lstrip()[each.find("\n4."):each.find("\n5.")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["interesting"][label_4])}""")
        pc_all_prompts_labelled_facts[i] = replace_in_q(before_questions + each.lstrip()[each.find("\n5."):each.find("\n6.")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["facts"][label_5])}""")
        pc_all_prompts_labelled_overall[i] = replace_in_q(before_questions + each.lstrip()[each.find("\n6."):each.find("\n\nAnswers")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["overall"][label_6])}""")
    else:
        pc_all_prompts_labelled[i] = each.lstrip() + f"""\n1. {str(label_1)}\n2. {str(label_2)}\n3. {str(label_3)}\n4. {str(label_4)}\n5. {str(label_5)}\n6. {str(label_6)}"""
        pc_all_prompts_labelled_understandable[i] = each.lstrip()[:each.find("\n2.")][:-3] + f"""\n\nAnswers: \n1. {str(label_1)}"""
        pc_all_prompts_labelled_natural[i] = before_questions_nofacts + each.lstrip()[each.find("\n2."):each.find("\n3.")][:-3] + f"""\n\nAnswers: \n1. {str(label_2)}"""
        pc_all_prompts_labelled_context[i] = before_questions_nofacts + each.lstrip()[each.find("\n3."):each.find("\n4.")][:-3] + f"""\n\nAnswers: \n1. {str(label_3)}"""
        pc_all_prompts_labelled_interesting[i] = before_questions_nofacts + each.lstrip()[each.find("\n4."):each.find("\n5.")][:-3] + f"""\n\nAnswers: \n1. {str(label_4)}"""
        pc_all_prompts_labelled_facts[i] = before_questions + each.lstrip()[each.find("\n5."):each.find("\n6.")][:-3] + f"""\n\nAnswers: \n1. {str(label_5)}"""
        pc_all_prompts_labelled_overall[i] = before_questions + each.lstrip()[each.find("\n6."):each.find("\n\nAnswers")][:-3] + f"""\n\nAnswers: \n1. {str(label_6)}"""

for i, each in enumerate(tc_all_prompts):
    prompt_here = tc_all_prompts[i]
    label_1 = round(np.mean(np.array(tc_all_labels[i][0])))
    label_2 = round(np.mean(np.array(tc_all_labels[i][1])))
    label_3 = round(np.mean(np.array(tc_all_labels[i][2])))
    label_4 = round(np.mean(np.array(tc_all_labels[i][3])))
    label_5 = round(np.mean(np.array(tc_all_labels[i][4])))
    label_6 = round(np.mean(np.array(tc_all_labels[i][5])))
    before_questions = each.lstrip()[:each.find("\n1.")]
    before_questions_nofacts = before_questions[:before_questions.find("Facts:")] + before_questions[before_questions.find("Generated "):]
    if change_int_to_word:
        tc_all_prompts_labelled[i] = replace_in_q(each.lstrip() + f"""\n1. {str(int2word["understandable"][label_1])}\n2. {str(int2word["natural"][label_2])}\n3. {str(int2word["context"][label_3])}\n4. {str(int2word["interesting"][label_4])}\n5. {str(int2word["facts"][label_5])}\n6. {str(int2word["overall"][label_6])}""")
        tc_all_prompts_labelled_understandable[i] = replace_in_q(before_questions_nofacts + each.lstrip()[each.find("\n1."):each.find("\n2.")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["understandable"][label_1])}""")
        tc_all_prompts_labelled_natural[i] = replace_in_q(before_questions_nofacts + each.lstrip()[each.find("\n2."):each.find("\n3.")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["natural"][label_2])}""")
        tc_all_prompts_labelled_context[i] = replace_in_q(before_questions_nofacts + each.lstrip()[each.find("\n3."):each.find("\n4.")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["context"][label_3])}""")
        tc_all_prompts_labelled_interesting[i] = replace_in_q(before_questions_nofacts + each.lstrip()[each.find("\n4."):each.find("\n5.")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["interesting"][label_4])}""")
        tc_all_prompts_labelled_facts[i] = replace_in_q(before_questions + each.lstrip()[each.find("\n5."):each.find("\n6.")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["facts"][label_5])}""")
        tc_all_prompts_labelled_overall[i] = replace_in_q(before_questions + each.lstrip()[each.find("\n6."):each.find("\n\nAnswers")][:-3] + f"""\n\nAnswers: \n1. {str(int2word["overall"][label_6])}""")
    else:
        tc_all_prompts_labelled[i] = each.lstrip() + f"""\n1. {str(label_1)}\n2. {str(label_2)}\n3. {str(label_3)}\n4. {str(label_4)}\n5. {str(label_5)}\n6. {str(label_6)}"""
        tc_all_prompts_labelled_understandable[i] = each.lstrip()[:each.find("\n2.")][:-3] + f"""\n\nAnswers: \n1. {str(label_1)}"""
        tc_all_prompts_labelled_natural[i] = before_questions_nofacts + each.lstrip()[each.find("\n2."):each.find("\n3.")][:-3] + f"""\n\nAnswers: \n1. {str(label_2)}"""
        tc_all_prompts_labelled_context[i] = before_questions_nofacts + each.lstrip()[each.find("\n3."):each.find("\n4.")][:-3] + f"""\n\nAnswers: \n1. {str(label_3)}"""
        tc_all_prompts_labelled_interesting[i] = before_questions_nofacts + each.lstrip()[each.find("\n4."):each.find("\n5.")][:-3] + f"""\n\nAnswers: \n1. {str(label_4)}"""
        tc_all_prompts_labelled_facts[i] = before_questions + each.lstrip()[each.find("\n5."):each.find("\n6.")][:-3] + f"""\n\nAnswers: \n1. {str(label_5)}"""
        tc_all_prompts_labelled_overall[i] = before_questions + each.lstrip()[each.find("\n6."):each.find("\n\nAnswers")][:-3] + f"""\n\nAnswers: \n1. {str(label_6)}"""

remove_question_number = True
# if remove_question_number:
for i, each in enumerate(pc_all_prompts_labelled_overall):
    pc_all_prompts_labelled_overall[i] = each.replace("Given your answers above, w", "W")
for i, each in enumerate(tc_all_prompts_labelled_overall):
    tc_all_prompts_labelled_overall[i] = each.replace("Given your answers above, w", "W")

for i, each in enumerate(pc_all_prompts_labelled_facts):
    pc_all_prompts_labelled_facts[i] = each.replace("response is conditioned", "response is supposed to be conditioned")
    pc_all_prompts_labelled_facts[i] = each.replace("how well does the", "does the")
for i, each in enumerate(tc_all_prompts_labelled_facts):
    tc_all_prompts_labelled_facts[i] = each.replace("response is conditioned", "response is supposed to be conditioned")
    tc_all_prompts_labelled_facts[i] = each.replace("how well does the", "does the")

# print(pc_all_prompts_labelled[0])

# print(pc_all_prompts_labelled_understandable[0])

# print(pc_all_prompts_labelled_natural[0])

# print(pc_all_prompts_labelled_context[0])

# print(pc_all_prompts_labelled_interesting[0])

# print(pc_all_prompts_labelled_facts[0])

# print(pc_all_prompts_labelled_overall[0])

import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoModel, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import pandas as pd
from torch.utils.data import Dataset, random_split, Subset
import random
from torch import cuda
import os
import numpy as np

# model_checkpoint = "gpt2-medium"
model_checkpoint = PARAMS["model_checkpoint"]
# model_checkpoint = "bigscience/bloom-560m"
# model_checkpoint = "bigscience/bloom-1b1"

# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint,
                            bos_token="<|startoftext|>",
                            eos_token="<|endoftext|>",
                            pad_token="<|pad|>")


model = GPT2LMHeadModel.from_pretrained(model_checkpoint).to(device)
# model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").cuda()
# Resize the token embeddings because we've just added 3 new tokens 
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id


tc_lines = []
for a, b, c, d, e, f in zip(tc_all_prompts_labelled_understandable, tc_all_prompts_labelled_natural, tc_all_prompts_labelled_context, tc_all_prompts_labelled_interesting, tc_all_prompts_labelled_facts, tc_all_prompts_labelled_overall):
    tc_lines.append(a)
    tc_lines.append(b)
    tc_lines.append(c)
    tc_lines.append(d)
    tc_lines.append(e)
    tc_lines.append(f)

pc_lines = []
for a, b, c, d, e, f in zip(pc_all_prompts_labelled_understandable, pc_all_prompts_labelled_natural, pc_all_prompts_labelled_context, pc_all_prompts_labelled_interesting, pc_all_prompts_labelled_facts, pc_all_prompts_labelled_overall):
    pc_lines.append(a)
    pc_lines.append(b)
    pc_lines.append(c)
    pc_lines.append(d)
    pc_lines.append(e)
    pc_lines.append(f)

max_length = 512
tc_descriptions = [description for description in tc_lines if len(tokenizer.encode(description)) < max_length-2]
pc_descriptions = [description for description in pc_lines if len(tokenizer.encode(description)) < max_length-2]
pc_max_length = max([len(tokenizer.encode(description)) for description in pc_descriptions])
tc_max_length = max([len(tokenizer.encode(description)) for description in tc_descriptions])
# print(pc_max_length)
# print(len(pc_descriptions))
# print(tc_max_length)
# print(len(tc_descriptions))


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

tc_dataset = PromptsDataset(tc_descriptions, tokenizer, max_length) 
pc_dataset = PromptsDataset(pc_descriptions, tokenizer, max_length)



# tc_train_size = int(0.7 * len(tc_dataset))
# pc_train_size = int(0.7 * len(pc_dataset))

tc_train_size = 1512
pc_train_size = 1260

tc_val_size = 288
pc_val_size = 270

tc_test_size = 312 
pc_test_size = 270 

tc_train_dataset = Subset(tc_dataset, range(tc_train_size))
pc_train_dataset = Subset(pc_dataset, range(pc_train_size))

tc_val_dataset = Subset(tc_dataset, range(tc_train_size, tc_train_size + tc_val_size))
pc_val_dataset = Subset(pc_dataset, range(pc_train_size, pc_train_size + pc_val_size))

tc_test_dataset = Subset(tc_dataset, range(tc_train_size + tc_val_size, tc_train_size + tc_val_size + tc_test_size))
pc_test_dataset = Subset(pc_dataset, range(pc_train_size + pc_val_size, pc_train_size + pc_val_size + pc_test_size))


int2word = {
    "understandable": {0: "no", 1: "yes"},
    "natural": {1: "no", 2: "somewhat", 3: "yes"},
    "context": {1: "no", 2: "somewhat", 3: "yes"},
    "interesting": {1: "dull", 2: "somewhat interesting", 3: "interesting"},
    "facts": {0: "no", 1: "yes"},
    "overall": {1: "very bad", 2: "bad", 3: "neutral", 4: "good", 5: "very good"}
}

label2int = {
    "understandable": {"no": 0, "yes": 1},
    "natural" : {"no": 1, "somewhat": 2, "yes": 3},
    "context": {"no": 1, "somewhat": 2, "yes": 3},
    "interesting": {"dull": 1, "somewhat interesting": 2, "interesting": 3},
    "facts": {"no": 0, "yes": 1},
    "overall": {"very bad": 1, "bad": 2, "neutral": 3, "good": 4, "very good": 5},
}

tc_pc_concat_testdataset = torch.utils.data.ConcatDataset([tc_test_dataset, pc_test_dataset])
test_lines = []
test_labels = []
for i, each in enumerate(tc_pc_concat_testdataset):
    datapoint = tokenizer.decode(tc_pc_concat_testdataset[i][0], skip_special_tokens=True)
    unlabelled_here = ".".join(datapoint.split(".")[:-1]) + ". "
    test_lines.append(unlabelled_here)
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
    test_labels.append(label_int)


tc_pc_concat_traindataset = torch.utils.data.ConcatDataset([tc_train_dataset, pc_train_dataset])
tc_pc_concat_valdataset = torch.utils.data.ConcatDataset([tc_val_dataset, tc_val_dataset])


lr=PARAMS["learning_rate"]
# bs=10
ne=PARAMS["num_epochs"]
training_args = TrainingArguments(output_dir=PARAMS["output_dir"],
                                  overwrite_output_dir=PARAMS["overwrite_output_dir"],
                                  num_train_epochs=ne,
                                  learning_rate=lr,
                                  save_strategy='epoch',
                                  evaluation_strategy='epoch',
                                  logging_strategy="epoch",
                                  logging_first_step=True,
                                  logging_dir=f"""{PARAMS['output_dir']}/LOGS/""",
                                #   logging_steps=200,
                                #   save_steps =1000,
#                                   per_device_train_batch_size=1,
                                  auto_find_batch_size=True,
#                                   gradient_accumulation_steps=bs,
                                #   warmup_steps=500,
#                                   weight_decay=0.01,  
                                  load_best_model_at_end=True,
                                  fp16=False,
                                  report_to="neptune",
                                  )
                                #   logging_dir='./logs')

    

optimizer = AdamW(model.parameters(), lr=PARAMS['learning_rate'])
num_training_steps = PARAMS['num_epochs'] * 6930
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

trainer = Trainer(model=model, args=training_args,  
                  train_dataset=tc_pc_concat_traindataset,
                  eval_dataset=tc_pc_concat_valdataset, 
                  
                  optimizers=(optimizer, lr_scheduler),
                  
                  # This custom collate function is necessary 
                  # to built batches of data
                  data_collator=lambda data: 
              {'input_ids': torch.stack([f[0] for f in data]),       
               'attention_mask': torch.stack([f[1] for f in data]),
               'labels': torch.stack([f[0] for f in data])},
                 )

# print("train dataloader len = ")
# print(len(trainer.get_train_dataloader()))

# Start training process!
trainer.train()
