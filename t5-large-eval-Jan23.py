#!/usr/bin/python
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
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from datasets import load_dataset
from pytorch_lightning import seed_everything
import pickle as pkl
from scipy.stats import spearmanr, pearsonr
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoModel, GPT2LMHeadModel
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
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
    
# j = 1
running = "tc"
for j in range(1, 4):
#     for each in [6928, 6062, 5196, 4330, 3464, 2598, 1732, 866]:
    for each in [5058, 4496, 3934, 3372, 2810, 2248, 1686, 1124, 562]:
        model_checkpoint = f'{running}-{j}-t5-large-finetuned-10-epochs-2e-05-lr/checkpoint-{each}'
        output_dir = model_checkpoint

        print(f"Model Checkpoint = {model_checkpoint}\n")

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large",
                                    bos_token="<|startoftext|>",
                                    eos_token="<|endoftext|>",
                                    pad_token="<|pad|>")
        # model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").cuda()
        # Resize the token embeddings because we've just added 3 new tokens 
        # model.resize_token_embeddings(len(tokenizer))
        # model.config.pad_token_id = tokenizer.pad_token_id
        # model.eval()

        label2int = {
            "understandable": {"no": 0, "yes": 1},
            "natural" : {"no": 1, "somewhat": 2, "yes": 3},
            "context": {"no": 1, "somewhat": 2, "yes": 3},
            "interesting": {"dull": 1, "somewhat interesting": 2, "interesting": 3},
            "facts": {"no": 0, "yes": 1},
            "overall": {"very bad": 1, "bad": 2, "neutral": 3, "good": 4, "very good": 5},
        }

        with open(f"data/{running}-split{j}-val.pkl", "rb") as f1:
            val_set = pkl.load(f1)

        with open(f"data/{running}-split{j}-test.pkl", "rb") as f1:
            test_set = pkl.load(f1)


        val_lines = []
        val_labels = []
        for i, each in enumerate(val_set):
            datapoint = gpt2_tokenizer.decode(val_set[i][0], skip_special_tokens=True)
            unlabelled_here = ".".join(datapoint.split(".")[:-1]) + ". "
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
        for i, each in enumerate(test_set):
            datapoint = gpt2_tokenizer.decode(test_set[i][0], skip_special_tokens=True)
            unlabelled_here = ".".join(datapoint.split(".")[:-1]) + ". "
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

        valtest_lines = val_lines + test_lines
        valtest_labels = val_labels + test_labels


        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=2000)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)
        model.eval()

        prefix = "analyze the following dialog context and answer the subsequent question on the response: "
        max_input_length = 2000
        max_target_length = 8


        def get_results(test_lines, test_labels, dataset_and_split):
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

            prefix = "analyze the following dialog context and answer the subsequent question on the response: "

            print(f"\nModel Checkpoint = {model_checkpoint}")
            print(f"Dataset and Split = {dataset_and_split}")
            print(f"Total = {len(test_lines)}")
            for i, each in tqdm(enumerate(test_lines)):
        #         tokenized = tokenizer(each)
                each_with_prefix = prefix + each
                tokenized = tokenizer(each_with_prefix)
                input_ids = torch.tensor([tokenized.input_ids]).to(device)
                unlabelled_here = each_with_prefix
                if len(input_ids) < 1000:
        #             generation = model.generate(input_ids, do_sample=False, max_new_tokens=100, num_beams=5)
                    ## do_sample=False enables greedy decoding
        #             predicted_label = tokenizer.batch_decode(generation, skip_special_tokens=True)[0].replace(each, "").lstrip().rstrip()
                    generation = model.generate(input_ids, do_sample=False, max_new_tokens=100, num_beams=5)
                    predicted_label = tokenizer.batch_decode(generation, skip_special_tokens=True)[0].lstrip().rstrip()
                    try:
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
                    except:
                        continue

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

            srho1, sp1 = spearmanr(predicted_q1, labels_q1)
            srho2, sp2 = spearmanr(predicted_q2, labels_q2)
            srho3, sp3 = spearmanr(predicted_q3, labels_q3)
            srho4, sp4 = spearmanr(predicted_q4, labels_q4)
            srho5, sp5 = spearmanr(predicted_q5, labels_q5)
            srho6, sp6 = spearmanr(predicted_q6, labels_q6)

            prho1, pp1 = pearsonr(predicted_q1, labels_q1)
            prho2, pp2 = pearsonr(predicted_q2, labels_q2)
            prho3, pp3 = pearsonr(predicted_q3, labels_q3)
            prho4, pp4 = pearsonr(predicted_q4, labels_q4)
            prho5, pp5 = pearsonr(predicted_q5, labels_q5)
            prho6, pp6 = pearsonr(predicted_q6, labels_q6)

            with open(os.path.join(output_dir, f"RESULTS_{dataset_and_split}.txt"), "w") as f1:
                f1.write(f"Model Checkpoint = {model_checkpoint}\n\n")
                f1.write(f"Spearmann Correlation = \n")
                f1.write(f"Q1: Rho = {srho1}, p = {sp1}\n")
                f1.write(f"Q2: Rho = {srho2}, p = {sp2}\n")
                f1.write(f"Q3: Rho = {srho3}, p = {sp3}\n")
                f1.write(f"Q4: Rho = {srho4}, p = {sp4}\n")
                f1.write(f"Q5: Rho = {srho5}, p = {sp5}\n")
                f1.write(f"Q6: Rho = {srho6}, p = {sp6}\n")
                f1.write(f"\n\nPearson Correlation = \n")
                f1.write(f"Q1: Rho = {prho1}, p = {pp1}\n")
                f1.write(f"Q2: Rho = {prho2}, p = {pp2}\n")
                f1.write(f"Q3: Rho = {prho3}, p = {pp3}\n")
                f1.write(f"Q4: Rho = {prho4}, p = {pp4}\n")
                f1.write(f"Q5: Rho = {prho5}, p = {pp5}\n")
                f1.write(f"Q6: Rho = {prho6}, p = {pp6}\n")

                f1.write("\n\nLengths:\n")
                f1.write(f"q1 = {len(predicted_q1)}\n")
                f1.write(f"q2 = {len(predicted_q2)}\n")
                f1.write(f"q3 = {len(predicted_q3)}\n")
                f1.write(f"q4 = {len(predicted_q4)}\n")
                f1.write(f"q5 = {len(predicted_q5)}\n")
                f1.write(f"q6 = {len(predicted_q6)}\n")

                f1.write("\n\nLabels:\n")
                f1.write(f"q1 = {labels_q1}\n\n")
                f1.write(f"q2 = {labels_q2}\n\n")
                f1.write(f"q3 = {labels_q3}\n\n")
                f1.write(f"q4 = {labels_q4}\n\n")
                f1.write(f"q5 = {labels_q5}\n\n")
                f1.write(f"q6 = {labels_q6}\n\n")

                f1.write("\n\nPredictions:\n")
                f1.write(f"q1 = {predicted_q1}\n\n")
                f1.write(f"q2 = {predicted_q2}\n\n")
                f1.write(f"q3 = {predicted_q3}\n\n")
                f1.write(f"q4 = {predicted_q4}\n\n")
                f1.write(f"q5 = {predicted_q5}\n\n")
                f1.write(f"q6 = {predicted_q6}\n\n")


        get_results(valtest_lines, valtest_labels, dataset_and_split=f"{running}_valtest")

