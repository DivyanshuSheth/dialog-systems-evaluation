#!/usr/bin/env python
import random
import time
import numpy as np
import os
import ast
from pprint import pprint
import json
import pandas as pd
import argparse

import math
from tqdm import tqdm
import pandas as pd
import torch
from torch import cuda
from torch.utils.data import random_split, Subset
import matplotlib.pyplot as plt
import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from pytorch_lightning import seed_everything
from scipy.stats import spearmanr, pearsonr
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments, AdamW, get_scheduler
import pandas as pd
from datasets import load_dataset, Dataset, load_metric, disable_caching
RANDOM_SEED = 42
seed_everything(RANDOM_SEED)
disable_caching()
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_questions_mapping = {
    "tc_usr": {
        "Engaging": "Interesting (dull/somewhat interesting/interesting): Is the response dull or interesting?",
        "Maintains Context": "Maintains Context (no/somewhat/yes): Does the response serve as a valid continuation of the preceding conversation?",
        "Natural": "Natural (no/somewhat/yes): Does the response seem like something that a person would naturally say?",
        "Overall": "Overall Quality (very bad/bad/neutral/good/very good): What is your overall impression of the quality of the generated response?",
        "Understandable": "Understandable (no/yes): Is the response understandable given the previous context?",
        "Uses Knowledge": "Uses Knowledge (no/yes): Given the facts that the response is conditioned on, does the response use the facts?",
    },
    "pc_usr": {
        "Engaging": "Interesting (dull/somewhat interesting/interesting): Is the response dull or interesting?",
        "Maintains Context": "Maintains Context (no/somewhat/yes): Does the response serve as a valid continuation of the preceding conversation?",
        "Natural": "Natural (no/somewhat/yes): Does the response seem like something that a person would naturally say?",
        "Overall": "Overall Quality (very bad/bad/neutral/good/very good): What is your overall impression of the quality of the generated response?",
        "Understandable": "Understandable (no/yes): Is the response understandable given the previous context?",
        "Uses Knowledge": "Uses Knowledge (no/yes): Given the facts that the response is conditioned on, does the response use the facts?",
    },
    "fed": {
        "Interesting": "Interesting (no/somewhat/yes): To the average person, is the response interesting?",
        "Engaging": "Engaging (no/somewhat/yes): Is the response engaging?",
        "Specific": "Specific (no/somewhat/yes): Is the response generic or specific to the conversation?",
        "Relevant": "Relevant (no/somewhat/yes): Is the response relevant to the conversation?",
        "Correct": "Correct (no/somewhat/yes): Is the response correct or was there a misunderstanding of the conversation?",
        "Semantically appropriate": "Semantically appropriate (no/somewhat/yes): Is the response semantically appropriate?",
        "Understandable": "Understandable (no/yes): Is the response understandable?",
        "Fluent": "Fluent (no/somewhat/yes): Is the response fluently written?",
        "Coherent": "Coherent (no/somewhat/yes): Throughout the dialog, is B coherent and maintaining a good conversation flow?",
        "Error recovery": "Error Recovery (no/somewhat/yes): Is B able to recover from errors that it makes?",
        "Consistent": "Consistent (no/yes): Is B consistent in the information it provides throughout the conversation?",
        "Diverse": "Diverse (no/somewhat/yes): Is there diversity in B's responses?",
        "Depth": "Depth (no/somewhat/yes): Does B discuss topics in depth?",
        "Likeable": "Likeable (no/somewhat/yes): Does B display a likeable personality?",
        "Understanding": "Understanding (no/somewhat/yes): Does B seem to understand the user?",
        "Flexible": "Flexible (no/somewhat/yes): Is B flexible and adaptable to the user and their interests?",
        "Informative": "Informative (no/somewhat/yes): Is B informative throughout the conversation?",
        "Inquisitive": "Inquisitive (no/somewhat/yes): Is B inquisitive throughout the conversation?",
        "Overall-turn": "Overall Quality (very bad/bad/neutral/good/very good): What is your overall impression of the final response?",
        "Overall-dialog": "Overall Quality (very bad/bad/neutral/good/very good): What is your overall impression of the dialogue by B?",
    },
    "dstc9": {
        "likeable": "Likeable (no/somewhat/yes): Does B display a likeable personality?",
        "flexible": "Flexible (no/somewhat/yes): Is B flexible and adaptable to the user and their interests?",
        "coherent": "Coherent (no/somewhat/yes): Throughout the dialog, is B coherent and maintaining a good conversation flow?",
        "consistent": "Consistent (no/yes): Is B consistent in the information it provides throughout the conversation?",
        "topic depth": "Depth (no/somewhat/yes): Does B discuss topics in depth?",
        "overall": "Overall Quality (very bad/bad/neutral/good/very good): What is your overall impression of the dialogue by B?",
        "inquisitive": "Inquisitive (no/somewhat/yes): Is B inquisitive throughout the conversation?",
        "diverse": "Diverse (no/somewhat/yes): Is there diversity in B's responses?",
        "understanding": "Understanding (no/somewhat/yes): Does B seem to understand the user?",
        "informative": "Informative (no/somewhat/yes): Is B informative throughout the conversation?",
        "error recovery": "Error Recovery (no/somewhat/yes): Is B able to recover from errors that it makes?",
    },
    "grade": {
        "coherent": "Coherence (completely incoherent/incoherent/somewhat coherent/coherent/very coherent): How coherent is the response with the previous context?"
    },
    "predictiveengage": {
        "engaging": "Engaging (completely unengaging/unengaging/somewhat engaging/moderately engaging/very engaging): How engaging is the response?"
    },
    "holisticeval-context": {
        "context_coherence": "Coherence (completely incoherent/incoherent/somewhat coherent/coherent/very coherent): How coherent is the response with the previous context?"
    },
    "holisticeval-fluency": {
        "fluency": "Fluent (very inarticulate/inarticulate/somewhat fluent/moderately fluent/very fluent): Is the dialogue fluently written?"
    }
}

dataset_int2label_mapping = {
    "tc_usr": {
        "Engaging": {1: "dull", 2: "somewhat interesting", 3: "interesting"},
        "Maintains Context": {1: "no", 2: "somewhat", 3: "yes"},
        "Natural": {1: "no", 2: "somewhat", 3: "yes"},
        "Overall": {1: "very bad", 2: "bad", 3: "neutral", 4: "good", 5: "very good"},
        "Understandable": {0: "no", 1: "yes"},
        "Uses Knowledge": {0: "no", 1: "yes"},
    },
    "pc_usr": {
        "Engaging": {1: "dull", 2: "somewhat interesting", 3: "interesting"},
        "Maintains Context": {1: "no", 2: "somewhat", 3: "yes"},
        "Natural": {1: "no", 2: "somewhat", 3: "yes"},
        "Overall": {1: "very bad", 2: "bad", 3: "neutral", 4: "good", 5: "very good"},
        "Understandable": {0: "no", 1: "yes"},
        "Uses Knowledge": {0: "no", 1: "yes"},
    },
    "fed": {
        "Interesting": {0: "no", 1: "somewhat", 2: "yes"},
        "Engaging": {0: "no", 1: "somewhat", 2: "yes"},
        "Specific": {0: "no", 1: "somewhat", 2: "yes"},
        "Relevant": {0: "no", 1: "somewhat", 2: "yes"},
        "Correct": {0: "no", 1: "somewhat", 2: "yes"},
        "Semantically appropriate": {0: "no", 1: "somewhat", 2: "yes"},
        "Understandable": {0: "no", 1: "yes"},
        "Fluent": {0: "no", 1: "somewhat", 2: "yes"},
        "Coherent": {0: "no", 1: "somewhat", 2: "yes"},
        "Error recovery": {0: "no", 1: "somewhat", 2: "yes"},
        "Consistent": {0: "no", 1: "yes"},
        "Diverse": {0: "no", 1: "somewhat", 2: "yes"},
        "Depth": {0: "no", 1: "somewhat", 2: "yes"},
        "Likeable": {0: "no", 1: "somewhat", 2: "yes"},
        "Understanding": {0: "no", 1: "somewhat", 2: "yes"},
        "Flexible": {0: "no", 1: "somewhat", 2: "yes"},
        "Informative": {0: "no", 1: "somewhat", 2: "yes"},
        "Inquisitive": {0: "no", 1: "somewhat", 2: "yes"},
        "Overall-turn": {0: "very bad", 1: "bad", 2: "neutral", 3: "good", 4: "very good"},
        "Overall-dialog": {0: "very bad", 1: "bad", 2: "neutral", 3: "good", 4: "very good"},
    },
    "dstc9": {
        "likeable": {1: "no", 2: "somewhat", 3: "yes"},
        "flexible": {1: "no", 2: "somewhat", 3: "yes"},
        "coherent": {1: "no", 2: "somewhat", 3: "yes"},
        "consistent": {0: "no", 1: "yes"},
        "topic depth": {1: "no", 2: "somewhat", 3: "yes"},
        "overall": {1: "very bad", 2: "bad", 3: "neutral", 4: "good", 5: "very good"},
        "inquisitive": {1: "no", 2: "somewhat", 3: "yes"},
        "diverse": {1: "no", 2: "somewhat", 3: "yes"},
        "understanding": {1: "no", 2: "somewhat", 3: "yes"},
        "informative": {1: "no", 2: "somewhat", 3: "yes"},
        "error recovery": {1: "no", 2: "somewhat", 3: "yes"},
    },
    "grade": {
        "coherent": {1: "completely incoherent", 2: "incoherent", 3: "somewhat coherent", 4: "coherent", 5: "very coherent"}
    },
    "predictiveengage": {
        "engaging": {1: "completely unengaging", 2: "unengaging", 3: "somewhat engaging", 4: "moderately engaging", 5: "very engaging"}
    },
    "holisticeval-context": {
        "context_coherence": {1: "completely incoherent", 2: "incoherent", 3: "somewhat coherent", 4: "coherent", 5: "very coherent"},
    },
    "holisticeval-fluency": {
        "fluency": {1: "very inarticulate", 2: "inarticulate", 3: "somewhat fluent", 4: "moderately fluent", 5: "very fluent"},
    }
}

dataset_label2int_mapping = {
    "tc_usr": {
        "Engaging": {"dull": 1, "somewhat interesting": 2, "interesting": 3},
        "Maintains Context": {"no": 1, "somewhat": 2, "yes": 3},
        "Natural" : {"no": 1, "somewhat": 2, "yes": 3},
        "Overall": {"very bad": 1, "bad": 2, "neutral": 3, "good": 4, "very good": 5},
        "Understandable": {"no": 0, "yes": 1},
        "Uses Knowledge": {"no": 0, "yes": 1},
    },
    "pc_usr": {
        "Engaging": {"dull": 1, "somewhat interesting": 2, "interesting": 3},
        "Maintains Context": {"no": 1, "somewhat": 2, "yes": 3},
        "Natural" : {"no": 1, "somewhat": 2, "yes": 3},
        "Overall": {"very bad": 1, "bad": 2, "neutral": 3, "good": 4, "very good": 5},
        "Understandable": {"no": 0, "yes": 1},
        "Uses Knowledge": {"no": 0, "yes": 1},
    },
    "fed": {
        "Interesting": {"no": 0, "somewhat": 1, "yes": 2},
        "Engaging": {"no": 0, "somewhat": 1, "yes": 2},
        "Specific": {"no": 0, "somewhat": 1, "yes": 2},
        "Relevant": {"no": 0, "somewhat": 1, "yes": 2},
        "Correct": {"no": 0, "somewhat": 1, "yes": 2},
        "Semantically appropriate": {"no": 0, "somewhat": 1, "yes": 2},
        "Understandable": {"no": 0, "yes": 1},
        "Fluent": {"no": 0, "somewhat": 1, "yes": 2},
        "Coherent": {"no": 0, "somewhat": 1, "yes": 2},
        "Error recovery": {"no": 0, "somewhat": 1, "yes": 2},
        "Consistent": {"no": 0, "yes": 1},
        "Diverse": {"no": 0, "somewhat": 1, "yes": 2},
        "Depth": {"no": 0, "somewhat": 1, "yes": 2},
        "Likeable": {"no": 0, "somewhat": 1, "yes": 2},
        "Understanding": {"no": 0, "somewhat": 1, "yes": 2},
        "Flexible": {"no": 0, "somewhat": 1, "yes": 2},
        "Informative": {"no": 0, "somewhat": 1, "yes": 2},
        "Inquisitive": {"no": 0, "somewhat": 1, "yes": 2},
        "Overall-turn": {"very bad": 0, "bad": 1, "neutral": 2, "good": 3, "very good": 4},
        "Overall-dialog": {"very bad": 0, "bad": 1, "neutral": 2, "good": 3, "very good": 4},
    },
    "dstc9": {
        "likeable": {"no": 1, "somewhat": 2, "yes": 3},
        "flexible": {"no": 1, "somewhat": 2, "yes": 3},
        "coherent": {"no": 1, "somewhat": 2, "yes": 3},
        "consistent": {"no": 0, "yes": 1},
        "topic depth": {"no": 1, "somewhat": 2, "yes": 3},
        "overall": {"very bad": 1, "bad": 2, "neutral": 3, "good": 4, "very good": 5},
        "inquisitive": {"no": 1, "somewhat": 2, "yes": 3},
        "diverse": {"no": 1, "somewhat": 2, "yes": 3},
        "understanding": {"no": 1, "somewhat": 2, "yes": 3},
        "informative": {"no": 1, "somewhat": 2, "yes": 3},
        "error recovery": {"no": 1, "somewhat": 2, "yes": 3},
    },
    "grade": {
        "coherent": {"completely incoherent": 1, "incoherent": 2, "somewhat coherent": 3, "coherent": 4, "very coherent": 5},
    },
    "predictiveengagage": {
        "engaging": {"completely unengaging": 1, "unengaging": 2, "somewhat engaging": 3, "moderately engaging": 4, "very engaging": 5}
    },
    "holisticeval-context": {
        "context_coherence": {"completely incoherent": 1, "incoherent": 2, "somewhat coherent": 3, "coherent": 4, "very coherent": 5},
    },
    "holisticeval-fluency": {
        "fluency": {"very inarticulate": 1, "inarticulate": 2, "somewhat fluent": 3, "moderately fluent": 4, "very fluent": 5},
    }
}

def process_dstc6_dataset(dstc6_dataset):
    dstc6_dataset_formatted = []
    current_dialog = {"dataset": "dstc6", "dataset_subclass": "NA", "dataset_id": "NA", "dialog_id": 1, "dialog": [], "dialog_annotations": "NA", "facts": "NA", "num_final_responses": None, "final_responses": [], "annotation_final_responses": []}
    num_final_responses = 0
    dialog_line_num = 0
    final_responses = []
    annotation_final_responses = []
    for i, line in enumerate(dstc6_dataset):
        if line.startswith("### Dialog:") and i != 0:
            current_dialog["num_final_responses"] = num_final_responses
            current_dialog["final_responses"] = final_responses
            current_dialog["annotation_final_responses"] = annotation_final_responses
            dstc6_dataset_formatted.append(current_dialog)
            current_dialog = {"dataset": "dstc6", "dataset_subclass": "NA", "dataset_id": "NA", "dialog_id": int(line.replace("### Dialog: ", "").strip()), "dialog": [], "dialog_annotations": "NA", "facts": "NA", "num_final_responses": None, "final_responses": [], "annotation_final_responses": []}
            num_final_responses = 0
            dialog_line_num = 0
            final_responses = []
            annotation_final_responses = []
        elif line.startswith("U: "):
            # print("Hey")
            if len(current_dialog["dialog"]) == 0:
                dialog_line_num = 1
            else:
                dialog_line_num += 1
            current_dialog["dialog"].append({"index": dialog_line_num, "sender": "U", "text": line.replace("U: ", "").strip(), "sender_class": "NA"})
        elif line.startswith("S: "):
            dialog_line_num += 1
            current_dialog["dialog"].append({"index": dialog_line_num, "sender": "S", "text": line.replace("S: ", "").strip(), "sender_class": "NA"})
        elif line.startswith("S_"):
            num_final_responses += 1
            annotations_start_index = line.find("[") + 1
            annotations_end_index = line.find("]")
            annotations = line[annotations_start_index:annotations_end_index].split(",")
            for k, every in enumerate(annotations):
                annotations[k] = int(every)
            final_responses.append({"text": line[annotations_end_index + 2:].strip(), "model": "NA"})
            annotation_final_responses.append({"overall": annotations})
    return dstc6_dataset_formatted

def process_dstc9_chatbotX_dataset(dstc9_dataset, chatbot_num):
    dstc9_chatbotX_dataset_formatted = []
    num_final_responses = 0
    dialog_line_num = 0
    final_responses = []
    annotation_final_responses = []

    for i, datapoint in enumerate(dstc9_dataset):
        dialog_annotations = {"all": {"consistent": [datapoint["consistent"]], 
                                    "likeable": [datapoint["likeable"]], 
                                    "diverse": [datapoint["diverse"]], 
                                    "informative": [datapoint["informative"]], 
                                    "coherent": [datapoint["coherent"]], 
                                    "overall": [datapoint["human (overall)"]], 
                                    "understanding": [datapoint["understanding"]],
                                    "flexible": [datapoint["flexible"]],
                                    "topic depth": [datapoint["topic depth"]],
                                    "error recovery": [datapoint["error recovery"]],
                                    "inquisitive": [datapoint["inquisitive"]]},
                                    "individual": "NA"}
        
        current_dialog = {"dataset": "dstc9", 
                        "dataset_subclass": f"chatbot{chatbot_num}", 
                        "dataset_id": "NA", 
                        "dialog_id": i + 1, 
                        "dialog": [], 
                        "dialog_annotations": dialog_annotations, 
                        "facts": "NA", "num_final_responses": 0, "final_responses": "NA", "annotation_final_responses": "NA"}
        dialog_lines = datapoint['context'].split("\n")
        for dialog_line_num, line in enumerate(dialog_lines):
            current_dialog["dialog"].append({"index": dialog_line_num + 1, 
                                            "sender": "system" if line.find("System: ") != -1 else "user", 
                                            "text": line.replace("System: ", "").replace("User: ", ""), 
                                            "sender_class": "NA"})
        dstc9_chatbotX_dataset_formatted.append(current_dialog)
    return dstc9_chatbotX_dataset_formatted

def process_holisticeval_context_dataset(holisticeval_dailydialog_context):
    holisticeval_dailydialog_dataset_context_formatted = []
    num_final_responses = 0
    dialog_line_num = 0
    final_responses = []
    annotation_final_responses = []

    for i in range(len(holisticeval_dailydialog_context)):

        dialog_id = int(holisticeval_dailydialog_context.at[i, 0])
        dialog_context = holisticeval_dailydialog_context.at[i, 1]
        dialog_final_response = holisticeval_dailydialog_context.at[i, 2]
        dialog_final_a1 = holisticeval_dailydialog_context.at[i, 3]
        dialog_final_a2 = holisticeval_dailydialog_context.at[i, 4]
        dialog_final_a3 = holisticeval_dailydialog_context.at[i, 5]
        dialog_final_a4 = holisticeval_dailydialog_context.at[i, 6]
        dialog_final_a5 = holisticeval_dailydialog_context.at[i, 7]
        dialog_final_a6 = holisticeval_dailydialog_context.at[i, 8]
        dialog_final_a7 = holisticeval_dailydialog_context.at[i, 9]
        dialog_final_a8 = holisticeval_dailydialog_context.at[i, 10]
        dialog_final_a9 = holisticeval_dailydialog_context.at[i, 11]
        dialog_final_a10 = holisticeval_dailydialog_context.at[i, 12]

        final_responses = [{"text": dialog_final_response, "model": "NA"}]
        annotation_final_responses = [{"context_coherence": [int(dialog_final_a1),
                                                int(dialog_final_a2),
                                                int(dialog_final_a3),
                                                int(dialog_final_a4),
                                                int(dialog_final_a5),
                                                int(dialog_final_a6),
                                                int(dialog_final_a7),
                                                int(dialog_final_a8),
                                                int(dialog_final_a9),
                                                int(dialog_final_a10)]}]

        current_dialog = {"dataset": "holisticeval-context", 
                        "dataset_subclass": "NA", 
                        "dataset_id": "NA", 
                        "dialog_id": dialog_id, 
                        "dialog": [], 
                        "dialog_annotations": "NA", 
                        "facts": "NA", 
                        "num_final_responses": 1, 
                        "final_responses": final_responses, 
                        "annotation_final_responses": annotation_final_responses}
        current_dialog["dialog"] = [{"index": 1, 
                                    "sender": "human", 
                                    "text": dialog_context, 
                                    "sender_class": "NA"}]
        holisticeval_dailydialog_dataset_context_formatted.append(current_dialog)
    return holisticeval_dailydialog_dataset_context_formatted  

def process_holisticeval_fluency_dataset(holisticeval_dailydialog_fluency):
    holisticeval_dailydialog_dataset_fluency_formatted = []
    num_final_responses = 0
    dialog_line_num = 0
    final_responses = "NA"
    annotation_final_responses = "NA"

    # for i, datapoint in enumerate(holisticeval_dailydialog_dataset_context):
    for i in range(len(holisticeval_dailydialog_fluency)):

        dialog_id = int(holisticeval_dailydialog_fluency.at[i, 0])
        dialog_context = holisticeval_dailydialog_fluency.at[i, 1]
        dialog_context_a1 = holisticeval_dailydialog_fluency.at[i, 2]
        dialog_context_a2 = holisticeval_dailydialog_fluency.at[i, 3]
        dialog_context_a3 = holisticeval_dailydialog_fluency.at[i, 4]
        dialog_context_a4 = holisticeval_dailydialog_fluency.at[i, 5]
        dialog_context_a5 = holisticeval_dailydialog_fluency.at[i, 6]
        dialog_context_a6 = holisticeval_dailydialog_fluency.at[i, 7]
        dialog_context_a7 = holisticeval_dailydialog_fluency.at[i, 8]
        dialog_context_a8 = holisticeval_dailydialog_fluency.at[i, 9]
        dialog_context_a9 = holisticeval_dailydialog_fluency.at[i, 10]
        dialog_context_a10 = holisticeval_dailydialog_fluency.at[i, 11]


        annotations = {"all": {"fluency": [int(dialog_context_a1),
                                    int(dialog_context_a2),
                                    int(dialog_context_a3),
                                    int(dialog_context_a4),
                                    int(dialog_context_a5),
                                    int(dialog_context_a6),
                                    int(dialog_context_a7),
                                    int(dialog_context_a8),
                                    int(dialog_context_a9),
                                    int(dialog_context_a10)],},
                    "individual": "NA"}

        dialog = {"index": 1, 
                "sender": "NA", 
                "text": holisticeval_dailydialog_fluency.at[i, 1], 
                "sender_class": "NA"}
        current_dialog = {"dataset": "holisticeval-fluency", 
                        "dataset_subclass": "NA", 
                        "dataset_id": "NA", 
                        "dialog_id": dialog_id, 
                        "dialog": [dialog], 
                        "dialog_annotations": annotations, 
                        "facts": "NA", 
                        "num_final_responses": 0, 
                        "final_responses": "NA", 
                        "annotation_final_responses": "NA"}
        holisticeval_dailydialog_dataset_fluency_formatted.append(current_dialog)
    return holisticeval_dailydialog_dataset_fluency_formatted

def process_predengage_dataset(predengage_data):
    predengage_dailydialog_dataset_formatted = []

    for i in range(len(predengage_data)):
        dialog_id = i + 1
        dialog_context = predengage_data.at[i, "query"]
        
        dialog = [{"index": 1, 
                "sender": "NA",
                "text": predengage_data.at[i, "query"],
                "sender_class": "NA"}]
        
        num_final_responses = 1
        final_responses = [{"text": predengage_data.at[i, "response"], "model": "NA"}]
        
        human_score = predengage_data.at[i, "human_score"]
        # ruber_unref_score = predengage_data.at[i, "Ruber_Unref_Score"]
        # ctx_unref_score = predengage_data.at[i, "CTX_Unref_Score"]
        # eng_score_meanpooling = predengage_data.at[i, "Eng_Score_MeanPooling"]
        # eng_score_maxpooling = predengage_data.at[i, "Eng_Score_MaxPooling"]

        annotations_final_responses = [{"engaging": [int(human_score)]}] 
                                        # "ruber_unref_score": [ruber_unref_score],
                                        # "ctx_unref_score": [ctx_unref_score],
                                        # "eng_score_meanpooling": [eng_score_meanpooling],
                                        # "eng_score_maxpooling": [eng_score_maxpooling]}]

        
        current_dialog = {"dataset": "predictiveengage", 
                        "dataset_subclass": "NA", 
                        "dataset_id": "NA", 
                        "dialog_id": int(dialog_id), 
                        "dialog": dialog, 
                        "dialog_annotations": "NA", 
                        "facts": "NA", 
                        "num_final_responses": num_final_responses,
                        "final_responses": final_responses, 
                        "annotation_final_responses": annotations_final_responses}
        predengage_dailydialog_dataset_formatted.append(current_dialog)
    return predengage_dailydialog_dataset_formatted

def process_grade_dataset(grade_data, dataset_type):
    grade_dataset_formatted = []

    for datapoint in grade_data:
        if datapoint["Dataset"] == dataset_type:
            dialog_id = datapoint["ID"]
            
            dialog = []
            dialog_line_1 = datapoint["Context"].split("|||")[0]
            dialog_line_2 = datapoint["Context"].split("|||")[1]
            dialog.append({"index": 1, 
                        "sender": "NA", 
                        "text": dialog_line_1,
                        "sender_class": "NA"})
            dialog.append({"index": 2, 
                        "sender": "NA", 
                        "text": dialog_line_2,
                        "sender_class": "NA"})
            
            num_final_responses = 1
            final_responses = [{"text": datapoint["Response"], "model": datapoint["DialogModel"]}]
            annotations_final_responses = [{"coherent": ast.literal_eval(datapoint["HumanScores"])}]
            
            current_dialog = {"dataset": "grade", 
                            "dataset_subclass": dataset_type, 
                            "dataset_id": "NA", 
                            "dialog_id": dialog_id, 
                            "dialog": dialog, 
                            "dialog_annotations": "NA", 
                            "facts": "NA", 
                            "num_final_responses": num_final_responses,
                            "final_responses": final_responses, 
                            "annotation_final_responses": annotations_final_responses}
            
            grade_dataset_formatted.append(current_dialog)
    return grade_dataset_formatted

def process_pctc_dataset(data, dataset_type):
    usr_dataset_formatted = []
    for every in data:
        dialog_lines_here = every["context"].split("\n")
        dialog_here = []
        for i, each in enumerate(dialog_lines_here):
            dialog_line_dict = {"index": i + 1, "sender": "NA", "text": each, "sender_class": "NA"}
            dialog_here.append(dialog_line_dict)
        final_responses = [{"text": response["response"].strip(), "model": response["model"]} for response in every["responses"]]
        responses_annotations = []
        annotations_list = ["Understandable", "Natural", "Maintains Context", "Engaging", "Uses Knowledge", "Overall"]
        for response in every["responses"]:
            responses_annotations.append({k: response[k] for k in annotations_list})
        new_data_format = {"dataset": dataset_type, "dataset_subclass": "NA", "dataset_id": "NA", "dialog_id": "NA", "dialog": dialog_here, "dialog_annotations": "NA", "facts": every["fact"].split("\n"), "num_final_responses": len(final_responses), "final_responses": final_responses, "annotation_final_responses": responses_annotations}
        usr_dataset_formatted.append(new_data_format)
    return usr_dataset_formatted

def process_fed_dataset(fed_dataset):
    fed_dataset_formatted = []
    for every in fed_dataset:
        dialog_lines_here = every["context"].split("\n")
        dialog_here = []
        for i, each in enumerate(dialog_lines_here):
            dialog_line_dict = {"index": i + 1, "sender": "System" if each.find("System") != -1 else "User", "text": each.replace("System: ", "").replace("User: ", ""), "sender_class": "NA"}
            dialog_here.append(dialog_line_dict)
        # final_responses = [{"text": response["response"].strip(), "model": response["model"]} for response in every["responses"]]
        # if "response" not in every.keys():
        #     continue
        final_response_annotations_list = [
                            "Interesting",
                            "Engaging",
                            "Specific",
                            "Relevant",
                            "Correct",
                            "Semantically appropriate",
                            "Understandable",
                            "Fluent",]
        dialog_annotations_list = [
                            "Coherent",
                            "Error recovery",
                            "Consistent",
                            "Diverse",
                            "Depth",
                            "Likeable",
                            "Understanding",
                            "Flexible",
                            "Informative",
                            "Inquisitive",]
        responses_annotations = [{}]
        dialog_annotations = {"all": {}, "individual": "NA"}
        for each in dialog_annotations_list:
            dialog_annotations["all"][each] = []
        if "response" in every.keys():
            response = every["response"]
            final_responses = [{"text": every["response"], "model": every["system"]}]
            final_response_annotations_list.append("Overall")
        else:
            final_responses = "NA"
            dialog_annotations_list.append("Overall")

        for k in final_response_annotations_list:
            try:
                responses_annotations[0][k] = every["annotations"][k]
            except:
                responses_annotations[0][k] = "NA"
        for k in dialog_annotations_list:
            try:
                dialog_annotations["all"][k] = every["annotations"][k]
            except:
                dialog_annotations["all"][k] = "NA"
        new_data_format = {"dataset": "fed", "dataset_subclass": "NA", "dataset_id": "NA", "dialog_id": "NA", "dialog": dialog_here, "dialog_annotations": dialog_annotations, "facts": "NA", "num_final_responses": "NA" if final_responses == "NA" else len(final_responses), "final_responses": final_responses, "annotation_final_responses": responses_annotations}
        fed_dataset_formatted.append(new_data_format)
    return fed_dataset_formatted

def read_data(data_folder="data"):
    with open(os.path.join(data_folder, "pc_usr_data.json"), "r") as f1:
        pc_dataset = json.load(f1)
    with open(os.path.join(data_folder, "tc_usr_data.json"), "r") as f1:
        tc_dataset = json.load(f1)
    with open(os.path.join(data_folder, "fed_data.json"), "r") as f1:
        fed_dataset = json.load(f1)
    with open(os.path.join(data_folder, "dstc6-human_rating_scores.txt"), "r") as f1:
        dstc6_dataset = f1.readlines()
    with open(os.path.join(data_folder, "grade_dailydialog_convai_empathetic_data.json"), "r") as f1:
        grade_data = json.load(f1)
    with open(os.path.join(data_folder, "dstc9-chatbot1.json"), "r") as f1:
        dstc9_chatbot1_dataset = json.load(f1)
    with open(os.path.join(data_folder, "dstc9-chatbot2.json"), "r") as f1:
        dstc9_chatbot2_dataset = json.load(f1)
    with open(os.path.join(data_folder, "dstc9-chatbot3.json"), "r") as f1:
        dstc9_chatbot3_dataset = json.load(f1)
    with open(os.path.join(data_folder, "dstc9-chatbot4.json"), "r") as f1:
        dstc9_chatbot4_dataset = json.load(f1)
    with open(os.path.join(data_folder, "dstc9-chatbot5.json"), "r") as f1:
        dstc9_chatbot5_dataset = json.load(f1)
    with open(os.path.join(data_folder, "dstc9-chatbot6.json"), "r") as f1:
        dstc9_chatbot6_dataset = json.load(f1)
    with open(os.path.join(data_folder, "dstc9-chatbot7.json"), "r") as f1:
        dstc9_chatbot7_dataset = json.load(f1)
    with open(os.path.join(data_folder, "dstc9-chatbot8.json"), "r") as f1:
        dstc9_chatbot8_dataset = json.load(f1)
    with open(os.path.join(data_folder, "dstc9-chatbot9.json"), "r") as f1:
        dstc9_chatbot9_dataset = json.load(f1)
    with open(os.path.join(data_folder, "dstc9-chatbot10.json"), "r") as f1:
        dstc9_chatbot10_dataset = json.load(f1)
    holisticeval_dailydialog_context = pd.read_csv(os.path.join(data_folder, "holisticeval-context-data.csv"), header=None)
    holisticeval_dailydialog_fluency = pd.read_csv(os.path.join(data_folder, "holisticeval-fluency-data.csv"), header=None)
    predengage_data = pd.read_csv(os.path.join(data_folder, "predictiveengage-eng-scores-gen-gtruth-data.csv"))
    
    return pc_dataset, tc_dataset, fed_dataset, dstc6_dataset, grade_data, dstc9_chatbot1_dataset, dstc9_chatbot2_dataset, dstc9_chatbot3_dataset, dstc9_chatbot4_dataset, dstc9_chatbot5_dataset, dstc9_chatbot6_dataset, dstc9_chatbot7_dataset, dstc9_chatbot8_dataset, dstc9_chatbot9_dataset, dstc9_chatbot10_dataset, holisticeval_dailydialog_context, holisticeval_dailydialog_fluency, predengage_data

def format_datasets(use_dstc6, pc_dataset, tc_dataset, fed_dataset, dstc6_dataset, grade_data, dstc9_chatbot1_dataset, dstc9_chatbot2_dataset, dstc9_chatbot3_dataset, dstc9_chatbot4_dataset, dstc9_chatbot5_dataset, dstc9_chatbot6_dataset, dstc9_chatbot7_dataset, dstc9_chatbot8_dataset, dstc9_chatbot9_dataset, dstc9_chatbot10_dataset, holisticeval_dailydialog_context, holisticeval_dailydialog_fluency, predengage_data):
    dstc9_chatbot1_dataset_formatted = process_dstc9_chatbotX_dataset(dstc9_chatbot1_dataset, 1)
    print("DSTC9-1 dataset processed!")
    dstc9_chatbot2_dataset_formatted = process_dstc9_chatbotX_dataset(dstc9_chatbot2_dataset, 2)
    print("DSTC9-2 dataset processed!")
    dstc9_chatbot3_dataset_formatted = process_dstc9_chatbotX_dataset(dstc9_chatbot3_dataset, 3)
    print("DSTC9-3 dataset processed!")
    dstc9_chatbot4_dataset_formatted = process_dstc9_chatbotX_dataset(dstc9_chatbot4_dataset, 4)
    print("DSTC9-4 dataset processed!")
    dstc9_chatbot5_dataset_formatted = process_dstc9_chatbotX_dataset(dstc9_chatbot5_dataset, 5)
    print("DSTC9-5 dataset processed!")
    dstc9_chatbot6_dataset_formatted = process_dstc9_chatbotX_dataset(dstc9_chatbot6_dataset, 6)
    print("DSTC9-6 dataset processed!")
    dstc9_chatbot7_dataset_formatted = process_dstc9_chatbotX_dataset(dstc9_chatbot7_dataset, 7)
    print("DSTC9-7 dataset processed!")
    dstc9_chatbot8_dataset_formatted = process_dstc9_chatbotX_dataset(dstc9_chatbot8_dataset, 8)
    print("DSTC9-8 dataset processed!")
    dstc9_chatbot9_dataset_formatted = process_dstc9_chatbotX_dataset(dstc9_chatbot9_dataset, 9)
    print("DSTC9-9 dataset processed!")
    dstc9_chatbot10_dataset_formatted = process_dstc9_chatbotX_dataset(dstc9_chatbot10_dataset, 10)
    print("DSTC9-10 dataset processed!")
    holisticeval_dailydialog_dataset_context_formatted = process_holisticeval_context_dataset(holisticeval_dailydialog_context)
    print("HolisticEval-Context dataset processed!")
    holisticeval_dailydialog_dataset_fluency_formatted = process_holisticeval_fluency_dataset(holisticeval_dailydialog_fluency)
    print("HolisticEval-Fluency dataset processed!")
    predengage_dailydialog_dataset_formatted = process_predengage_dataset(predengage_data)
    print("PredictiveEngage dataset processed!")
    grade_convai_dataset_formatted = process_grade_dataset(grade_data, "convai2")
    print("GRADE-ConvAI dataset processed!")
    grade_dailydialog_dataset_formatted = process_grade_dataset(grade_data, "dailydialog_EVAL")
    print("GRADE-DailyDialog dataset processed!")
    grade_empatheticdialogues_dataset_formatted = process_grade_dataset(grade_data, "empatheticdialogues")
    print("GRADE-EmpatheticDialogues dataset processed!")
    pc_usr_dataset_formatted = process_pctc_dataset(pc_dataset, "pc_usr")
    print("PC-USR dataset processed!")
    tc_usr_dataset_formatted = process_pctc_dataset(tc_dataset, "tc_usr")
    print("TC-USR dataset processed!")
    fed_dataset_formatted = process_fed_dataset(fed_dataset)
    print("FED dataset processed!")
    if use_dstc6:
        dstc6_dataset_formatted = process_dstc6_dataset(dstc6_dataset)
        print("DSTC6 dataset processed!")
        
    return fed_dataset_formatted, tc_usr_dataset_formatted, pc_usr_dataset_formatted, grade_empatheticdialogues_dataset_formatted, grade_dailydialog_dataset_formatted, grade_convai_dataset_formatted, predengage_dailydialog_dataset_formatted, holisticeval_dailydialog_dataset_fluency_formatted, holisticeval_dailydialog_dataset_context_formatted, dstc9_chatbot1_dataset_formatted, dstc9_chatbot2_dataset_formatted, dstc9_chatbot3_dataset_formatted, dstc9_chatbot4_dataset_formatted, dstc9_chatbot5_dataset_formatted, dstc9_chatbot6_dataset_formatted, dstc9_chatbot7_dataset_formatted, dstc9_chatbot8_dataset_formatted, dstc9_chatbot9_dataset_formatted, dstc9_chatbot10_dataset_formatted
    
    

def combine_datasets(use_dstc6, fed_dataset_formatted, tc_usr_dataset_formatted, pc_usr_dataset_formatted, grade_empatheticdialogues_dataset_formatted, grade_dailydialog_dataset_formatted, grade_convai_dataset_formatted, predengage_dailydialog_dataset_formatted, holisticeval_dailydialog_dataset_fluency_formatted, holisticeval_dailydialog_dataset_context_formatted, dstc9_chatbot1_dataset_formatted, dstc9_chatbot2_dataset_formatted, dstc9_chatbot3_dataset_formatted, dstc9_chatbot4_dataset_formatted, dstc9_chatbot5_dataset_formatted, dstc9_chatbot6_dataset_formatted, dstc9_chatbot7_dataset_formatted, dstc9_chatbot8_dataset_formatted, dstc9_chatbot9_dataset_formatted, dstc9_chatbot10_dataset_formatted):
    all_datasets_combined = []
    all_datasets_combined.extend(fed_dataset_formatted)
    all_datasets_combined.extend(tc_usr_dataset_formatted)
    all_datasets_combined.extend(pc_usr_dataset_formatted)
    all_datasets_combined.extend(grade_empatheticdialogues_dataset_formatted)
    all_datasets_combined.extend(grade_dailydialog_dataset_formatted)
    all_datasets_combined.extend(grade_convai_dataset_formatted)
    all_datasets_combined.extend(predengage_dailydialog_dataset_formatted)
    all_datasets_combined.extend(holisticeval_dailydialog_dataset_fluency_formatted)
    all_datasets_combined.extend(holisticeval_dailydialog_dataset_context_formatted)
    all_datasets_combined.extend(dstc9_chatbot1_dataset_formatted)
    all_datasets_combined.extend(dstc9_chatbot2_dataset_formatted)
    all_datasets_combined.extend(dstc9_chatbot3_dataset_formatted)
    all_datasets_combined.extend(dstc9_chatbot4_dataset_formatted)
    all_datasets_combined.extend(dstc9_chatbot5_dataset_formatted)
    all_datasets_combined.extend(dstc9_chatbot6_dataset_formatted)
    all_datasets_combined.extend(dstc9_chatbot7_dataset_formatted)
    all_datasets_combined.extend(dstc9_chatbot8_dataset_formatted)
    all_datasets_combined.extend(dstc9_chatbot9_dataset_formatted)
    all_datasets_combined.extend(dstc9_chatbot10_dataset_formatted)
    if use_dstc6:
        all_datasets_combined.extend(dstc6_dataset_formatted)
    return all_datasets_combined

def save_standardized_json(use_dstc6, standardized_datasets_save_dir, all_datasets_combined, fed_dataset_formatted, tc_usr_dataset_formatted, pc_usr_dataset_formatted, grade_empatheticdialogues_dataset_formatted, grade_dailydialog_dataset_formatted, grade_convai_dataset_formatted, predengage_dailydialog_dataset_formatted, holisticeval_dailydialog_dataset_fluency_formatted, holisticeval_dailydialog_dataset_context_formatted, dstc9_chatbot1_dataset_formatted, dstc9_chatbot2_dataset_formatted, dstc9_chatbot3_dataset_formatted, dstc9_chatbot4_dataset_formatted, dstc9_chatbot5_dataset_formatted, dstc9_chatbot6_dataset_formatted, dstc9_chatbot7_dataset_formatted, dstc9_chatbot8_dataset_formatted, dstc9_chatbot9_dataset_formatted, dstc9_chatbot10_dataset_formatted):
    with open(os.path.join(standardized_datasets_save_dir, "all-combined-standardized.json"), "w") as f1:
        json.dump(all_datasets_combined, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_fed.json"), "w") as f1:
        json.dump(fed_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_tc_usr.json"), "w") as f1:
        json.dump(tc_usr_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_pc_usr.json"), "w") as f1:
        json.dump(pc_usr_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_grade_empatheticdialogues.json"), "w") as f1:
        json.dump(grade_empatheticdialogues_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_grade_dailydialog.json"), "w") as f1:
        json.dump(grade_dailydialog_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_grade_convai.json"), "w") as f1:
        json.dump(grade_convai_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_predengage.json"), "w") as f1:
        json.dump(predengage_dailydialog_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_holisticeval-fluency.json"), "w") as f1:
        json.dump(holisticeval_dailydialog_dataset_fluency_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_holisticeval-context.json"), "w") as f1:
        json.dump(holisticeval_dailydialog_dataset_context_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_dstc9-chatbot1.json"), "w") as f1:
        json.dump(dstc9_chatbot1_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_dstc9-chatbot2.json"), "w") as f1:
        json.dump(dstc9_chatbot2_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_dstc9-chatbot3.json"), "w") as f1:
        json.dump(dstc9_chatbot3_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_dstc9-chatbot4.json"), "w") as f1:
        json.dump(dstc9_chatbot4_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_dstc9-chatbot5.json"), "w") as f1:
        json.dump(dstc9_chatbot5_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_dstc9-chatbot6.json"), "w") as f1:
        json.dump(dstc9_chatbot6_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_dstc9-chatbot7.json"), "w") as f1:
        json.dump(dstc9_chatbot7_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_dstc9-chatbot8.json"), "w") as f1:
        json.dump(dstc9_chatbot8_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_dstc9-chatbot9.json"), "w") as f1:
        json.dump(dstc9_chatbot9_dataset_formatted, f1, indent=4)
    with open(os.path.join(standardized_datasets_save_dir, "standardized_dstc9-chatbot10.json"), "w") as f1:
        json.dump(dstc9_chatbot10_dataset_formatted, f1, indent=4)
    if use_dstc6:
        with open(os.path.join(standardized_datasets_save_dir, "standardized_dstc6.json"), "w") as f1:
            json.dump(dstc6_dataset_formatted, f1, indent=4)
    return all_datasets_combined, fed_dataset_formatted, tc_usr_dataset_formatted, pc_usr_dataset_formatted, grade_empatheticdialogues_dataset_formatted, grade_dailydialog_dataset_formatted, grade_convai_dataset_formatted, predengage_dailydialog_dataset_formatted, holisticeval_dailydialog_dataset_fluency_formatted, holisticeval_dailydialog_dataset_context_formatted, dstc9_chatbot1_dataset_formatted, dstc9_chatbot2_dataset_formatted, dstc9_chatbot3_dataset_formatted, dstc9_chatbot4_dataset_formatted, dstc9_chatbot5_dataset_formatted, dstc9_chatbot6_dataset_formatted, dstc9_chatbot7_dataset_formatted, dstc9_chatbot8_dataset_formatted, dstc9_chatbot9_dataset_formatted, dstc9_chatbot10_dataset_formatted

def create_training_data(dataset_name, all_datasets_combined, output_dir, store_data_files, dataset_subclass="NA"):
    if store_data_files and not os.path.exists(output_dir):
        os.makedirs(output_dir)
#     with open(all_datasets_json_path, "r") as f1:
#         dataset = json.load(f1)
    training_prompts_list_labelled = []
    training_prompts_list_unlabelled = []
    training_labels_list_float = []
    training_labels_list_int_rounded = []
    training_labels_list_word = []
#     pprint(dataset[1])
    for enum, each in enumerate(all_datasets_combined):
        if each["dataset"] == dataset_name and each["dataset_subclass"] == dataset_subclass:
#             dataset_here = each["dataset"]
            ### DIALOG CONTEXT ###
            train_datapoint = """Dialog context: """
            turn_num = 0
            ends_at_a = False
            ends_at_b = False
            for dialog_context_line in each["dialog"]:
                if dialog_context_line["text"] != "":
                    if dialog_context_line["index"] % 2 != 0:
                        turn_num += 1
                        train_datapoint = train_datapoint + f"\nTurn {turn_num}.A: " + dialog_context_line["text"]
                        ends_at_a = True
                        ends_at_b = False
                    else:
                        train_datapoint = train_datapoint + f"\nTurn {turn_num}.B: " + dialog_context_line["text"]
                        ends_at_b = True
                        ends_at_a = False
    #             if each["facts"] != "NA":
            train_datapoint += "\n\n"
            ### DIALOG CONTEXT ###

            ### FACTS ###
            if each["facts"] != "NA":
                train_datapoint_withfacts = train_datapoint + "Facts: \n" + " ".join(each["facts"]) + "\n\n"
            ### FACTS ###

            ### FINAL RESPONSES ###
            if ends_at_b:
                final_turn_num = turn_num + 1
                final_speaker = "A"
            else:
                final_turn_num = turn_num
                final_speaker = "B"
            
            if each["num_final_responses"] != 0 and each["num_final_responses"] != "NA":
                for final_response_num, final_response in enumerate(each["final_responses"]):
                    train_datapoint_complete_1 = train_datapoint + f"Final response: \nTurn {final_turn_num}.{final_speaker}: " + final_response["text"].replace("System: ", "").replace("User: ", "")
                    if each["facts"] != "NA":
                        train_datapoint_complete_1_withfacts = train_datapoint_withfacts + f"Final response: \nTurn {final_turn_num}.{final_speaker}: " + final_response["text"]
                    for eval_type in each["annotation_final_responses"][final_response_num].keys():
                        if each["annotation_final_responses"][final_response_num][eval_type] != "NA":
    #                         int_annotations_list_here = each["annotation_final_responses"][final_response_num][eval_type]
                            int_annotations_list_here = [i for i in each["annotation_final_responses"][final_response_num][eval_type] if type(i) == int]
#                             try:
                            int_annotation_avg_here = sum(int_annotations_list_here) / len(int_annotations_list_here)
#                             except:
#                                 print(int_annotations_list_here)
    #                             print(int_annotations_list_here)
                            label_here_float_avg = int_annotation_avg_here
                            label_here_int_rounded = round(int_annotation_avg_here)
                            if eval_type == "Overall" and dataset_name == "fed":
                                eval_type_mapping = "Overall-turn"
                            else:
                                eval_type_mapping = eval_type
    #                         try:
                            label_here_word = dataset_int2label_mapping[dataset_name][eval_type_mapping][label_here_int_rounded]
    #                         except:
    #                             print(dataset_name)
    #                             print(eval_type_mapping)
    #                             print(label_here_int_rounded)

                            if (dataset_name == "pc_usr" or dataset_name == "tc_usr") and (eval_type == "Overall" or eval_type == "Uses Knowledge"):
                                train_datapoint_complete_2 = train_datapoint_complete_1_withfacts + f"\n\nQuestion on the final response:\n{dataset_questions_mapping[dataset_name][eval_type_mapping]}\n\nAnswer:\n"
                                train_datapoint_complete_3 = train_datapoint_complete_2 + f"{label_here_word}"
                            else:
                                train_datapoint_complete_2 = train_datapoint_complete_1 + f"\n\nQuestion on the final response:\n{dataset_questions_mapping[dataset_name][eval_type_mapping]}\n\nAnswer:\n"
                                train_datapoint_complete_3 = train_datapoint_complete_2 + f"{label_here_word}"

                            training_prompts_list_labelled.append(train_datapoint_complete_3)
                            training_prompts_list_unlabelled.append(train_datapoint_complete_2)
                            training_labels_list_float.append(label_here_float_avg)
                            training_labels_list_int_rounded.append(label_here_int_rounded)
                            training_labels_list_word.append(label_here_word)
                
            if each["dialog_annotations"] != "NA":
                for eval_type in each["dialog_annotations"]["all"].keys():
                    if each["dialog_annotations"]["all"][eval_type] != "NA":
                        int_annotations_list_here = [i for i in each["dialog_annotations"]["all"][eval_type] if type(i) == int]
                        try:
                            int_annotation_avg_here = sum(int_annotations_list_here) / len(int_annotations_list_here)
                            label_here_float_avg = int_annotation_avg_here
                            label_here_int_rounded = round(int_annotation_avg_here)
                            if eval_type == "Overall" and dataset_name == "fed":
                                eval_type_mapping = "Overall-dialog"
                            else:
                                eval_type_mapping = eval_type
#                             try:
                            label_here_word = dataset_int2label_mapping[dataset_name][eval_type_mapping][label_here_int_rounded]
#                             except:
#                                 print(eval_type_mapping)
#                                 print(int_annotation_avg_here)
#                                 print(label_here_int_rounded)
#                                 print(int_annotations_list_here)
#                                 print(each["dialog_annotations"]["all"][eval_type])
                            if dataset_name == "holisticeval-fluency":
                                train_datapoint_complete_2 = train_datapoint + f"Question on A's opening dialogue:\n{dataset_questions_mapping[dataset_name][eval_type_mapping]}\n\nAnswer:\n"
                            else:
                                train_datapoint_complete_2 = train_datapoint + f"Question on all of B's dialogue responses:\n{dataset_questions_mapping[dataset_name][eval_type_mapping]}\n\nAnswer:\n"
                            train_datapoint_complete_3 = train_datapoint_complete_2 + f"{label_here_word}"

                            training_prompts_list_labelled.append(train_datapoint_complete_3)
                            training_prompts_list_unlabelled.append(train_datapoint_complete_2)
                            training_labels_list_float.append(label_here_float_avg)
                            training_labels_list_int_rounded.append(label_here_int_rounded)
                            training_labels_list_word.append(label_here_word)
                        except:
#                             print(f"Issue with combined dataset datapoint #{enum}:")
#                             print(f"Int annotations list here for {eval_type} = {int_annotations_list_here}\nOriginal annotations for {eval_type} = ", each["dialog_annotations"]["all"][eval_type], "\n")
                            continue
#                             print(each)

                ### FINAL RESPONSES ###

    dataset_dict = {
        "labelled_prompts": training_prompts_list_labelled,
        "unlabelled_prompts": training_prompts_list_unlabelled,
        "labels_float_avg": training_labels_list_float,
        "labels_int_rounded": training_labels_list_int_rounded,
        "labels_word": training_labels_list_word,
    }

    if dataset_name == "dstc9" or dataset_name == "grade":
        dataset_name_write = dataset_name + "_" + dataset_subclass
    else:
        dataset_name_write = dataset_name
        
    if store_data_files:
        with open(os.path.join(output_dir, f"train_format_{dataset_name_write}.json"), "w") as f1:
            json.dump(dataset_dict, f1, indent=4)

    print(f"Created {dataset_name_write} training data!")
    
#     print(train_datapoint_complete_3)
    return dataset_dict

def create_all_training_data(output_dir, store_data_files):
    tc_dataset_dict = create_training_data("tc_usr", all_datasets_combined, output_dir, store_data_files)
    pc_dataset_dict = create_training_data("pc_usr", all_datasets_combined, output_dir, store_data_files)
    fed_dataset_dict = create_training_data("fed", all_datasets_combined, output_dir, store_data_files)
#     for dataset_subclass in ["chatbot1", "chatbot2", "chatbot3", "chatbot4", "chatbot5", "chatbot6", "chatbot7", "chatbot8", "chatbot9", "chatbot10"]:
    dstc9_chatbot1_dataset_dict = create_training_data("dstc9", all_datasets_combined, output_dir, store_data_files, "chatbot1")
    dstc9_chatbot2_dataset_dict = create_training_data("dstc9", all_datasets_combined, output_dir, store_data_files, "chatbot2")
    dstc9_chatbot3_dataset_dict = create_training_data("dstc9", all_datasets_combined, output_dir, store_data_files, "chatbot3")
    dstc9_chatbot4_dataset_dict = create_training_data("dstc9", all_datasets_combined, output_dir, store_data_files, "chatbot4")
    dstc9_chatbot5_dataset_dict = create_training_data("dstc9", all_datasets_combined, output_dir, store_data_files, "chatbot5")
    dstc9_chatbot6_dataset_dict = create_training_data("dstc9", all_datasets_combined, output_dir, store_data_files, "chatbot6")
    dstc9_chatbot7_dataset_dict = create_training_data("dstc9", all_datasets_combined, output_dir, store_data_files, "chatbot7")
    dstc9_chatbot8_dataset_dict = create_training_data("dstc9", all_datasets_combined, output_dir, store_data_files, "chatbot8")
    dstc9_chatbot9_dataset_dict = create_training_data("dstc9", all_datasets_combined, output_dir, store_data_files, "chatbot9")
    dstc9_chatbot10_dataset_dict = create_training_data("dstc9", all_datasets_combined, output_dir, store_data_files, "chatbot10")
#     for dataset_subclass in ["convai2", "empatheticdialogues", "dailydialog_EVAL"]:
    grade_convai2_dataset_dict = create_training_data("grade", all_datasets_combined, output_dir, store_data_files, "convai2")
    grade_empathetic_dataset_dict = create_training_data("grade", all_datasets_combined, output_dir, store_data_files, "empatheticdialogues")
    grade_dailydialog_dataset_dict = create_training_data("grade", all_datasets_combined, output_dir, store_data_files, "dailydialog_EVAL")
    predengage_dataset_dict = create_training_data("predictiveengage", all_datasets_combined, output_dir, store_data_files)
    holisticeval_context_dataset_dict = create_training_data("holisticeval-context", all_datasets_combined, output_dir, store_data_files)
    holisticeval_fluency_dataset_dict = create_training_data("holisticeval-fluency", all_datasets_combined, output_dir, store_data_files)
    all_dataset_dicts = {
        "tc_usr": tc_dataset_dict,
        "pc_usr": pc_dataset_dict,
        "fed": fed_dataset_dict,
        "dstc9-chatbot1": dstc9_chatbot1_dataset_dict,
        "dstc9-chatbot2": dstc9_chatbot2_dataset_dict,
        "dstc9-chatbot3": dstc9_chatbot3_dataset_dict,
        "dstc9-chatbot4": dstc9_chatbot4_dataset_dict,
        "dstc9-chatbot5": dstc9_chatbot5_dataset_dict,
        "dstc9-chatbot6": dstc9_chatbot6_dataset_dict,
        "dstc9-chatbot7": dstc9_chatbot7_dataset_dict,
        "dstc9-chatbot8": dstc9_chatbot8_dataset_dict,
        "dstc9-chatbot9": dstc9_chatbot9_dataset_dict,
        "dstc9-chatbot10": dstc9_chatbot10_dataset_dict,
        "grade-convai2": grade_convai2_dataset_dict,
        "grade-empathetic": grade_empathetic_dataset_dict,
        "grade-dailydialog": grade_dailydialog_dataset_dict,
        "predictiveengage": predengage_dataset_dict,
        "holisticeval-context": holisticeval_context_dataset_dict,
        "holisticeval-fluency": holisticeval_fluency_dataset_dict,
    }
    return all_dataset_dicts
       
def combine_training_format_datasets(combine_list):
    combined = {"labelled_prompts": [], "unlabelled_prompts": [], "labels_float_avg": [], "labels_int_rounded": [], "labels_word": []}
    for each in combine_list:
        combined["labelled_prompts"].extend(each["labelled_prompts"])
        combined["unlabelled_prompts"].extend(each["unlabelled_prompts"])
        combined["labels_float_avg"].extend(each["labels_float_avg"])
        combined["labels_int_rounded"].extend(each["labels_int_rounded"])
        combined["labels_word"].extend(each["labels_word"])
    return combined

def train_data_gen(train_combined):
    for i in range(len(train_combined["unlabelled_prompts"])):
        yield {"unlabelled_prompts": train_combined["unlabelled_prompts"][i], 
               "labels_word": train_combined["labels_word"][i]}

def test_data_gen(test_combined):
    for i in range(len(test_combined["unlabelled_prompts"])):
        yield {"unlabelled_prompts": test_combined["unlabelled_prompts"][i], 
               "labels_word": test_combined["labels_word"][i]}

def preprocess_tokenize(datapoints, tokenizer, max_input_length, max_target_length):
    prefix = """Analyze the following dialogue and answer the subsequent question based on it: """
    inputs = [prefix + unlabelled_prompt for unlabelled_prompt in datapoints["unlabelled_prompts"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
#     with tokenizer.as_target_tokenizer():
    labels = tokenizer(datapoints["labels_word"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
        
def get_train_test_splits(all_dataset_dicts, args_test_datasets, val_data_fraction, tokenizer, max_input_length=512, max_target_length=8):
    test_datasets_names = [dataset for dataset in args_test_datasets.split(",")]
    print("Test Datasets: ", test_datasets_names)
    test_combine_list = []
    trainval_combine_list = []
    for key in all_dataset_dicts.keys():
        if key in test_datasets_names:
            test_combine_list.append(all_dataset_dicts[key])
        else:
            trainval_combine_list.append(all_dataset_dicts[key])
    trainval_combined = combine_training_format_datasets(trainval_combine_list)
    test_combined = combine_training_format_datasets(test_combine_list)
    
    trainval_dataset = Dataset.from_generator(train_data_gen, gen_kwargs={"train_combined": trainval_combined})
    test_dataset = Dataset.from_generator(test_data_gen, gen_kwargs={"test_combined": test_combined})
    trainval_dataset = trainval_dataset.train_test_split(test_size=val_data_fraction, shuffle=True, seed=RANDOM_SEED)
    train_dataset = trainval_dataset["train"]
    val_dataset = trainval_dataset["test"]

    fn_kwargs = {"tokenizer": tokenizer,
                 "max_input_length": max_input_length,
                 "max_target_length": max_target_length}
    tokenized_train = train_dataset.map(preprocess_tokenize, batched=True, fn_kwargs=fn_kwargs, num_proc=8)
    tokenized_val = val_dataset.map(preprocess_tokenize, batched=True, fn_kwargs=fn_kwargs, num_proc=8)
    tokenized_test = test_dataset.map(preprocess_tokenize, batched=True, fn_kwargs=fn_kwargs, num_proc=8)
    
    return tokenized_train, tokenized_val, tokenized_test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_data_files", help="whether or not to save processed dataset files", action="store_true", default=False)
    parser.add_argument("--data_dirpath", help="path to the data folder where to store files",
                        default="data")
    parser.add_argument("--use_dstc6", help="include the DSTC6 dataset in the training data",
                        action="store_true", default=False)
    parser.add_argument('--test_datasets', help='comma delimited list of datasets to not train on', type=str)
    parser.add_argument('--model_checkpoint', help='model checkpoint from huggingface to be used', default='t5-large')
    parser.add_argument('--val_data_fraction', help='fractional value (e.g., 0.15) indicating the fraction of non-test data that should be used as validation data', default=0.15)
    parser.add_argument('--max_learning_rate', help='maximum value of learning rate during training (lr scheduling will happen)', default=2e-5)
    parser.add_argument('--train_batch_size', help='batch size to use while training', default=4)
    parser.add_argument('--eval_batch_size', help='batch size to use while training', default=8)
    parser.add_argument('--gradient_accumulation_steps', help='gradient accumulation steps for training', default=1)
    parser.add_argument('--num_epochs', help='the number of epochs to run training for', default=5)
    parser.add_argument('--models_save_dirpath', help='path to the directory where trained model checkpoints should be stored', default="saved-models")
    parser.add_argument('--save_steps', help='number of training steps between two successive model saves', default=1000)
    parser.add_argument('--eval_steps', help='number of training steps before evaluation on the validation data is done', default=1000)
    parser.add_argument('--no_wandb_logging', help='whether to not use wandb', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    print(f"Data directory: {args.data_dirpath}")
#     print("DSTC 6 ARG = ", args.use_dstc6)
    standardized_datasets_save_dir = os.path.join(args.data_dirpath, "standardized-format/")
    if args.store_data_files and not os.path.exists(standardized_datasets_save_dir):
        os.makedirs(standardized_datasets_save_dir)
    if not os.path.exists(args.models_save_dirpath):
        os.makedirs(args.models_save_dirpath)
    
    pc_dataset, tc_dataset, fed_dataset, dstc6_dataset, grade_data, dstc9_chatbot1_dataset, dstc9_chatbot2_dataset, dstc9_chatbot3_dataset, dstc9_chatbot4_dataset, dstc9_chatbot5_dataset, dstc9_chatbot6_dataset, dstc9_chatbot7_dataset, dstc9_chatbot8_dataset, dstc9_chatbot9_dataset, dstc9_chatbot10_dataset, holisticeval_dailydialog_context, holisticeval_dailydialog_fluency, predengage_data = read_data(args.data_dirpath)
    
    fed_dataset_formatted, tc_usr_dataset_formatted, pc_usr_dataset_formatted, grade_empatheticdialogues_dataset_formatted, grade_dailydialog_dataset_formatted, grade_convai_dataset_formatted, predengage_dailydialog_dataset_formatted, holisticeval_dailydialog_dataset_fluency_formatted, holisticeval_dailydialog_dataset_context_formatted, dstc9_chatbot1_dataset_formatted, dstc9_chatbot2_dataset_formatted, dstc9_chatbot3_dataset_formatted, dstc9_chatbot4_dataset_formatted, dstc9_chatbot5_dataset_formatted, dstc9_chatbot6_dataset_formatted, dstc9_chatbot7_dataset_formatted, dstc9_chatbot8_dataset_formatted, dstc9_chatbot9_dataset_formatted, dstc9_chatbot10_dataset_formatted = format_datasets(args.use_dstc6, pc_dataset, tc_dataset, fed_dataset, dstc6_dataset, grade_data, dstc9_chatbot1_dataset, dstc9_chatbot2_dataset, dstc9_chatbot3_dataset, dstc9_chatbot4_dataset, dstc9_chatbot5_dataset, dstc9_chatbot6_dataset, dstc9_chatbot7_dataset, dstc9_chatbot8_dataset, dstc9_chatbot9_dataset, dstc9_chatbot10_dataset, holisticeval_dailydialog_context, holisticeval_dailydialog_fluency, predengage_data)
    
    all_datasets_combined = combine_datasets(args.use_dstc6, fed_dataset_formatted, tc_usr_dataset_formatted, pc_usr_dataset_formatted, grade_empatheticdialogues_dataset_formatted, grade_dailydialog_dataset_formatted, grade_convai_dataset_formatted, predengage_dailydialog_dataset_formatted, holisticeval_dailydialog_dataset_fluency_formatted, holisticeval_dailydialog_dataset_context_formatted, dstc9_chatbot1_dataset_formatted, dstc9_chatbot2_dataset_formatted, dstc9_chatbot3_dataset_formatted, dstc9_chatbot4_dataset_formatted, dstc9_chatbot5_dataset_formatted, dstc9_chatbot6_dataset_formatted, dstc9_chatbot7_dataset_formatted, dstc9_chatbot8_dataset_formatted, dstc9_chatbot9_dataset_formatted, dstc9_chatbot10_dataset_formatted)
    
    if args.store_data_files:
        save_standardized_json(args.use_dstc6, standardized_datasets_save_dir, all_datasets_combined, fed_dataset_formatted, tc_usr_dataset_formatted, pc_usr_dataset_formatted, grade_empatheticdialogues_dataset_formatted, grade_dailydialog_dataset_formatted, grade_convai_dataset_formatted, predengage_dailydialog_dataset_formatted, holisticeval_dailydialog_dataset_fluency_formatted, holisticeval_dailydialog_dataset_context_formatted, dstc9_chatbot1_dataset_formatted, dstc9_chatbot2_dataset_formatted, dstc9_chatbot3_dataset_formatted, dstc9_chatbot4_dataset_formatted, dstc9_chatbot5_dataset_formatted, dstc9_chatbot6_dataset_formatted, dstc9_chatbot7_dataset_formatted, dstc9_chatbot8_dataset_formatted, dstc9_chatbot9_dataset_formatted, dstc9_chatbot10_dataset_formatted)
        print("Saved all datasets in a common standard format!")
    
    print("\nCreating training data...")
    output_dir = os.path.join(args.data_dirpath, "training")
    if args.store_data_files:
        print(f"Storing training data in {output_dir}...")
    all_dataset_dicts = create_all_training_data(output_dir, args.store_data_files)
    print("Created all training datasets!")
    
    print("\nTokenizing and making train/val/test splits...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenized_train, tokenized_val, tokenized_test = get_train_test_splits(all_dataset_dicts, args.test_datasets, float(args.val_data_fraction), tokenizer) 
    print("Final training data ready!")
    model_name = args.model_checkpoint.split("/")[-1]
    seq2seqargs = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.models_save_dirpath, f"{model_name}-finetuned-{args.num_epochs}-ne-{args.max_learning_rate}-lr-{args.train_batch_size}-bs-{args.test_datasets}-test"),
        num_train_epochs=int(args.num_epochs),
        evaluation_strategy="steps",
        eval_steps=int(args.eval_steps),
        per_device_train_batch_size=int(args.train_batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.max_learning_rate),
        lr_scheduler_type="linear",
        logging_strategy="steps",
        logging_first_step=False,
        logging_steps=int(args.eval_steps),
        save_strategy="steps",
        save_steps=int(args.save_steps),
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        fp16=False,
        report_to="none" if args.no_wandb_logging else "wandb",
    )
    print(f"\n\nInitializing model {args.model_checkpoint}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        seq2seqargs,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    print("Ready to train. Starting training...\n")
    trainer.train()
    print("Training complete!")

    
