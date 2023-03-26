#!/usr/bin/env python
from process_data_and_train import *
RANDOM_SEED = 42
seed_everything(RANDOM_SEED)
disable_caching()
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_test_split(all_dataset_dicts, args_test_datasets, tokenizer, max_input_length=1024, max_target_length=10):
    test_datasets_names = [dataset for dataset in args_test_datasets.split(",")]
    print("Test Datasets: ", test_datasets_names)
    test_combine_dict = {}
    for key in all_dataset_dicts.keys():
        if key in test_datasets_names:
            test_combine_dict[key] = all_dataset_dicts[key]
    test_combined = combine_test_format_datasets(test_combine_dict)
    test_dataset = Dataset.from_generator(test_data_gen, gen_kwargs={"test_combined": test_combined})

    fn_kwargs = {"tokenizer": tokenizer,
                 "max_input_length": max_input_length,
                 "max_target_length": max_target_length}
    tokenized_test = test_dataset.map(preprocess_tokenize, batched=True, fn_kwargs=fn_kwargs, num_proc=1)

    return tokenized_test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_data_files", help="whether or not to save processed dataset files", action="store_true", default=False)
    parser.add_argument("--data_dirpath", help="path to the folder where the data is stored",
                        default="data")
    parser.add_argument("--use_dstc6", help="include the DSTC6 dataset in the training data",
                        action="store_true", default=False)
    parser.add_argument('--model_checkpoints_dir', help='directory containing all model checkpoints to evaluate using', default='saved-models/t5-large-2e-05-lr-pc_usr-test-f760cd2a/')
#     parser.add_argument('--eval_batch_size', help='batch size to use while eval', default=8)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    pc_dataset, tc_dataset, fed_dataset, dstc6_dataset, grade_data, dstc9_chatbot1_dataset, dstc9_chatbot2_dataset, dstc9_chatbot3_dataset, dstc9_chatbot4_dataset, dstc9_chatbot5_dataset, dstc9_chatbot6_dataset, dstc9_chatbot7_dataset, dstc9_chatbot8_dataset, dstc9_chatbot9_dataset, dstc9_chatbot10_dataset, holisticeval_dailydialog_context, holisticeval_dailydialog_fluency, predengage_data = read_data(args.data_dirpath)
    
    fed_dataset_formatted, tc_usr_dataset_formatted, pc_usr_dataset_formatted, grade_empatheticdialogues_dataset_formatted, grade_dailydialog_dataset_formatted, grade_convai_dataset_formatted, predengage_dailydialog_dataset_formatted, holisticeval_dailydialog_dataset_fluency_formatted, holisticeval_dailydialog_dataset_context_formatted, dstc9_chatbot1_dataset_formatted, dstc9_chatbot2_dataset_formatted, dstc9_chatbot3_dataset_formatted, dstc9_chatbot4_dataset_formatted, dstc9_chatbot5_dataset_formatted, dstc9_chatbot6_dataset_formatted, dstc9_chatbot7_dataset_formatted, dstc9_chatbot8_dataset_formatted, dstc9_chatbot9_dataset_formatted, dstc9_chatbot10_dataset_formatted = format_datasets(args.use_dstc6, pc_dataset, tc_dataset, fed_dataset, dstc6_dataset, grade_data, dstc9_chatbot1_dataset, dstc9_chatbot2_dataset, dstc9_chatbot3_dataset, dstc9_chatbot4_dataset, dstc9_chatbot5_dataset, dstc9_chatbot6_dataset, dstc9_chatbot7_dataset, dstc9_chatbot8_dataset, dstc9_chatbot9_dataset, dstc9_chatbot10_dataset, holisticeval_dailydialog_context, holisticeval_dailydialog_fluency, predengage_data)
    
    all_datasets_combined = combine_datasets(args.use_dstc6, fed_dataset_formatted, tc_usr_dataset_formatted, pc_usr_dataset_formatted, grade_empatheticdialogues_dataset_formatted, grade_dailydialog_dataset_formatted, grade_convai_dataset_formatted, predengage_dailydialog_dataset_formatted, holisticeval_dailydialog_dataset_fluency_formatted, holisticeval_dailydialog_dataset_context_formatted, dstc9_chatbot1_dataset_formatted, dstc9_chatbot2_dataset_formatted, dstc9_chatbot3_dataset_formatted, dstc9_chatbot4_dataset_formatted, dstc9_chatbot5_dataset_formatted, dstc9_chatbot6_dataset_formatted, dstc9_chatbot7_dataset_formatted, dstc9_chatbot8_dataset_formatted, dstc9_chatbot9_dataset_formatted, dstc9_chatbot10_dataset_formatted)
    
    print("\nCreating training format data...")
    output_dir = os.path.join(args.data_dirpath, "training")
    all_dataset_dicts = create_all_training_data(all_datasets_combined, output_dir, args.store_data_files)
    print("Created all training format datasets!")
    
    checkpoint_dirs = [item for item in os.listdir(args.model_checkpoints_dir) if os.path.isdir(os.path.join(args.model_checkpoints_dir, item))]
    with open(os.path.join(args.model_checkpoints_dir, "run_config.txt"), "r") as f1:
        run_config_lines = f1.readlines()
    for line in run_config_lines:
        if line.startswith("Test datasets: "):
            test_datasets = line.replace("Test datasets: ", "").replace("\n", "")
        if line.startswith("Model checkpoint: "):
            model_name = line.replace("Model checkpoint: ", "").replace("\n", "")
        if line.startswith("Unique run ID: "):
            unique_run_id = line.replace("Unique run ID: ", "").replace("\n", "")
    print(f"\nTest datasets: {test_datasets}")
    print("Tokenizing and getting test split...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_checkpoints_dir, checkpoint_dirs[0]))
    tokenized_test = get_test_split(all_dataset_dicts, test_datasets, tokenizer)
    
    for checkpoint_dirname in checkpoint_dirs:
        model_load_dir = os.path.join(args.model_checkpoints_dir, checkpoint_dirname)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_load_dir).to(device)
        model.eval()
    
        print(f"\n\nStarting evaluation with {checkpoint_dirname}...")
        answer_is_label = 0
        generated_answers = []
        for i in tqdm(range(len(tokenized_test))):
            dataset_here = tokenized_test[i]["dataset"]
            dataset_questions_types = dataset_questions_mapping[dataset_here].keys()
            for q_type in dataset_questions_types:
                dataset_question_here = dataset_questions_mapping[dataset_here][q_type]
                if dataset_question_here  in tokenized_test[i]["unlabelled_prompts"]:
                    question_here = q_type
            input_ids = torch.tensor([tokenized_test[i]["input_ids"]]).to(device)
            generation = model.generate(input_ids, do_sample=False, return_dict_in_generate=True, output_scores=True, max_new_tokens=20)
            sequences = generation.sequences
            scores = generation.scores

            decoded_answer = tokenizer.decode(sequences[0], skip_special_tokens=True)
            if decoded_answer in dataset_label2int_mapping[dataset_here][question_here].keys():
                is_label = True
                answer_is_label += 1
                decoded_int = dataset_label2int_mapping[dataset_here][question_here][decoded_answer]
            else:
                is_label = False
                decoded_int = None
            generation_dict = {"generated_answer": decoded_answer,
                               "correct_label_word": tokenized_test[i]["labels_word"],
                               "is_label": is_label,
                               "generated_answer_int": decoded_int,
                               "correct_label_int_rounded": tokenized_test[i]["labels_int_rounded"],
                               "correct_label_float_avg": tokenized_test[i]["labels_float_avg"],
                               "question_type": question_here,
                               "dataset": dataset_here,
                               "unlabelled_prompt": tokenized_test[i]["unlabelled_prompts"],
                              }
            generated_answers.append(generation_dict)
        
        with open(os.path.join(args.model_checkpoints_dir, f"generated_answers_{checkpoint_dirname}.json"), "w") as f1:
            json.dump(generated_answers, f1, indent=4)
            
        eval_dict = {}
        for each in generated_answers:
            if each["dataset"] not in eval_dict.keys():
                eval_dict[each["dataset"]] = {}
            if each["question_type"] not in eval_dict[each["dataset"]].keys():
                eval_dict[each["dataset"]][each["question_type"]] = {"predicted": [], "ground_truth": []}
            if each["is_label"] == True:
                eval_dict[each["dataset"]][each["question_type"]]["predicted"].append(each["generated_answer_int"])
                eval_dict[each["dataset"]][each["question_type"]]["ground_truth"].append(each["correct_label_float_avg"])

        with open(os.path.join(args.model_checkpoints_dir, f"eval_results_{checkpoint_dirname}.txt"), "w") as f1:
            f1.write("Unique (training) run ID: " + str(unique_run_id) + "\n")
            f1.write("Model: " + str(model_name) + "\n")
            f1.write("Checkpoint: " + str(checkpoint_dirname) + "\n")
            f1.write("Path: " + str(model_load_dir) + "\n")
            f1.write("\nTotal datapoints: " + str(len(tokenized_test)) + "\n")
            f1.write("Num. where answer is label: " + str(answer_is_label) + "\n")
            f1.write("Num. where answer isn't label: " + str(len(tokenized_test) - answer_is_label) + "\n")
            f1.write("% where answer is label: " + str((answer_is_label / len(tokenized_test)) * 100) + "\n\n")
            for dataset in eval_dict.keys():
                for question_type in eval_dict[dataset]:
                    s_rho, s_p = spearmanr(eval_dict[dataset][question_type]["predicted"], eval_dict[dataset][question_type]["ground_truth"])
                    p_rho, p_p = pearsonr(eval_dict[dataset][question_type]["predicted"], eval_dict[dataset][question_type]["ground_truth"])
                    f1.write(f"Dataset: {dataset}, Question Type: {question_type} --\nSpearmann: Rho: {s_rho} | p: {s_p}\nPearson: Rho: {p_rho} | p: {p_p}\n\n")
                    print(f"Dataset: {dataset}, Question Type: {question_type} --\nSpearmann: Rho: {s_rho} | p: {s_p}\nPearson: Rho: {p_rho} | p: {p_p}\n\n")
        print("Total datapoints: ", len(tokenized_test))
        print("Num. where answer is label: ", answer_is_label)
        print("Num. where answer isn't label: ", len(tokenized_test) - answer_is_label)
        print("% where answer is label: ", (answer_is_label / len(tokenized_test)) * 100)
        print(f"\nEvaluation complete for {checkpoint_dirname}!\n")
