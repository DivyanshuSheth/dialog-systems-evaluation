#!/bin/bash

## Get TC, PC, FED, DSTC6, HolisticEval, PredEngage, GRADE datasets ##
mkdir -p "data"
cd "data"
echo "Downloading datasets..."
wget -q "http://shikib.com/tc_usr_data.json" -O "tc_usr_data.json"
wget -q "http://shikib.com/pc_usr_data.json" -O "pc_usr_data.json"
wget -q "http://shikib.com/fed_data.json" -O "fed_data.json"
wget -q "https://www.dropbox.com/s/mx9mv7wb3c6xzrb/human_rating_scores.txt?dl=1" -O "dstc6-human_rating_scores.txt"
wget -q "https://raw.githubusercontent.com/alexzhou907/dialogue_evaluation/master/context_data_release.csv" -O "holisticeval-context-data.csv"
wget -q "https://raw.githubusercontent.com/alexzhou907/dialogue_evaluation/master/fluency_data_release.csv" -O "holisticeval-fluency-data.csv"
wget -q "https://raw.githubusercontent.com/PlusLabNLP/PredictiveEngagement/master/data/Eng_Scores_queries_gen_gtruth_replies.csv" -O "predictiveengage-eng-scores-gen-gtruth-data.csv"
wget -q "https://raw.githubusercontent.com/li3cmz/GRADE/main/evaluation/human_score/human_judgement.json" -O "grade_dailydialog_convai_empathetic_data.json"
## 

## Get DSTC9 dataset ##
wget -q "https://raw.githubusercontent.com/ictnlp/DialoFlow/main/FlowScore/data/chatbot1.json" -O "dstc9-chatbot1.json"
wget -q "https://raw.githubusercontent.com/ictnlp/DialoFlow/main/FlowScore/data/chatbot2.json" -O "dstc9-chatbot2.json"
wget -q "https://raw.githubusercontent.com/ictnlp/DialoFlow/main/FlowScore/data/chatbot3.json" -O "dstc9-chatbot3.json"
wget -q "https://raw.githubusercontent.com/ictnlp/DialoFlow/main/FlowScore/data/chatbot4.json" -O "dstc9-chatbot4.json"
wget -q "https://raw.githubusercontent.com/ictnlp/DialoFlow/main/FlowScore/data/chatbot5.json" -O "dstc9-chatbot5.json"
wget -q "https://raw.githubusercontent.com/ictnlp/DialoFlow/main/FlowScore/data/chatbot6.json" -O "dstc9-chatbot6.json"
wget -q "https://raw.githubusercontent.com/ictnlp/DialoFlow/main/FlowScore/data/chatbot7.json" -O "dstc9-chatbot7.json"
wget -q "https://raw.githubusercontent.com/ictnlp/DialoFlow/main/FlowScore/data/chatbot8.json" -O "dstc9-chatbot8.json"
wget -q "https://raw.githubusercontent.com/ictnlp/DialoFlow/main/FlowScore/data/chatbot9.json" -O "dstc9-chatbot9.json"
wget -q "https://raw.githubusercontent.com/ictnlp/DialoFlow/main/FlowScore/data/chatbot10.json" -O "dstc9-chatbot10.json"
##

echo -e "All datasets downloaded!"

cd ../