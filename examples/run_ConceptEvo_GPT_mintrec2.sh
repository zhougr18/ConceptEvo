#!/usr/bin bash

for dataset in 'MIntRec2.0'
do
    for seed in 0 1 2
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'ConceptEvo_GPT' \
        --method 'ConceptEvo_GPT' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '0' \
        --video_feats_path 'swin_roi.pkl' \
        --audio_feats_path 'wavlm_feats.pkl' \
        --text_backbone 'bert-large-uncased' \
        --bert_base_uncased_path '/models/bert-large-uncased' \
        --config_file_name 'ConceptEvo_GPT_mintrec2' \
        --results_file_name 'results_ConceptEvo_GPT_mintrec2.csv'
    done
done