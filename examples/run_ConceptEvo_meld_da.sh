#!/usr/bin bash

for dataset in 'MELD-DA'
do
    for seed in 0 1 2
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'ConceptEvo' \
        --method 'ConceptEvo' \
        --data_mode 'multi-class' \
        --tune \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '0' \
        --video_feats_path 'swin_feats.pkl' \
        --audio_feats_path 'wavlm_feats.pkl' \
        --text_backbone 'bert-large-uncased' \
        --bert_base_uncased_path '/models/bert-large-uncased' \
        --config_file_name 'mult_meld_da' \
        --results_file_name 'results_ConceptEvo_meld_da.csv'
    done
done