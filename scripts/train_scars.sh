CUDA_VISIBLE_DEVICES=0 python retrieval_based_text_generation.py \
 --dataset_name "scars" \

CUDA_VISIBLE_DEVICES=0 python cross_modality_coteaching.py \
 --dataset_name "scars" \
 --experiment_name "scars_seed4" \
 --seed 4 \
 --output_dir 'exp' \