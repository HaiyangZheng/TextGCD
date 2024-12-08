CUDA_VISIBLE_DEVICES=0 python retrieval_based_text_generation.py \
 --dataset_name "cub" \

CUDA_VISIBLE_DEVICES=0 python cross_modality_coteaching.py \
 --dataset_name "cub" \
 --experiment_name "cub_seed4" \
 --seed 4 \
 --output_dir 'exp' \