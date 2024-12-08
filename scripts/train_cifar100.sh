CUDA_VISIBLE_DEVICES=0 python retrieval_based_text_generation.py \
 --dataset_name "cifar100" \

CUDA_VISIBLE_DEVICES=0 python cross_modality_coteaching.py \
 --dataset_name "cifar100" \
 --experiment_name "cifar100_seed4" \
 --seed 4 \
 --output_dir 'exp' \
