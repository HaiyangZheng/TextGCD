CUDA_VISIBLE_DEVICES=0 python retrieval_based_text_generation.py \
 --dataset_name "cifar10" \

CUDA_VISIBLE_DEVICES=0 python cross_modality_coteaching.py \
 --dataset_name "cifar10" \
 --experiment_name "cifar10_seed4" \
 --seed 4 \
 --tau_u 0.1 \
 --tau_t_start 0.07 \
 --tau_t_end 0.04 \
 --output_dir 'exp' \