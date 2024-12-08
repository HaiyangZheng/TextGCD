import os
import random
import logging
from datetime import datetime
import numpy as np
import torch
from tensorboardX import SummaryWriter


# Set random seeds for reproducibility
def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Set up directories for the experiment
def set_experiment_directories(experiment_name, output_dir):
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    experiment_dir = os.path.join(output_dir, f"{experiment_name}-{current_time}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    subdirs = ['logs', 'models', 'tensorboard_logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    return experiment_dir


# Set up logging configuration
def set_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


# Set computation device (GPU or CPU)
def set_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Set up TensorBoard for logging metrics
def set_tensorboard(tensorboard_log_dir, args):
    writer = SummaryWriter(tensorboard_log_dir)
    args_str = '\n '.join(f'{k}={v}' for k, v in vars(args).items())
    writer.add_text('Arguments', args_str)
    return writer


# Log initial experiment settings
def log_experiment_settings(logger, args):
    action = "Training" if not args.evaluate else "Evaluating"
    logger.info(f"{action} {args.experiment_name} with the following settings:")
    args_str = '\n '.join(f'{k}={v}' for k, v in vars(args).items())
    logger.info(f'Command-line arguments: {args_str}')


# validate evaluation path
def validate_evaluation_path(evaluate_path):
    essential_paths = [
        evaluate_path,
        os.path.join(evaluate_path, 'logs', 'log.txt'),
        os.path.join(evaluate_path, 'models', 'model.pth')
    ]
    for path in essential_paths:
        if not os.path.exists(path):
            raise ValueError(f"Invalid path: {path}. Please provide a valid path for evaluation.")


# Initialize the experiment environment
def init_experiment(args):
    set_random_seeds(args.seed)
    
    if not args.evaluate:
        if hasattr(args, 'interrupted_path') and os.path.exists(args.interrupted_path):
            experiment_dir = args.interrupted_path
        else:
            experiment_dir = set_experiment_directories(args.experiment_name, args.output_dir)
        
        args.log_path = os.path.join(experiment_dir, 'logs', 'log.txt')
        args.model_path = os.path.join(experiment_dir, 'models', 'model.pth')
        tensorboard_log_dir = os.path.join(experiment_dir, 'tensorboard_logs')
    else:
        validate_evaluation_path(args.evaluate_path)
        args.log_path = os.path.join(args.evaluate_path, 'logs', 'log.txt')
        args.model_path = os.path.join(args.evaluate_path, 'models', 'model.pth')
    
    logger = set_logging(args.log_path)
    args.device = set_device()
    log_experiment_settings(logger, args)
    
    if not args.evaluate:
        writer = set_tensorboard(tensorboard_log_dir, args)
        return args, logger, writer

    return args, logger