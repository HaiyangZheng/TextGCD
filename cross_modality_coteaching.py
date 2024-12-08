import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import SGD
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.get_datasets import get_datasets, get_class_splits
from data.augmentations import get_transform, tokenize_with_augmentation
from model import CustomCLIP, CustomCosineAnnealingLR, ImageViewGenerator, TextViewGenerator, load_clip_to_cpu
from utils import init_experiment, select_confident_samples, image_text_contrastive_loss, simgcd_loss, coteaching_pseudolabel_loss, evaluate_accuracy

from config import cub_root, cub_retrieved_text_path, flowers_root, flowers_retrieved_text_path, scars_root, scars_retrieved_text_path, \
                   pets_root, pets_retrieved_text_path, cifar10_root, cifar10_retrieved_text_path, cifar100_root, cifar100_retrieved_text_path, imagenet_root, imagenet_retrieved_text_path, osr_split_dir

def train_one_epoch(args, logger, writer, loader, model, optimizer, scheduler, epoch, selected_samples_t, selected_samples_i):
    model.train()
    total_loss = 0.0
    total_loss_base = 0.0
    total_loss_con = 0.0
    total_loss_pseudo_image = 0.0
    total_loss_pseudo_text = 0.0
    

    teacher_temp_schedule = np.concatenate((np.linspace(args.tau_t_start, args.tau_t_end, args.warmup_teacher_temp_epochs),np.ones(args.epochs - args.warmup_teacher_temp_epochs) * args.tau_t_end))

    param_group_names = ['classifier_head', 'base_parameters']

    for batch_idx, (images, labels, img_id, descriptive_text_token, _, mask) in enumerate(tqdm(loader, desc="Training")):
        mask = mask[:, 0]
        labels, mask = labels.cuda(non_blocking=True), mask.cuda(non_blocking=True).bool()
        images = torch.cat(images, dim=0).cuda(non_blocking=True)
        descriptive_text_token = torch.cat(descriptive_text_token, dim=0).squeeze(1).cuda(non_blocking=True)

        logits_image, logits_text, image_features, descriptive_text_features = model(images, descriptive_text_token)

        loss_con = image_text_contrastive_loss(image_features, descriptive_text_features, model.model.logit_scale.exp(), args)

        loss_base = simgcd_loss(logits_image, logits_text, labels, mask, teacher_temp_schedule, epoch, args)

        loss_pseudo_image = coteaching_pseudolabel_loss(selected_samples_t, logits_image, img_id, args)
        
        loss_pseudo_text = coteaching_pseudolabel_loss(selected_samples_i, logits_text, img_id, args)

        loss = loss_base + loss_con + loss_pseudo_image + loss_pseudo_text

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_base += loss_base.item()
        total_loss_con += loss_con.item()
        total_loss_pseudo_image += loss_pseudo_image.item()
        total_loss_pseudo_text += loss_pseudo_text.item()
        

        iter_idx = epoch * len(train_loader) + batch_idx
        writer.add_scalars('Loss', {
            'loss_base': loss_base.item(),
            'loss_con': loss_con.item(),
            'loss_pseudo_image': loss_pseudo_image.item(),
            'loss_pseudo_text': loss_pseudo_text.item(),
            'total_loss': loss.item()
        }, iter_idx)

    scheduler.step()
    logger.info(
    f"Epoch {epoch+1}/{args.epochs}, Total Loss: {total_loss / len(loader):.4f}, "
    f"Base Loss: {total_loss_base / len(loader):.4f}, Con Loss: {total_loss_con / len(loader):.4f}, "
    f"Pseudo Loss Image: {total_loss_pseudo_image / len(loader):.4f}, Pseudo Loss Text: {total_loss_pseudo_text / len(loader):.4f}"
    )
    for idx, param_group in enumerate(optimizer.param_groups):
        logger.info(f"   Param Group: {param_group_names[idx]}, Learning Rate: {param_group['lr']:.4f}")


def test(model, test_loader, args):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _, descriptive_text_token, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        descriptive_text_token = descriptive_text_token.squeeze(1).cuda(non_blocking=True)
        with torch.no_grad():
            logits_image, logits_text, _, _ = model(images, descriptive_text_token)

            classifier_image_probs = F.softmax(logits_image, dim=-1)
            classifier_text_probs = F.softmax(logits_text, dim=-1)

            averaged_probs = 0.5 * classifier_image_probs + 0.5 * classifier_text_probs

            preds.append(averaged_probs.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    total_acc, old_acc, new_acc = evaluate_accuracy(preds, targets, mask)

    return total_acc, old_acc, new_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CCT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output_dir', default='exp', type=str)
    parser.add_argument('--experiment_name', default='cub_trueepoch', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--dataset_name', default='cub', type=str, help='options: cifar10, cifar100, imagenet_100, cub, scars, aircraft, herb, pets, flowers, food')
    parser.add_argument('--backbone_name', default='ViT-B/16', type=str, help="choose from 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'")

    parser.add_argument('--cub_root', default=cub_root, type=str)
    parser.add_argument('--cub_retrieved_text_path', default=cub_retrieved_text_path, type=str)
    parser.add_argument('--flowers_root', default=flowers_root, type=str)
    parser.add_argument('--flowers_retrieved_text_path', default=flowers_retrieved_text_path, type=str)
    parser.add_argument('--scars_root', default=scars_root, type=str)
    parser.add_argument('--scars_retrieved_text_path', default=scars_retrieved_text_path, type=str)
    parser.add_argument('--pets_root', default=pets_root, type=str)
    parser.add_argument('--pets_retrieved_text_path', default=pets_retrieved_text_path, type=str)
    parser.add_argument('--cifar10_root', default=cifar10_root, type=str)
    parser.add_argument('--cifar10_retrieved_text_path', default=cifar10_retrieved_text_path, type=str)
    parser.add_argument('--cifar100_root', default=cifar100_root, type=str)
    parser.add_argument('--cifar100_retrieved_text_path', default=cifar100_retrieved_text_path, type=str)
    parser.add_argument('--imagenet_root', default=imagenet_root, type=str)
    parser.add_argument('--imagenet_retrieved_text_path', default=imagenet_retrieved_text_path, type=str)
    parser.add_argument('--osr_split_dir', type=str, default=osr_split_dir)

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--base_lr', default=0.0005, type=float)
    parser.add_argument('--classifier_lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--prop_train_labels', default=0.5, type=float, help="Proportion of labeled samples selected for known categories.")
    parser.add_argument('--use_ssb_splits', default=True, action='store_true', help="Follow SimGCD use this for Semantic Shift Benchmarks.")

    parser.add_argument('--transform', default='imagenet', type=str, help="The name of image transform.") 
    parser.add_argument('--n_views', default=2, type=int, help="Two views for contrastive learning.")

    parser.add_argument('--selecting_ratio', default=0.6, type=float, help="It is r in the paper. It is coefficient for selecting high-confidence samples.")
    parser.add_argument('--lambda_loss', default=0.2, type=float, help="It is the coefficient in the loss_base")
    parser.add_argument('--warm_up_epochs', default=10, type=int, help="It is the e_w in paper.")
    parser.add_argument('--class_aligning_epochs', default=5, type=int, help="It is the e_a in paper.") 

    parser.add_argument('--num_attributes', default=2, type=int, help="Numbers of selected attributes for descriptive text.")
    parser.add_argument('--num_tags', default=3, type=int, help="Numbers of selected tags for descriptive text.")

    parser.add_argument('--tau_s', default=0.1, type=float, help="Temperature for supervised classification")
    parser.add_argument('--tau_u', default=0.05, type=float, help="Temperature for unsupervised cluster loss. It is the student temperature in SimGCD.")
    # Note: for cifar10, tau_t_start sets to 0.07, and tau_t_end sets to 0.04
    parser.add_argument('--tau_t_start', default=0.035, type=float, help='Initial value for the teacher temperature. Please refer to SimGCD.')
    parser.add_argument('--tau_t_end', default=0.02, type=float, help='Final value (after linear warmup)of the teacher temperature. Please refer to SimGCD.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--memax_weight', default=2, type=float, help="This is the coefficient for the mean-entropy maximization regularizer, as referenced in SimGCD.")

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args, logger, writer = init_experiment(args)

    
    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    logger.info(f"Loading CLIP (backbone: {args.backbone_name})")
    backbone = load_clip_to_cpu(args.backbone_name).float()

    logger.info("Building custom CLIP")
    model = CustomCLIP(backbone, args.mlp_out_dim).to(args.device)


    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    logger.info("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        param.requires_grad_(False)

    for name, param in model.named_parameters():
        if "transformer.resblocks.11" in name:
            param.requires_grad_(True)
        if "visual.proj" in name:
            param.requires_grad_(True)
        if "text_projection" in name:
            param.requires_grad_(True)
        if "image_classifier" in name:
            param.requires_grad_(True)
        if "text_classifier" in name:
            param.requires_grad_(True)

    params_names = [name for name, param in model.named_parameters() if param.requires_grad]
    logger.info("Parameters that require gradients: %s", params_names)


    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    # transform: text-[text,text] image-[image, image]
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ImageViewGenerator(base_transform=train_transform, n_views=args.n_views)
    text_transform = TextViewGenerator(base_transform=tokenize_with_augmentation, n_views=args.n_views)
            

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, _ = get_datasets(args.dataset_name, train_transform, test_transform, text_transform, args)
    logger.info(f"len of train dataset: {len(train_dataset)}")
    logger.info(f"len of test dataset: {len(unlabelled_train_examples_test)}")


    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))


    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)
    # val_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)


    # --------------------
    # OPTIMIZER&LR_SCHEDULER
    # --------------------
    classifier_params_train = [p for name, p in model.named_parameters() if "classifier" in name and p.requires_grad]
    classifier_params_train_name = [name for name, p in model.named_parameters() if "classifier" in name and p.requires_grad]
    logger.info("Parameters in classifier with big lr: %s", classifier_params_train_name)

    # params of last layers
    other_params_train = [p for name, p in model.named_parameters() if "classifier" not in name and p.requires_grad]

    # set learning rates
    classifier_lr = args.classifier_lr # lr for classifiers
    base_lr = args.base_lr  # lr for last layers

    # Create the optimizer, setting different learning rates for different parameter groups
    optimizer_train = SGD([
        {'params': classifier_params_train, 'lr': classifier_lr},
        {'params': other_params_train, 'lr': base_lr}
    ])

    scheduler_train = CustomCosineAnnealingLR(optimizer_train, classifier_params_train, T_max=args.epochs, eta_min=classifier_lr*1e-3)


    # --------------------
    # CO-TEACHING PARAMETER
    # --------------------    
    selecting_nums = math.floor(len(test_loader_unlabelled.dataset) / args.mlp_out_dim * args.selecting_ratio)
    logger.info(f"Selecting {selecting_nums} high-confidence samples for co-teaching.")


    # --------------------
    # TRAIN
    # -------------------- 
    best_test_acc_lab = 0
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        
        # co-teaching stage
        if epoch > args.class_aligning_epochs + args.warm_up_epochs:
            selected_samples_i = select_confident_samples(model, test_loader_unlabelled, selecting_nums)
            selected_samples_t = select_confident_samples(model, test_loader_unlabelled, selecting_nums, from_image=False)
        # class_aligning stage
        elif epoch > args.warm_up_epochs:
            selected_samples_i = None
            selected_samples_t = select_confident_samples(model, test_loader_unlabelled, selecting_nums, from_image=False)
        # warm_up stage
        else:
            selected_samples_i = None
            selected_samples_t = None
        if selected_samples_i:
            logger.info(f"len of image selected samples: {len(selected_samples_i)}")
        if selected_samples_t:
            logger.info(f"len of text selected samples: {len(selected_samples_t)}")

        train_one_epoch(args, logger, writer, train_loader, model, optimizer_train, scheduler_train, epoch, selected_samples_t, selected_samples_i)

        total_acc, old_acc, new_acc = test(model, test_loader_unlabelled, args)
        logger.info(f"Weighted Accuracies: All {total_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}")

        writer.add_scalar('Accuracy/All', total_acc, epoch)
        writer.add_scalar('Accuracy/Old', old_acc, epoch)
        writer.add_scalar('Accuracy/New', new_acc, epoch)

        # checkpoint = {
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer_train.state_dict(),
        #     'scheduler_state_dict': scheduler_train.state_dict()
        # }
        # torch.save(checkpoint, args.model_path)

        # if total_acc > best_test_acc_lab:
        #     best_test_acc_lab = total_acc
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer_train.state_dict(),
        #         'scheduler_state_dict': scheduler_train.state_dict()
        #     }
        #     torch.save(checkpoint, args.model_path)
        #     logger.info("Saved model!")

    writer.close()