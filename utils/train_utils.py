import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment as linear_assignment


def select_confident_samples(model, dataloader, num_samples, from_image=True):

    logits_list = []
    image_id_list = []

    for images, _, image_id, descriptive_text_token, _ in dataloader:
        image_id_list.append(image_id)
        images = images.cuda(non_blocking=True)
        descriptive_text_token = descriptive_text_token.squeeze(1).cuda(non_blocking=True)

        with torch.no_grad():
            if from_image:
                logits, _, _, _ = model(images, descriptive_text_token)
            else:
                _, logits, _, _ = model(images, descriptive_text_token)
        
        logits_list.append(logits)

    # Apply softmax with temperature scaling to logits to highlight the dominant class.
    logits_all = torch.cat(logits_list, dim=0)
    logits_all = F.softmax(logits_all / 0.01, dim=-1)

    # Select the indices of the top 'num_samples' most confident samples for each class based on their logits.
    top_k_per_cls = [logits_all[:, i].argsort(descending=True)[:num_samples] for i in range(logits_all.shape[1])]

    # Initialize an empty dictionary to store confident samples.
    confident_samples_map = {}
    # Flatten the list of image IDs from nested lists to a single list.
    image_id_flattened = [img_id for sublist in image_id_list for img_id in sublist]
    for idx, image_indices in enumerate(top_k_per_cls):
        for image_index in image_indices:
            image_id = image_id_flattened[image_index]
            # Map image ID of the confident sample to its corresponding class index.
            confident_samples_map[image_id] = idx

    return confident_samples_map


def image_text_contrastive_loss(image_features, text_features, tau_c, args):

    logits_per_image = tau_c * image_features @ text_features.t()
    targets = torch.arange(len(image_features),dtype=torch.long, device=args.device)

    loss_con = F.cross_entropy(logits_per_image, targets)

    return loss_con


def simgcd_distillLoss(student_out, teacher_out):
    total_loss = 0
    n_loss_terms = 0
    for iq, q in enumerate(teacher_out):
        for v in range(len(student_out)):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
    total_loss /= n_loss_terms
    return total_loss


def simgcd_loss(logits_image, logits_text, labels, mask, teacher_temp_schedule, epoch, args):


    sup_labels = torch.cat([labels[mask] for _ in range(2)], dim=0)

    ## image loss
    sup_logits_image = torch.cat([f[mask] for f in (logits_image / args.tau_s).chunk(2)], dim=0)
    loss_cls_image = nn.CrossEntropyLoss()(sup_logits_image, sup_labels)

    # Note: args.tau_u is the student_temp in SimGCD, tau_t_(start & end) is the teacher_temp in SimGCD, please refer to SimGCD/model.py/DistillLoss()
    student_out_image = logits_image / args.tau_u
    student_out_image = student_out_image.chunk(2)
    teacher_out_image = logits_image.detach()
    teacher_out_image = F.softmax(teacher_out_image / teacher_temp_schedule[epoch], dim=-1)
    teacher_out_image = teacher_out_image.chunk(2)
    loss_cluster_image = simgcd_distillLoss(student_out_image, teacher_out_image)

    avg_probs_image = (logits_image / args.tau_u).softmax(dim=1).mean(dim=0)
    me_max_loss_image = - torch.sum(torch.log(avg_probs_image**(-avg_probs_image))) + math.log(float(len(avg_probs_image)))
    loss_cluster_image += args.memax_weight * me_max_loss_image

    loss_base_image = args.lambda_loss * loss_cls_image + (1-args.lambda_loss) * loss_cluster_image

    ## text loss
    sup_logits_text = torch.cat([f[mask] for f in (logits_text / args.tau_s).chunk(2)], dim=0)
    loss_cls_text = nn.CrossEntropyLoss()(sup_logits_text, sup_labels)  

    student_out_text = logits_text / args.tau_u
    student_out_text = student_out_text.chunk(2)
    teacher_out_text = logits_text.detach()
    teacher_out_text = F.softmax(teacher_out_text / teacher_temp_schedule[epoch], dim=-1)
    teacher_out_text = teacher_out_text.chunk(2)
    loss_cluster_text = simgcd_distillLoss(student_out_text, teacher_out_text)

    avg_probs_text = (logits_text / args.tau_u).softmax(dim=1).mean(dim=0)
    me_max_loss_text = - torch.sum(torch.log(avg_probs_text**(-avg_probs_text))) + math.log(float(len(avg_probs_text)))
    loss_cluster_text += args.memax_weight * me_max_loss_text 

    loss_base_text = args.lambda_loss * loss_cls_text + (1-args.lambda_loss) * loss_cluster_text     

    return loss_base_image + loss_base_text


def coteaching_pseudolabel_loss(selected_samples, logits, images_id, args):

    pseudo_labels = []
    selected_logits = []

    # Duplicate the image ID list for 2 views
    doubled_images_id = images_id + images_id
    if selected_samples:
        # Collect pseudo labels and corresponding logits
        for idx, img_id in enumerate(doubled_images_id):
            if img_id in selected_samples:
                pseudo_labels.append(selected_samples[img_id])
                selected_logits.append(logits[idx])

    # Compute cross-entropy loss if there are pseudo labels available
    if pseudo_labels:
        pseudo_labels = torch.tensor(pseudo_labels, device=args.device)
        selected_logits = torch.stack(selected_logits).to(args.device)
        selected_logits /= args.tau_s  # Apply temperature scaling
        loss = nn.CrossEntropyLoss()(selected_logits, pseudo_labels)
    else:
        # Return zero loss if no pseudo labels are present
        loss = torch.zeros((), device=args.device)

    return loss


def evaluate_accuracy(preds, targets, mask):
    mask = mask.astype(bool)
    targets = targets.astype(int)
    preds = preds.astype(int)

    old_classes_gt = set(targets[mask])
    new_classes_gt = set(targets[~mask])

    assert preds.size == targets.size
    D = max(preds.max(), targets.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds.size):
        w[preds[i], targets[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind])
    total_instances = preds.size

    total_acc /= total_instances

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc