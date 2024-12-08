import os
import csv
import torch
import pickle
import argparse
import open_clip

import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from data.cub import CustomCub2011_RTG
from data.oxford_flowers import OxfordFlowers_RTG
from data.stanford_cars import CarsDataset_RTG
from data.oxford_pets import OxfordPet_RTG
from data.cifar import CustomCIFAR10_RTG, CustomCIFAR100_RTG
from data.imagenet import ImageNetDataset_RTG

from config import cub_root, flowers_root, scars_root, pets_root, cifar10_root, cifar100_root, imagenet_root

RTG_dataset_funcs = {
    'cifar10': CustomCIFAR10_RTG,
    'cifar100': CustomCIFAR100_RTG,
    'imagenet_100': ImageNetDataset_RTG,
    'imagenet_1k': ImageNetDataset_RTG,
    'cub': CustomCub2011_RTG,
    'scars': CarsDataset_RTG,
    'pets': OxfordPet_RTG,
    'flowers': OxfordFlowers_RTG,
}

class RTG(nn.Module):
    def __init__( 
        self, 
        auxiliary_model,
        tags_path="Lexicon/Lexicon_tags.csv",
        attributes_path="Lexicon/Lexicon_attributes.csv",
        tags_textfeatures_path="Lexicon/Lexicon_tags_textfeatures.pt",
        attributes_textfeatures_path="Lexicon/Lexicon_attributes_textfeatures.pt",
        device='cuda',
    ):
        super().__init__()

        self.device = device
        self.auxiliary_model = auxiliary_model.to(self.device)

        # Load tags and attributes
        self.tags = self.load_csv(tags_path)
        self.attributes = self.load_csv(attributes_path)

        # Process or load text features
        self.tags_textfeatures = self.load_or_process_features(self.tags, tags_textfeatures_path)
        self.attributes_textfeatures = self.load_or_process_features(self.attributes, attributes_textfeatures_path)

    def load_csv(self, file_path):
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            return [row[0] for row in reader]

    def load_or_process_features(self, text_data, save_path):
        if not os.path.exists(save_path):
            return self.process_text_features(text_data, save_path)
        else:
            return torch.load(save_path, map_location=self.device)

    def process_text_features(self, text_data, save_path):
        print("No saved text features of Lexicon for auxiliary model.")
        batch_size = 128
        data_loader = DataLoader(text_data, batch_size=batch_size, shuffle=False)
        text_features = []

        for batch_text in tqdm(data_loader, desc=f'Processing {os.path.basename(save_path)}'):
            text_tokens = open_clip.tokenize(batch_text).to(self.device)
            with torch.no_grad():
                batch_features = self.auxiliary_model.encode_text(text_tokens)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            text_features.append(batch_features.cpu())  # Ensure tensor is on CPU before appending

        text_features = torch.cat(text_features, dim=0)
        torch.save(text_features, save_path)  
        return text_features.to(self.device)  

    def __call__(self, samples, num_tags=3, num_attributes=2, contrastive_th=0.2, return_tags=True, return_attributes=True):
        if return_tags:
            samples = self.forward_tags(samples, num_tags=num_tags, contrastive_th=contrastive_th)
        if return_attributes:
            samples = self.forward_attributes(samples, num_attributes=num_attributes, contrastive_th=contrastive_th)

        return samples

    def forward_tags(self, samples, num_tags=3, contrastive_th=0.2):
        # Get Image Features
        tags = []
        try:
            image_features = self.auxiliary_model.encode_image(
                samples["image"].to(self.device)
            )
        except:
            image_features = self.auxiliary_model.get_image_features(
                pixel_values=samples["image"]
            )
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_scores = self.auxiliary_model.logit_scale * image_features @ self.tags_textfeatures.T
        top_scores, top_indexes = text_scores.float().cpu().topk(k=num_tags, dim=-1)
        for scores, indexes in zip(top_scores, top_indexes):
            filter_indexes = indexes[scores >= contrastive_th]
            if len(filter_indexes) > 0:
                top_k_tags = [self.tags[index] for index in filter_indexes]
            else:
                top_k_tags = []
            tags.append(top_k_tags)
        samples[f"tags"] = tags
        return samples

    def forward_attributes(self, samples, num_attributes=2, contrastive_th=0.2):
        # Get Image Features
        attributes = []
        try:
            image_features = self.auxiliary_model.encode_image(
                samples["image"].to(self.device)
            )
        except:
            image_features = self.auxiliary_model.get_image_features(
                pixel_values=samples["image"]
            )
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_scores = self.auxiliary_model.logit_scale * image_features @ self.attributes_textfeatures.T
        top_scores, top_indexes = (
            text_scores.float().cpu().topk(k=num_attributes, dim=-1)
        )
        for scores, indexes in zip(top_scores, top_indexes):
            filter_indexes = indexes[scores >= contrastive_th]
            if len(filter_indexes) > 0:
                top_k_tags = [self.attributes[index] for index in filter_indexes]
            else:
                top_k_tags = []
            attributes.append(top_k_tags)
        samples[f"attributes"] = attributes
        return samples


def generate_tags_attributes_for_batches(dataloader, text_generator, output_file, idx_file, start_idx=0, num_tags=3, num_attributes=2):
    all_results = []  
    for batch_idx, samples in enumerate(tqdm(dataloader, desc="Processing batches", initial=start_idx, total=len(dataloader))):
        if batch_idx < start_idx:
            continue  # Skip already processed batches
        
        with torch.no_grad():
            outputs = text_generator(samples)
            batch_results = []

            for i, output in enumerate(outputs['tags']):
                tags_i = output[:num_tags]
                attributes_i = outputs['attributes'][i][:num_attributes]
                batch_results.append({
                    'image_id': samples["image_id"][i],
                    'attributes': attributes_i,
                    'tags': tags_i
                    
                })

            all_results.extend(batch_results)  

            with open(output_file, 'wb') as f:
                pickle.dump(all_results, f)

            with open(idx_file, 'w') as f_idx:
                f_idx.write(str(batch_idx + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RTG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--auxiliary_model_name', default='hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K', type=str, help='For convenience, we use the openclip library to import the auxiliary model ViT-H-CLIP')
    parser.add_argument('--dataset_name', default='cub', type=str, help='options: cifar10, cifar100, imagenet, cub, scars, pets, flowers, food')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_tags', default=3, type=int)
    parser.add_argument('--num_attributes', default=2, type=int)

    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--crop_pct', default=0.875, type=float)
    parser.add_argument('--interpolation', default=3, type=int) 
    parser.add_argument('--transform', default='imagenet', type=str) 

    parser.add_argument('--cub_root', default=cub_root, type=str)
    parser.add_argument('--flowers_root', default=flowers_root, type=str)
    parser.add_argument('--scars_root', default=scars_root, type=str)
    parser.add_argument('--pets_root', default=pets_root, type=str)
    parser.add_argument('--cifar10_root', default=cifar10_root, type=str)
    parser.add_argument('--cifar100_root', default=cifar100_root, type=str)
    parser.add_argument('--imagenet_root', default=imagenet_root, type=str)
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()

    # ----------------------
    # AUXILIARY MODEL
    # ----------------------
    auxiliary_model = open_clip.create_model_and_transforms(args.auxiliary_model_name)[0]


    # --------------------
    # DATASET & DATALOADER
    # --------------------
    sample_transform = open_clip.create_model_and_transforms(args.auxiliary_model_name)[2]
    dataset_root = getattr(args, f'{args.dataset_name}_root', None)
    dataset = RTG_dataset_funcs[args.dataset_name](root=dataset_root, transform=sample_transform)
    sample_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


    # ----------------------
    # RETRIEVED BASED TEXT GENERATOR
    # ----------------------
    text_generator = RTG(auxiliary_model=auxiliary_model)


    # ----------------------
    # OUTPUT FILE
    # ----------------------
    output_file = f'retrieved_text/{args.dataset_name}_retrieved_text.npy'
    idx_file = f'retrieved_text/{args.dataset_name}_retrieved_text.idx'


    # ----------------------
    # BREAKPOINT
    # ----------------------
    # Note: to continue from the breakpoint.
    start_idx = 0
    if os.path.exists(idx_file):
        with open(idx_file, 'r') as f_idx:
            start_idx = int(f_idx.read().strip()) 


    # ----------------------
    # GENERATE TEXT
    # ----------------------
    generate_tags_attributes_for_batches(sample_dataloader, text_generator, output_file, idx_file, start_idx=start_idx, num_tags=args.num_tags, num_attributes=args.num_attributes)