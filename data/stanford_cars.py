import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from PIL import Image

from data.data_utils import subsample_instances
from utils import process_file, construct_text
from clip import clip

tag_probability_words = ["most likely", "probably", "perhaps"]
attribute_probability_words = ["Likely", "Perhaps", "Could be"]

class CARS_Base(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, train=True, limit=0, root=None, transform=None):

        metas = os.path.join(root, 'stanford_cars/devkit/cars_train_annos.mat') if train else os.path.join(root, 'stanford_cars/devkit/cars_test_annos_withlabels.mat')
        root = os.path.join(root, 'stanford_cars/cars_train/') if train else os.path.join(root, 'stanford_cars/cars_test/')

        self.loader = default_loader
        self.root = root
        self.data = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data.append(root + img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __len__(self):
        return len(self.data)
    
class CarsDataset_RTG(CARS_Base):
    def __init__(self, train=True, limit=0, root=None, transform=None):
        super().__init__(train, limit, root, transform)

    def __getitem__(self, idx):

        path = self.data[idx]
        image = Image.open(path).convert('RGB')
        image_id = os.path.basename(path)
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return {"image": image, "target": target, "image_id": image_id, "id":idx}
    
class CarsDataset(CARS_Base):
    def __init__(self, train=True, limit=0, root=None, retrieved_text_path=None, transform=None, text_transform=None, num_tags=0, num_attributes=0):
        self.retrieved_text = process_file(retrieved_text_path)
        self.text_transform = text_transform
        self.num_tags = num_tags
        self.num_attributes = num_attributes
        super().__init__(train, limit, root, transform)

    def safe_tokenize(self, text):
        while True:
            try:
                token = clip.tokenize(text)
                return token
            except RuntimeError:
                words = text.split()
                if len(words) <= 1:  
                    return None
                text = " ".join(words[:-1]) 

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1
        path = self.data[idx]
        image_id = os.path.basename(path)

        retrieved_text = self.retrieved_text[image_id]

        tags = retrieved_text['tags'][:self.num_tags]
        attributes = retrieved_text['attributes'][:self.num_attributes]

        base_text = 'A photo of a car'

        # Constructing tag text
        tag_text = []
        for i, tag in enumerate(tags):
            tag_text.append(f"{tag_probability_words[i]} a {tag}")
        
        # Constructing attribute text
        attribute_text = []
        for i, attribute in enumerate(attributes):
            attribute_text.append(f"{attribute_probability_words[i]} {attribute}")

        descriptive_text = base_text + ", "  + ", ".join(tag_text) + ". " + ". ".join(attribute_text) + "."

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        if self.text_transform is not None:
            descriptive_text = self.text_transform(descriptive_text)
            descriptive_text_token = [self.safe_tokenize(t) for t in descriptive_text]
        else:
            descriptive_text_token = self.safe_tokenize(descriptive_text)

        return image, target, image_id, descriptive_text_token, idx
    
def subsample_dataset(dataset, idxs):

    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cars = np.array(include_classes) + 1     # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_scars_datasets(train_transform, 
                       test_transform, 
                       text_transform, 
                       train_classes=range(160), 
                       prop_train_labels=0.8,
                       split_train_val=False, 
                       seed=0,
                       args=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CarsDataset(root=args.scars_root, retrieved_text_path=args.scars_retrieved_text_path, transform=train_transform, text_transform=text_transform, train=True, num_tags=args.num_tags, num_attributes=args.num_attributes)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CarsDataset(root=args.scars_root, retrieved_text_path=args.scars_retrieved_text_path, transform=test_transform, text_transform=None, train=False, num_tags=args.num_tags, num_attributes=args.num_attributes)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets

if __name__ == '__main__':

    x = get_scars_datasets(None, None, train_classes=range(98), prop_train_labels=0.5, split_train_val=False)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].target))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].target))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')