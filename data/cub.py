import os
import pandas as pd
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from PIL import Image

from data.data_utils import subsample_instances
from utils import process_file, construct_text
from clip import clip

tag_probability_words = ["most likely", "probably", "perhaps"]
attribute_probability_words = ["Likely", "Perhaps", "Could be"]

class CustomCub2011_Base(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, 
                 root, 
                 train=True, 
                 transform=None, 
                 target_transform=None, 
                 loader=default_loader, 
                 download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)


class CustomCub2011_RTG(CustomCub2011_Base):
    def __init__(self, root=None, train=True, transform=None, target_transform=None, loader=default_loader, download=True):
        super().__init__(root, train, transform, target_transform, loader, download)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = Image.open(path).convert('RGB')
        image_id = os.path.basename(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"image": img, "target": target, "image_id": image_id, "id":idx}


class CustomCub2011(CustomCub2011_Base):
    def __init__(self, retrieved_text_path=None, root=None, train=True, transform=None, target_transform=None, text_transform=None, loader=default_loader, download=True, num_tags=0, num_attributes=0):
        
        self.retrieved_text = process_file(retrieved_text_path)
        self.text_transform = text_transform
        self.num_tags = num_tags
        self.num_attributes = num_attributes
        super().__init__(root, train, transform, target_transform, loader, download)

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
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        image_id = os.path.basename(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        retrieved_text = self.retrieved_text[image_id]

        tags = retrieved_text['tags'][:self.num_tags]
        attributes = retrieved_text['attributes'][:self.num_attributes]

        base_text = 'A photo of a bird'

        # Constructing tag text
        tag_text = []
        for i, tag in enumerate(tags):
            tag_text.append(f"{tag_probability_words[i]} {tag}")
        
        # Constructing attribute text
        attribute_text = []
        for i, attribute in enumerate(attributes):
            attribute_text.append(f"{attribute_probability_words[i]} {attribute}")

        descriptive_text = base_text + ", "  + ", ".join(tag_text) + ". " + ". ".join(attribute_text) + "."

        if self.text_transform is not None:
            descriptive_text = self.text_transform(descriptive_text)
            descriptive_text_token = [self.safe_tokenize(t) for t in descriptive_text]
        else:
            descriptive_text_token = self.safe_tokenize(descriptive_text)

        return img, target, image_id, descriptive_text_token, self.uq_idxs[idx]


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    # TODO: For now have no target transform
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cub_datasets(train_transform, 
                     test_transform, 
                     text_transform, 
                     train_classes=range(160), 
                     prop_train_labels=0.8,
                     split_train_val=False, 
                     seed=0, 
                     download=False,
                     args=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCub2011(root=args.cub_root, retrieved_text_path=args.cub_retrieved_text_path,transform=train_transform, text_transform=text_transform, train=True, download=download, num_tags=args.num_tags, num_attributes=args.num_attributes)

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
    test_dataset = CustomCub2011(root=args.cub_root, retrieved_text_path=args.cub_retrieved_text_path, transform=test_transform, text_transform=None, train=False, num_tags=args.num_tags, num_attributes=args.num_attributes)

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

    x = get_cub_datasets(None, None, split_train_val=False,
                         train_classes=range(100), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].data["target"].values))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].data["target"].values))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')