from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np

from data.data_utils import subsample_instances
from utils import process_file
from clip import clip

tag_probability_words = ["most likely", "probably", "perhaps"]
attribute_probability_words = ["Likely", "Perhaps", "Could be"]

class CustomCIFAR10_Base(CIFAR10):

    def __init__(self, root=None, train=True, transform=None, target_transform=None, download=False):

        super(CustomCIFAR10_Base, self).__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.uq_idxs = np.array(range(len(self)))

    def __len__(self):
        return len(self.targets)
    
class CustomCIFAR10_RTG(CustomCIFAR10_Base):
    def __init__(self, root=None, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        
    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return {"image": img, "target": label, "image_id": str(uq_idx), "id": uq_idx}

class CustomCIFAR10(CustomCIFAR10_Base):
    def __init__(self, retrieved_text_path, root=None, train=True, transform=None, target_transform=None, text_transform=None, download=False, num_tags=0, num_attributes=0):
        self.retrieved_text = process_file(retrieved_text_path)
        self.text_transform = text_transform    
        self.num_tags = num_tags
        self.num_attributes = num_attributes    
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

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

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        retrieved_text = self.retrieved_text[str(uq_idx)]

        tags = retrieved_text['tags'][:self.num_tags]
        attributes = retrieved_text['attributes'][:self.num_attributes]

        base_text = ''

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
        return img, label, str(uq_idx), descriptive_text_token, uq_idx

class CustomCIFAR100_Base(CIFAR100):
    def __init__(self, root=None, train=True, transform=None, target_transform=None, download=False):

        super(CustomCIFAR100_Base, self).__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.uq_idxs = np.array(range(len(self)))

    def __len__(self):
        return len(self.targets)
    
class CustomCIFAR100_RTG(CustomCIFAR100_Base):
    def __init__(self, root=None, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return {"image": img, "target": label, "image_id": str(uq_idx), "id": uq_idx}
    
class CustomCIFAR100(CustomCIFAR100_Base):
    def __init__(self, retrieved_text_path, root=None, train=True, transform=None, target_transform=None, text_transform=None, download=False, num_tags=0, num_attributes=0):
        self.retrieved_text = process_file(retrieved_text_path)
        self.text_transform = text_transform      
        self.num_tags = num_tags
        self.num_attributes = num_attributes  
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

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

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        retrieved_text = self.retrieved_text[str(uq_idx)]

        tags = retrieved_text['tags'][:self.num_tags]
        attributes = retrieved_text['attributes'][:self.num_attributes]

        base_text = ''

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
        return img, label, str(uq_idx), descriptive_text_token, uq_idx
    
def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cifar_10_datasets(train_transform, 
                          test_transform,
                          text_transform, 
                          train_classes=(0, 1, 8, 9),
                          prop_train_labels=0.8, 
                          split_train_val=False, 
                          seed=0,
                          args=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCIFAR10(root=args.cifar10_root, retrieved_text_path=args.cifar10_retrieved_text_path, transform=train_transform, text_transform=text_transform, train=True, num_tags=args.num_tags, num_attributes=args.num_attributes)

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
    test_dataset = CustomCIFAR10(root=args.cifar10_root, retrieved_text_path=args.cifar10_retrieved_text_path, transform=test_transform, text_transform=None, train=False, num_tags=args.num_tags, num_attributes=args.num_attributes)

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


def get_cifar_100_datasets(train_transform, 
                           test_transform, 
                           text_transform,
                           train_classes=range(80),
                           prop_train_labels=0.8, 
                           split_train_val=False, 
                           seed=0,
                           args=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCIFAR100(root=args.cifar100_root, retrieved_text_path=args.cifar100_retrieved_text_path, transform=train_transform, text_transform=text_transform, train=True, num_tags=args.num_tags, num_attributes=args.num_attributes)

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
    test_dataset = CustomCIFAR100(root=args.cifar100_root, retrieved_text_path=args.cifar100_retrieved_text_path, transform=test_transform, text_transform=None, train=False, num_tags=args.num_tags, num_attributes=args.num_attributes)

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

    x = get_cifar_100_datasets(None, None, split_train_val=False,
                         train_classes=range(80), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')