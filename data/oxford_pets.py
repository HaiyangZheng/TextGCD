import os
import os.path
import pathlib
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import numpy as np
from copy import deepcopy

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

from data.data_utils import subsample_instances
from utils import process_file
from clip import clip

tag_probability_words = ["most likely", "probably", "perhaps"]
attribute_probability_words = ["Likely", "Perhaps", "Could be"]

class OxfordIIITPet(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "segmentation")

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
        ]

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)

class OxfordPet_Base(OxfordIIITPet):

    def __init__(self, root=None, split='trainval', transform=None, target_transform=None, download=False):

        super(OxfordPet_Base, self).__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.data = np.array(self._images)
        self.targets = np.array(self._labels)
        self.uq_idxs = np.array(range(len(self)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self.data[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self.targets[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

class OxfordPet_RTG(OxfordPet_Base):
    def __init__(self, root=None, split='trainval', transform=None, target_transform=None, download=False):
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download)
        
    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return {"image": img, "target": label, "image_id": str(uq_idx), "id": uq_idx}

class OxfordPetDataset(OxfordPet_Base):
    def __init__(self, retrieved_text_path, root=None, split='trainval', transform=None, target_transform=None, text_transform=None, download=False, num_tags=0, num_attributes=0):
        self.retrieved_text = process_file(retrieved_text_path)
        self.text_transform = text_transform   
        self.num_tags = num_tags
        self.num_attributes = num_attributes     
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download)

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

        base_text = 'A photo of a pet'

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


def get_oxford_pets_datasets(train_transform, 
                          test_transform,
                          text_transform, 
                          train_classes=(0, 1, 8, 9),
                          prop_train_labels=0.8, 
                          split_train_val=False, 
                          seed=0,
                          args=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = OxfordPetDataset(root=args.pets_root, retrieved_text_path=args.pets_retrieved_text_path, transform=train_transform, text_transform=text_transform, num_tags=args.num_tags, num_attributes=args.num_attributes)

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
    test_dataset = OxfordPetDataset(root=args.pets_root, retrieved_text_path=args.pets_retrieved_text_path, transform=test_transform, text_transform=None, split='test', num_tags=args.num_tags, num_attributes=args.num_attributes)

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

    x = get_oxford_pets_datasets(None, None, split_train_val=False,
                         train_classes=range(10), prop_train_labels=0.5)

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