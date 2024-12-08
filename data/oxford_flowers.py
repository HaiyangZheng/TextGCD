import numpy as np
from copy import deepcopy

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from PIL import Image

from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from data.data_utils import subsample_instances
from utils import process_file, construct_text
from clip import clip

tag_probability_words = ["most likely", "probably", "perhaps"]
attribute_probability_words = ["Likely", "Perhaps", "Could be"]

class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)

class OxfordFlowers_Base(Flowers102):

    def __init__(self, root=None, split='train', transform=None, target_transform=None, download=False):

        super(OxfordFlowers_Base, self).__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.data = np.array(self._image_files)
        self.targets = np.array(self._labels)
        self.uq_idxs = np.array(range(len(self)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self.data[idx], self.targets[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
class OxfordFlowers_RTG(OxfordFlowers_Base):
    def __init__(self, root=None, split='train', transform=None, target_transform=None, download=False):
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download)
        
    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return {"image": img, "target": label, "image_id": str(uq_idx), "id": uq_idx}

class OxfordFlowers(OxfordFlowers_Base):
    def __init__(self, retrieved_text_path, root=None, split='train', transform=None, target_transform=None, text_transform=None, download=False, num_tags=0, num_attributes=0):
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

        base_text = 'A photo of a flower'

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

def get_oxford_flowers_datasets(train_transform, 
                          test_transform,
                          text_transform, 
                          train_classes=(0, 1, 8, 9),
                          prop_train_labels=0.8, 
                          split_train_val=False, 
                          seed=0,
                          args=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = OxfordFlowers(root=args.flowers_root, retrieved_text_path=args.flowers_retrieved_text_path, transform=train_transform, text_transform=text_transform, num_tags=args.num_tags, num_attributes=args.num_attributes)

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
    test_dataset = OxfordFlowers(root=args.flowers_root, retrieved_text_path=args.flowers_retrieved_text_path, transform=test_transform, text_transform=None, split='test', num_tags=args.num_tags, num_attributes=args.num_attributes)

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

    x = get_oxford_flowers_datasets(None, None, split_train_val=False,
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