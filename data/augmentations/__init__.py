from torchvision import transforms
import torch
import random

def get_transform(transform_type='imagenet', image_size=32, args=None):

    if transform_type == 'imagenet':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711))
        ])

    else:

        raise NotImplementedError

    return (train_transform, test_transform)

def introduce_typo(word):
    if len(word) < 3:
        return word
    
    # Select the index of the character to modify
    char_idx = random.randint(1, len(word) - 2)
    
    # Randomly choose an operation: replace, delete, or add
    action = random.choice(['replace', 'delete', 'add'])
    
    if action == 'replace':
        random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
        word = word[:char_idx] + random_char + word[char_idx + 1:]
    elif action == 'delete':
        word = word[:char_idx] + word[char_idx + 1:]
    else:  # add
        random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
        word = word[:char_idx] + random_char + word[char_idx:]
    
    return word

def tokenize_with_augmentation(text):

    words = text.split()
    
    # Introduce spelling errors to words with a certain probability
    augmented_words = [introduce_typo(word) if random.random() < 0.1 else word for word in words]
    # augmented_words = [introduce_typo(word) if random.random() < 0.1 and word != 'a' else word for word in words]

    return ' '.join(augmented_words)