import math
import torch
import torch.nn as nn
from clip import clip
from torch.optim.lr_scheduler import _LRScheduler


class CustomCLIP(nn.Module):
    def __init__(self, clip_model, class_nums):
        super().__init__()
        self.model = clip_model
        self.outputdim = clip_model.visual.output_dim

        self.image_classifier = nn.utils.weight_norm(nn.Linear(self.outputdim, class_nums, bias=False))
        self.image_classifier.weight_g.data.fill_(1)
        self.image_classifier.weight_g.requires_grad = False

        self.text_classifier = nn.utils.weight_norm(nn.Linear(self.outputdim, class_nums, bias=False))
        self.text_classifier.weight_g.data.fill_(1)
        self.text_classifier.weight_g.requires_grad = False

    def encode_image(self, image):
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, tokens):
        text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, images, text):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)

        logits_image = self.image_classifier(image_features)
        logits_text = self.text_classifier(text_features)

        return logits_image, logits_text, image_features, text_features
    

class CustomCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, classifier_params, T_max, eta_min=0, last_epoch=-1):
        self.classifier_params_ids = set(map(id, classifier_params))
        self.T_max = T_max
        self.eta_min = eta_min
        self.classifier_lr = optimizer.param_groups[0]['lr']
        self.base_lr = optimizer.param_groups[1]['lr']
        super(CustomCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (self.classifier_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            if any(id(p) in self.classifier_params_ids for p in param_group['params']) 
            else self.base_lr
            for param_group in self.optimizer.param_groups
        ]
    
    
class ImageViewGenerator(object):
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

    
class TextViewGenerator(object):
    """Generate multiple augmented views of a text based on the number of views specified using the same transformation applied multiple times."""
    def __init__(self, base_transform, n_views=2):
        # Ensure transformations is callable
        if not isinstance(base_transform, list):
            if not callable(base_transform):
                raise ValueError("The text transformation must be callable.")
        else:
            if not all(callable(f) for f in base_transform):
                raise ValueError("All elements in the text transformations list must be callable.")
            self.transformations = base_transform
        
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, text):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(text) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](text) for i in range(self.n_views)]
    

def load_clip_to_cpu(backbone_name):

    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model