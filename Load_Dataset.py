# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
from transformers import BertTokenizer, BertModel


class BertEmbedding:
    """
    BERT Embedding wrapper using HuggingFace transformers.
    Replaces bert_embedding library with same interface.
    """
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialized = False
    
    def _lazy_init(self):
        """Lazy initialization to avoid CUDA issues with multiprocessing."""
        if not self._initialized:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.model.eval()
            # Keep model on CPU to avoid multiprocessing issues
            self._initialized = True
    
    def __call__(self, sentences):
        """
        Get BERT embeddings for sentences.
        Returns list of tuples (tokens, embeddings) to match bert_embedding format.
        """
        self._lazy_init()
        
        results = []
        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(
                    sentence, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                # Keep everything on CPU to avoid multiprocessing CUDA errors
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.squeeze(0).numpy()
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
                results.append((tokens, embeddings))
        return results


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]  # Co
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = correct_dims(mask)
        text = self.rowtext[mask_filename]
        text = text.split('\n')
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        
        emb_dim = text.shape[1]
        if text.shape[0] > 50:
            text = text[:50, :]
        elif text.shape[0] < 50:
            pad = np.zeros((50 - text.shape[0], emb_dim), dtype=text.dtype)
            text = np.concatenate([text, pad], axis=0)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'label': mask, 'text': text}

        return sample, mask_filename


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]  # MoNuSeg
        mask_filename = os.path.splitext(image_filename)[0] + ".png"

        # mask_filename = self.mask_list[idx]  # Covid19
        # image_filename = mask_filename.replace('mask_', '')  # Covid19
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        # Try mask_filename first, then image_filename as fallback
        text = self.rowtext.get(mask_filename, self.rowtext.get(image_filename, None))
        if text is None:
            raise KeyError(f"Text not found for {mask_filename} or {image_filename}. Available keys: {list(self.rowtext.keys())[:5]}...")
        text = text.split('\n')
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        emb_dim = text.shape[1]
        if text.shape[0] > 50:
            text = text[:50, :]
        elif text.shape[0] < 50:
            pad = np.zeros((50 - text.shape[0], emb_dim), dtype=text.dtype)
            text = np.concatenate([text, pad], axis=0)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'image': image, 'label': mask, 'text': text}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename


##########################################################################
# Semi-Supervised Learning Dataset Classes
##########################################################################

class SemiSupervisedDataset(Dataset):
    """
    Dataset for semi-supervised learning.
    Handles both labeled and unlabeled data with split functionality.
    """
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False, image_size: int = 224, 
                 labeled_ratio: float = 0.25, is_labeled: bool = True, indices: list = None) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.all_images = os.listdir(self.input_path)
        self.all_images.sort()
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.bert_embedding = BertEmbedding()
        self.is_labeled = is_labeled
        
        # Split dataset if indices not provided
        if indices is not None:
            self.images_list = [self.all_images[i] for i in indices]
        else:
            n_total = len(self.all_images)
            n_labeled = int(n_total * labeled_ratio)
            
            # Shuffle indices for random split
            all_indices = list(range(n_total))
            random.shuffle(all_indices)
            
            if is_labeled:
                self.images_list = [self.all_images[i] for i in all_indices[:n_labeled]]
            else:
                self.images_list = [self.all_images[i] for i in all_indices[n_labeled:]]

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        mask_filename = os.path.splitext(image_filename)[0] + ".png"

        
        # Read image
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Read mask (for labeled data) or create dummy mask (for unlabeled)
        if self.is_labeled:
            mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask[mask <= 0] = 0
            mask[mask > 0] = 1
        else:
            # For unlabeled data, create dummy mask (will be replaced by pseudo-labels)
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        image, mask = correct_dims(image, mask)
        
        # Get text embedding
        text = self.rowtext.get(mask_filename, self.rowtext.get(image_filename, "nuclei segmentation task"))
        if isinstance(text, str):
            text = text.split('\n')
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        emb_dim = text.shape[1]
        if text.shape[0] > 50:
            text = text[:50, :]
        elif text.shape[0] < 50:
            pad = np.zeros((50 - text.shape[0], emb_dim), dtype=text.dtype)
            text = np.concatenate([text, pad], axis=0)

        if self.one_hot_mask and self.is_labeled:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'image': image, 'label': mask, 'text': text, 'is_labeled': self.is_labeled}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename


class RandomGeneratorSemi(object):
    """RandomGenerator for semi-supervised learning that handles is_labeled flag."""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        is_labeled = sample.get('is_labeled', True)
        
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        
        sample = {'image': image, 'label': label, 'text': text, 'is_labeled': is_labeled}
        return sample


def create_semi_supervised_datasets(dataset_path, task_name, row_text, train_tf, 
                                     labeled_ratio=0.25, image_size=224, seed=666):
    """
    Create labeled and unlabeled datasets for semi-supervised learning.
    
    Args:
        dataset_path: Path to dataset
        task_name: Name of task (MoNuSeg, Covid19, etc.)
        row_text: Dictionary of text descriptions
        train_tf: Transform to apply
        labeled_ratio: Ratio of labeled data (default 0.25 = 25%)
        image_size: Image size
        seed: Random seed for reproducibility
    
    Returns:
        labeled_dataset: Dataset with labeled data (25%)
        unlabeled_dataset: Dataset with unlabeled data (75%)
    """
    # Get all images
    input_path = os.path.join(dataset_path, 'img')
    all_images = os.listdir(input_path)

    # BẮT BUỘC PHẢI CÓ DÒNG NÀY ĐỂ ĐỒNG BỘ SEED TRÊN MỌI MÁY
    all_images.sort()

    n_total = len(all_images)
    n_labeled = int(n_total * labeled_ratio)
    
    # Set seed for reproducibility
    random.seed(seed)
    all_indices = list(range(n_total))
    random.shuffle(all_indices)
    
    labeled_indices = all_indices[:n_labeled]
    unlabeled_indices = all_indices[n_labeled:]
    
    # Create datasets
    labeled_dataset = SemiSupervisedDataset(
        dataset_path, task_name, row_text, train_tf,
        image_size=image_size, is_labeled=True, indices=labeled_indices
    )
    
    unlabeled_dataset = SemiSupervisedDataset(
        dataset_path, task_name, row_text, train_tf,
        image_size=image_size, is_labeled=False, indices=unlabeled_indices
    )
    
    return labeled_dataset, unlabeled_dataset

