#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset utilities for image captioning.
This module implements PyTorch dataset classes for loading and preprocessing images and captions.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.vocabulary import Vocabulary

class FlickrDataset(Dataset):
    """
    PyTorch dataset class for the Flickr8k dataset.
    Loads images and their corresponding captions.
    """
    
    def __init__(self, images_dir, captions_file, vocab, transform=None, max_length=50):
        """
        Initialize the dataset.
        
        Args:
            images_dir (str): Directory containing the images
            captions_file (str): Path to the captions CSV file
            vocab (Vocabulary): Vocabulary object for text processing
            transform (torchvision.transforms, optional): Image transformations
            max_length (int): Maximum caption length
        """
        self.images_dir = images_dir
        self.df = pd.read_csv(captions_file)
        self.vocab = vocab
        self.max_length = max_length
        
        # Define default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),  # Resize to 256x256
                transforms.CenterCrop(224),  # Center crop to 224x224
                transforms.ToTensor(),  # Convert to tensor (0-1)
                transforms.Normalize(  # Normalize with ImageNet mean and std
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Determine the word-to-index method in the vocab object
        if hasattr(self.vocab, 'word_to_idx'):
            self.word_dict = self.vocab.word_to_idx
        elif hasattr(self.vocab, 'word2idx'):
            self.word_dict = self.vocab.word2idx
        elif hasattr(self.vocab, 'stoi'):
            self.word_dict = self.vocab.stoi
        elif hasattr(self.vocab, 'token_to_id'):
            self.word_dict = self.vocab.token_to_id
        elif hasattr(self.vocab, 'get_idx'):
            self.word_to_idx_func = self.vocab.get_idx
            return
        elif hasattr(self.vocab, 'get_index'):
            self.word_to_idx_func = self.vocab.get_index
            return
        elif hasattr(self.vocab, '__call__'):
            # If the vocab object is callable, use it directly
            self.word_to_idx_func = self.vocab
            return
        else:
            # Try to find any method that might be intended for word-to-index conversion
            word_to_idx_candidates = [attr for attr in dir(self.vocab) if 'idx' in attr.lower() or 'index' in attr.lower() or 'id' in attr.lower()]
            if word_to_idx_candidates:
                self.word_to_idx_func = getattr(self.vocab, word_to_idx_candidates[0])
                return
            else:
                raise AttributeError(f"Could not find a word-to-index method in the Vocabulary class. "
                                     f"Please verify the Vocabulary implementation or provide a method name.")
        
        # If we're using a dictionary, create a method for word-to-index conversion
        self.word_to_idx_func = self._get_word_idx

    def _get_word_idx(self, word):
        """Helper method to get word index from dictionary."""
        return self.word_dict.get(word, self.word_dict[self.vocab.unk_token])

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, caption)
                image (torch.Tensor): Preprocessed image tensor
                caption (torch.Tensor): Caption token indices
        """
        # 1. Get caption text and image filename from DataFrame at the given index
        caption_text = str(self.df.iloc[idx]['caption'])
        image_name = self.df.iloc[idx]['image']
        
        # 2. Load the image from disk
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        # 3. Apply transformations to the image
        image = self.transform(image)
        
        # 4. Process the caption text: convert to token indices using vocabulary
        # First tokenize the text
        tokens = self.vocab.tokenize(caption_text)
        
        # Convert tokens to indices
        caption_indices = [self.word_to_idx_func(self.vocab.start_token)]
        caption_indices.extend([self.word_to_idx_func(token) for token in tokens])
        caption_indices.append(self.word_to_idx_func(self.vocab.end_token))
        
        # 5. Pad or truncate caption to max_length
        if len(caption_indices) > self.max_length:
            caption_indices = caption_indices[:self.max_length]
            caption_indices[-1] = self.word_to_idx_func(self.vocab.end_token)  # Ensure last token is end token
        else:
            caption_indices += [self.word_to_idx_func(self.vocab.pad_token)] * (self.max_length - len(caption_indices))
        
        # 6. Convert caption to a tensor
        caption = torch.tensor(caption_indices, dtype=torch.long)
        return image, caption


class FlickrDatasetWithID(FlickrDataset):
    """
    Extended Flickr dataset that also returns image IDs.
    Useful for evaluation and visualization.
    """
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset with image ID.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, caption, image_id)
        """
        # Get base items
        image, caption = super().__getitem__(idx)
        
        # Get image ID
        img_name = self.df.iloc[idx]['image']
        
        return image, caption, img_name


def get_data_loaders(data_dir, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker threads for loading data
        pin_memory (bool): Whether to pin memory (useful for GPU training)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab)
    """
    # Define paths
    images_dir = os.path.join(data_dir, "processed", "images")
    train_captions = os.path.join(data_dir, "processed", "train_captions.csv")
    val_captions = os.path.join(data_dir, "processed", "val_captions.csv")
    test_captions = os.path.join(data_dir, "processed", "test_captions.csv")
    vocab_path = os.path.join(data_dir, "processed", "vocabulary.pkl")
    
    # Load vocabulary
    vocab = Vocabulary.load(vocab_path)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    train_dataset = FlickrDatasetWithID(images_dir, train_captions, vocab, transform=train_transform)
    val_dataset = FlickrDatasetWithID(images_dir, val_captions, vocab, transform=val_transform)
    test_dataset = FlickrDatasetWithID(images_dir, test_captions, vocab, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Evaluate one image at a time
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, vocab