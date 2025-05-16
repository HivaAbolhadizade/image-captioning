#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vocabulary processing for image captioning.
This module handles building and managing the vocabulary for caption text.
"""

import re
import os
import pickle
import string
import pandas as pd
from collections import Counter
from tqdm import tqdm
import torch
import numpy as np  # (if you use np.integer or np.ndarray anywhere)

class Vocabulary:
    """
    Vocabulary class for processing and tokenizing caption text.
    Handles word-to-index and index-to-word mappings.
    
    Uses a simple tokenizer to avoid NLTK dependencies.
    """
    
    def __init__(self, freq_threshold=5, max_size=None):
        """
        Initialize the vocabulary.
        
        Args:
            freq_threshold (int): Minimum frequency for a word to be included in the vocabulary
            max_size (int, optional): Maximum vocabulary size (excluding special tokens)
        """
        # Special token indices
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        
        # Initialize mappings
        self.word2idx = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3
        }
        self.idx2word = {
            0: self.pad_token,
            1: self.start_token,
            2: self.end_token,
            3: self.unk_token
        }
        
        # Set initial counter index
        self.idx = 4
        
        # Set frequency threshold and maximum size
        self.freq_threshold = freq_threshold
        self.max_size = max_size
    
    def __len__(self):
        """Return the size of the vocabulary"""
        return len(self.word2idx)
    
    def tokenize(self, text):
        """
        Tokenize a caption text into a list of tokens.
        Uses a simple space-based tokenizer instead of NLTK to avoid dependencies.
        
        Args:
            text (str): Caption text to tokenize
            
        Returns:
            list: List of tokens
        """
        if pd.isna(text):
            return []
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove punctuation
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into tokens
        tokens = text.split()
        
        # Filter out tokens with length <= 1
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def build_vocabulary(self, caption_series):
        """
        Build the vocabulary from a series of captions.
        
        Args:
            caption_series (pandas.Series): Series of caption texts
            
        Returns:
            self: The vocabulary object
        """
        # Initialize a Counter to count word frequencies across all captions
        word_freq = Counter()

        # Process captions in batches to improve performance
        batch_size = 1000
        num_captions = len(caption_series)
        
        # Process in batches with a progress bar
        for i in tqdm(range(0, num_captions, batch_size), desc="Building vocabulary"):
            batch = caption_series.iloc[i:min(i+batch_size, num_captions)]
            
            # Tokenize each caption and update the counter with tokens
            for caption in batch:
                tokens = self.tokenize(caption)
                word_freq.update(tokens)

        # Sort words by frequency (most frequent first)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # Filter words based on frequency threshold
        filtered_words = [
            word for word, freq in sorted_words
            if freq >= self.freq_threshold
        ]
        
        # Apply max_size if specified
        if self.max_size:
            filtered_words = filtered_words[:self.max_size]

        # Add each filtered word to the vocabulary
        for word in filtered_words:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

        return self
    
    def numericalize(self, text, add_special_tokens=True):
        """
        Convert a caption text into a list of indices.
        
        Args:
            text (str): Caption text to convert
            add_special_tokens (bool): Whether to add start and end tokens
            
        Returns:
            list: List of token indices
        """
        indices = []
        
        # Tokenize the input text
        tokens = self.tokenize(text)

        # Convert each token to its corresponding index (use UNK token for unknown words)
        for token in tokens:
            index = self.word2idx.get(token, self.word2idx[self.unk_token])
            indices.append(index)

        # Add start and end tokens if requested
        if add_special_tokens:
            indices = [self.word2idx[self.start_token]] + indices + [self.word2idx[self.end_token]]

        return indices
    
    def decode(self, indices, join=False, remove_special=False):
        # Convert to list if tensor
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        # If it's a numpy array
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        # If it's a single int, wrap in list
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        # Now indices is a list
        tokens = [self.idx2word.get(idx, self.unk_token) for idx in indices]
        if remove_special:
            tokens = [t for t in tokens if t not in {self.pad_token, self.start_token, self.end_token}]
        if join:
            return ' '.join(tokens)
        return tokens
    
    def save(self, path):
        """
        Save the vocabulary to a file.
        
        Args:
            path (str): Path to save the vocabulary
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """
        Load a vocabulary from a file.
        
        Args:
            path (str): Path to the vocabulary file
            
        Returns:
            Vocabulary: Loaded vocabulary object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


def build_vocab_from_captions(captions_path, output_dir, freq_threshold=5, max_size=None):
    """
    Build and save a vocabulary from a captions file.
    
    Args:
        captions_path (str): Path to the captions CSV file
        output_dir (str): Directory to save the vocabulary
        freq_threshold (int): Minimum frequency for a word to be included
        max_size (int, optional): Maximum vocabulary size
        
    Returns:
        Vocabulary: The built vocabulary
    """
    # Load captions
    print(f"Loading captions from {captions_path}")
    captions_df = pd.read_csv(captions_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize vocabulary
    vocab = Vocabulary(freq_threshold=freq_threshold, max_size=max_size)
    
    # Build vocabulary
    print(f"Building vocabulary from {len(captions_df)} captions")
    vocab.build_vocabulary(captions_df['caption'])
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocabulary.pkl')
    vocab.save(vocab_path)
    
    print(f"Vocabulary built with {len(vocab)} words")
    print(f"Saved to {vocab_path}")
    
    return vocab


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Build vocabulary from captions')
    parser.add_argument('--captions_path', type=str, required=True, help='Path to captions CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for vocabulary')
    parser.add_argument('--freq_threshold', type=int, default=5, help='Minimum word frequency')
    parser.add_argument('--max_size', type=int, default=None, help='Maximum vocabulary size')
    
    args = parser.parse_args()
    
    vocab = build_vocab_from_captions(
        args.captions_path, 
        args.output_dir,
        freq_threshold=args.freq_threshold,
        max_size=args.max_size
    )
    
    print(f"Built vocabulary with {len(vocab)} words")