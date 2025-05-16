#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for image captioning.
This module implements metrics for evaluating caption quality.
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from models.caption_model import CaptionModel
from utils.vocabulary import Vocabulary
from torch.utils.data import DataLoader

# Make sure NLTK tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def calculate_bleu(references, hypotheses, max_n=4):
    """
    Calculate BLEU score for a set of references and hypotheses.
    
    Args:
        references (list): List of reference lists (multiple references per sample)
        hypotheses (list): List of hypothesis lists (one per sample)
        max_n (int): Maximum n-gram to consider
        
    Returns:
        list: BLEU scores for different n-grams (BLEU-1, BLEU-2, etc.)
    """
    # TODO: Implement BLEU score calculation
    # 1. Tokenize references and hypotheses if they're not already tokenized
    if isinstance(references[0][0], str):
        references = [[ref.split() for ref in refs] for refs in references]
    if isinstance(hypotheses[0], str):
        hypotheses = [txt.split() for txt in hypotheses]
    # 2. Set up smoothing function to handle zero counts
    smoother = SmoothingFunction().method1
    # 3. Calculate BLEU scores for different n-grams (BLEU-1 to BLEU-n)
    weight_list = [
        tuple((1. / n if i < n else 0.) for i in range(4))
        for n in range(1, max_n + 1)
    ]
    
    bleu_scores = [
        corpus_bleu(references, hypotheses, weights=weights, smoothing_function=smoother)
        for weights in weight_list
    ]
    # 4. Return list of BLEU scores
    
    return bleu_scores

def calculate_metrics(model, dataloader, vocab, device='cuda', max_samples=None, beam_size=1):
    model.eval()
    references_by_id = defaultdict(list)
    hypotheses_by_id = {}

    # If dataloader is a list, use it directly; else, iterate as normal
    if isinstance(dataloader, list):
        data_iter = dataloader
    else:
        data_iter = dataloader

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_iter, desc="Generating captions", leave=True)):
            # Support both DataLoader and list of tuples
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                image, caption, image_id = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                image, caption = batch
                image_id = [str(i)]  # fallback
            else:
                raise ValueError("Batch format not recognized.")

            if max_samples is not None and i >= max_samples:
                break

            # Move image to device and ensure batch dimension
            image = image.to(device)
            if image.dim() == 3:
                image = image.unsqueeze(0)

            # Generate caption
            predicted_ids = model.generate_caption(image, beam_size=beam_size)
            # Flatten predicted_ids if needed
            if isinstance(predicted_ids, torch.Tensor):
                predicted_ids = predicted_ids.cpu().numpy().tolist()
            if isinstance(predicted_ids, list) and len(predicted_ids) > 0 and isinstance(predicted_ids[0], list):
                predicted_ids = predicted_ids[0]

            predicted_caption = vocab.decode(predicted_ids, join=True, remove_special=True)

            # Reference caption
            ref = caption[0] if isinstance(caption, (list, tuple)) and len(caption) > 0 else caption
            if isinstance(ref, torch.Tensor):
                ref = ref.cpu().numpy().tolist()
            if isinstance(ref, (int, np.integer)):
                ref = [ref]
            reference_caption = vocab.decode(ref, join=True, remove_special=True)

            # Store results
            image_id = image_id[0] if isinstance(image_id, (list, tuple)) else str(image_id)
            references_by_id[image_id].append(reference_caption)
            hypotheses_by_id[image_id] = predicted_caption

    references = [references_by_id[image_id] for image_id in hypotheses_by_id.keys()]
    hypotheses = [hypotheses_by_id[image_id] for image_id in hypotheses_by_id.keys()]
    bleu_scores = calculate_bleu(references, hypotheses)
    return bleu_scores[-1]