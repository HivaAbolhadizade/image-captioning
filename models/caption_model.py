#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full caption model integrating encoder and decoder.
This module combines the CNN encoder and RNN decoder into a complete image captioning model.
"""

import torch
import torch.nn as nn
import numpy as np
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN

class CaptionModel(nn.Module):
    """
    Complete image captioning model with CNN encoder and RNN decoder.
    """
    
    def __init__(self, 
             embed_size=256, 
             hidden_size=512, 
             vocab_size=10000, 
             num_layers=1,
             encoder_model='resnet18',
             decoder_type='lstm',
             dropout=0.5,
             train_encoder=False,
             vocab=None,
             use_glove=False):
        """
        Initialize the caption model.
        
        Args:
            embed_size (int): Dimensionality of the embedding space
            hidden_size (int): Dimensionality of the RNN hidden state
            vocab_size (int): Size of the vocabulary
            num_layers (int): Number of layers in the RNN
            encoder_model (str): Name of the CNN backbone for the encoder
            decoder_type (str): Type of RNN cell ('lstm' or 'gru')
            dropout (float): Dropout probability
            train_encoder (bool): Whether to fine-tune the encoder
            vocab (Vocabulary): Vocabulary object for caption generation
        """
        super(CaptionModel, self).__init__()
        
        # Store vocabulary
        self.vocab = vocab
        
        # Initialize encoder and decoder
        self.encoder = EncoderCNN(
            model_name=encoder_model,
            embed_size=embed_size,
            pretrained=True,
            trainable=train_encoder)

        self.decoder = DecoderRNN(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            rnn_type=decoder_type,
            dropout=dropout,
            embedding_matrix=self.create_embedding_matrix() if use_glove else None
        )
    
    def forward(self, images, captions, hidden=None):
        """
        Forward pass for training with teacher forcing.
        
        Args:
            images (torch.Tensor): Input images [batch_size, 3, height, width]
            captions (torch.Tensor): Ground truth captions [batch_size, seq_length]
            hidden (tuple or torch.Tensor, optional): Initial hidden state for the RNN
            
        Returns:
            torch.Tensor: Output scores for each word in the vocabulary
                        Shape: [batch_size, seq_length, vocab_size]
            tuple or torch.Tensor: Final hidden state of the RNN
        """
        features = self.encoder(images)
        outputs, hidden = self.decoder(features, captions, hidden)
        return outputs, hidden
    
    def create_embedding_matrix(self):
        embedding_matrix = torch.zeros(len(self.vocab), 100)
        embeddings = {}
        with open("..\models\glove.6B.100d.txt", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[word] = vector

        for word, idx in self.vocab.word2idx.items():
            if word in embeddings:
                embedding_matrix[idx] = torch.from_numpy(embeddings[word])
            else:
            # Random initialization for unknown words (optional)
                embedding_matrix[idx] = torch.randn(100)

        return embedding_matrix

    def generate_caption(self, image, max_length=20, start_token=1, end_token=2, beam_size=1):
        """
        Generate a caption for an image using beam search.
        
        Args:
            image (torch.Tensor): Input image tensor
            max_length (int): Maximum caption length
            start_token (str): Start token
            end_token (str): End token
            beam_size (int): Beam size for beam search
            
        Returns:
            torch.Tensor: Generated caption indices
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not set. Please set the vocabulary before generating captions.")
            
        self.eval()
        
        # Get image features
        with torch.no_grad():
            image_features = self.encoder(image)
            sampled_ids = self.decoder.sample(
                features=image_features,
                max_length=max_length,
                start_token=start_token,
                end_token=end_token,
                beam_size=beam_size
            )
        
        # Initialize beam
        # start_idx = self.vocab.word2idx[start_token]
        # sequences = [([start_idx], 0.0, None, None)]  # (sequence, score, hidden, next_input)
        
        # # Beam search
        # for _ in range(max_length):
        #     candidates = []
            
        #     # Expand each beam
        #     for seq, score, hidden, next_input in sequences:
        #         if seq[-1] == self.vocab.word2idx[end_token]:
        #             candidates.append((seq, score, hidden, next_input))
        #             continue
                
        #         # Prepare input
        #         if next_input is None:
        #             next_input = torch.tensor([[seq[-1]]]).to(image.device)
                
        #         # Get model predictions
        #         with torch.no_grad():
        #             outputs, hidden = self.decoder(image_features, next_input, hidden)
        #             probs = outputs[0, -1, :]  # Get probabilities for next word
                
        #         # Get top beam_size candidates
        #         top_probs, top_indices = torch.topk(probs, beam_size)
                
        #         # Add candidates
        #         for prob, idx in zip(top_probs, top_indices):
        #             new_seq = seq + [idx.item()]
        #             new_score = score - torch.log(prob).item()  # Negative log likelihood
        #             candidates.append((new_seq, new_score, hidden, torch.tensor([[idx]]).to(image.device)))
            
        #     # Select top beam_size candidates
        #     sequences = sorted(candidates, key=lambda x: x[1])[:beam_size]
            
        #     # Check if all beams end with end token
        #     if all(seq[0][-1] == self.vocab.word2idx[end_token] for seq in sequences):
        #         break
        
        # Get best sequence: list
        # best_seq = sequences[0][0]
        # return torch.tensor([best_seq])
        # print(sampled_ids)
        return sampled_ids.cpu()
