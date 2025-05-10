#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full caption model integrating encoder and decoder.
This module combines the CNN encoder and RNN decoder into a complete image captioning model.
"""

import torch
import torch.nn as nn
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
             train_encoder=False):
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
        """
        super(CaptionModel, self).__init__()
        
        # ✅TODO: Initialize the encoder and decoder components
        # 1. Create an EncoderCNN instance with the specified parameters
        # 2. Create a DecoderRNN instance with the specified parameters
        self.encoder = EncoderCNN(embed_size, cnn_type=encoder_model, train_cnn=train_encoder)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, decoder_type=decoder_type, dropout=dropout) 
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
        
        # ✅TODO: Implement the forward pass of the full model
        # 1. Extract features from images using the encoder
        # 2. Use the decoder to generate captions based on the features and ground truth captions
        # 3. Return the outputs and final hidden state
        features = self.encoder(images)
        outputs, hidden = self.decoder(features, captions, hidden)
        return outputs, hidden
    
    def generate_caption(self, image, max_length=20, start_token=1, end_token=2, beam_size=1):
        """
        Generate a caption for a single image.
        
        Args:
            image (torch.Tensor): Input image [1, 3, height, width]
            max_length (int): Maximum caption length
            start_token (int): Index of the start token
            end_token (int): Index of the end token
            beam_size (int): Beam size for beam search (1 = greedy search)
            
        Returns:
            torch.Tensor: Generated caption token sequence [1, seq_length]
        """
        sampled_ids = list()
        # ✅TODO: Implement caption generation for inference
        # 1. Extract features from the image using the encoder (with torch.no_grad())
        # 2. Use the decoder to generate a caption based on the features
        # 3. Return the generated caption
        with torch.no_grad():
            features = self.encoder(image)

            if beam_size == 1:
                # Greedy decoding
                inputs = torch.tensor([[start_token]]).to(image.device)
                hidden = None

                for _ in range(max_length):
                    outputs, hidden = self.decoder(features, inputs, hidden)
                    predicted = outputs[:, -1, :].argmax(dim=-1)  # shape: [1]
                    sampled_ids.append(predicted.item())
                    if predicted.item() == end_token:
                        break
                    inputs = predicted.unsqueeze(1)  # shape: [1, 1]
            else:
                # Beam search
                sequences = [[[], 0.0, None, torch.tensor([[start_token]]).to(image.device)]]
                for _ in range(max_length):
                    all_candidates = []
                    for seq, score, hidden, inputs in sequences:
                        outputs, hidden = self.decoder(features, inputs, hidden)
                        probs = torch.nn.functional.log_softmax(outputs[:, -1, :], dim=-1)  # [1, vocab_size]
                        topk_probs, topk_indices = probs.topk(beam_size, dim=-1)  # [1, beam_size]

                        for k in range(beam_size):
                            next_token = topk_indices[0, k].item()
                            candidate = [seq + [next_token], score + topk_probs[0, k].item(), hidden, next_token]
                            all_candidates.append(candidate)

                    ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
                    sequences = []
                    for i in range(beam_size):
                        seq, score, hidden, next_token = ordered[i]
                        if next_token == end_token:
                            sampled_ids = seq
                            return torch.tensor([sampled_ids])
                        next_input = torch.tensor([[next_token]]).to(image.device)
                        sequences.append([seq, score, hidden, next_input])

            sampled_ids = sequences[0][0]  # Best sequence

        return torch.tensor([sampled_ids])  # Return first (and only) sequence in the batch
