#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flickr8k dataset downloader and preprocessor.
This script downloads the Flickr8k dataset and organizes it for the image captioning task.
"""

import os
import zipfile
import requests
import pandas as pd
from tqdm import tqdm
import shutil
import tarfile
import argparse

def download_file(url, destination):
    """
    Downloads a file from a URL to a destination with progress bar.
    
    Args:
        url (str): URL to download from
        destination (str): Path to save the downloaded file
    """
    if os.path.exists(destination):
        print(f"File already exists at {destination}. Skipping download.")
        return
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(destination, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

def extract_zip(zip_path, extract_path):
    """
    Extracts a zip file to a destination folder.
    
    Args:
        zip_path (str): Path to the zip file
        extract_path (str): Path to extract the contents to
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f"Extracting {os.path.basename(zip_path)}"):
            zip_ref.extract(member, extract_path)

def extract_tar(tar_path, extract_path):
    """
    Extracts a tar file to a destination folder.
    
    Args:
        tar_path (str): Path to the tar file
        extract_path (str): Path to extract the contents to
    """
    with tarfile.open(tar_path, 'r:*') as tar_ref:
        for member in tqdm(tar_ref.getmembers(), desc=f"Extracting {os.path.basename(tar_path)}"):
            tar_ref.extract(member, extract_path)

def process_captions(dataset_path):
    """
    Process captions from Flickr8k dataset and create a clean CSV file.
    
    Args:
        dataset_path (str): Path to the dataset directory
    
    Returns:
        str: Path to the processed captions CSV file
    """
    captions_path = os.path.join(dataset_path, "Flickr8k_text", "Flickr8k.token.txt")
    
    # ✅TODO: Read the captions file and process it into a structured format
    # 1. Read the captions file line by line
    # 2. Parse each line to extract image_name and caption
    # 3. Remove the #id suffix from image_name
    # 4. Create a list of dictionaries with 'image' and 'caption' keys
    data = []
    with open(captions_path, 'r') as file:
        for line in file:
            image_caption = line.strip().split('\t')
            image_id = image_caption[0].split('#')[0]
            caption = image_caption[1]
            data.append({'image': image_id, 'caption': caption})

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    
    # Create clean output dir
    output_dir = os.path.join(dataset_path, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, "captions.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Processed captions saved to {output_path}")
    return output_path

def organize_images(dataset_path):
    """
    Organizes the images into a clean directory structure.
    
    Args:
        dataset_path (str): Path to the dataset directory
    
    Returns:
        str: Path to the organized images directory
    """
    source_images_dir = os.path.join(dataset_path, "Flickr8k_Dataset", "Flicker8k_Dataset")
    
    if not os.path.exists(source_images_dir):
        source_images_dir = os.path.join(dataset_path, "Flicker8k_Dataset")  # Alternative path
    
    # Create clean output dir
    output_dir = os.path.join(dataset_path, "processed", "images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy images to new location
    for img_file in tqdm(os.listdir(source_images_dir), desc="Organizing images"):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            source = os.path.join(source_images_dir, img_file)
            destination = os.path.join(output_dir, img_file)
            
            if not os.path.exists(destination):
                shutil.copy2(source, destination)
    
    print(f"Images organized in {output_dir}")
    return output_dir

def create_splits(dataset_path):
    """
    Creates train/val/test splits based on the official Flickr8k splits.
    
    Args:
        dataset_path (str): Path to the dataset directory
    """
    processed_dir = os.path.join(dataset_path, "processed")
    flickr_text_dir = os.path.join(dataset_path, "Flickr8k_text")
    
    # ✅TODO: Read the split files and create sets of image filenames for each split
    # 1. Read Flickr_8k.trainImages.txt, Flickr_8k.devImages.txt, and Flickr_8k.testImages.txt
    # 2. Create sets containing the image filenames for train, validation, and test splits
    with open(os.path.join(flickr_text_dir, "Flickr_8k.trainImages.txt")) as f:
        train_images = set(line.strip() for line in f.readlines())
    with open(os.path.join(flickr_text_dir, "Flickr_8k.devImages.txt")) as f:
        val_images = set(line.strip() for line in f.readlines())
    with open(os.path.join(flickr_text_dir, "Flickr_8k.testImages.txt")) as f:
        test_images = set(line.strip() for line in f.readlines())
    
    # Load caption data
    captions_df = pd.read_csv(os.path.join(processed_dir, "captions.csv"))
    
    train_df = captions_df[captions_df['image'].isin(train_images)]
    val_df = captions_df[captions_df['image'].isin(val_images)]
    test_df = captions_df[captions_df['image'].isin(test_images)]
    # ✅TODO: Create DataFrames for each split by filtering the captions_df
    # 1. Filter captions_df to create train_df, val_df, and test_df based on image filename
    # 2. Save each DataFrame to a CSV file in the processed directory
    train_df.to_csv(os.path.join(processed_dir, "train_captions.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val_captions.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_captions.csv"), index=False)
    print(f"Created data splits: train ({len(train_df)} captions), val ({len(val_df)} captions), test ({len(test_df)} captions)")

def download_flickr8k(base_dir="./data"):
    """
    Downloads and prepares the Flickr8k dataset.
    
    Args:
        base_dir (str): Base directory to store the dataset
    
    Returns:
        dict: Dictionary with paths to the dataset components
    """
    dataset_path = os.path.join(base_dir, "flickr8k")
    os.makedirs(dataset_path, exist_ok=True)
    
    # URLs for Flickr8k dataset
    # Note: In a real implementation, you would use official download links
    # For this example, we're using placeholders that should be replaced with official sources
    images_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    
    images_zip = os.path.join(dataset_path, "Flickr8k_Dataset.zip")
    text_zip = os.path.join(dataset_path, "Flickr8k_text.zip")
    
    # Download dataset files
    download_file(images_url, images_zip)
    download_file(text_url, text_zip)
    
    # Extract dataset files
    extract_zip(images_zip, dataset_path)
    extract_zip(text_zip, dataset_path)
    
    # Process captions
    captions_path = process_captions(dataset_path)
    
    # Organize images
    images_path = organize_images(dataset_path)
    
    # Create data splits
    create_splits(dataset_path)
    
    return {
        "dataset_path": dataset_path,
        "images_path": images_path,
        "captions_path": captions_path,
        "processed_path": os.path.join(dataset_path, "processed")
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare the Flickr8k dataset")
    parser.add_argument("--data_dir", type=str, default="./data", 
                        help="Base directory to store the dataset")
    args = parser.parse_args()
    
    paths = download_flickr8k(args.data_dir)
    
    print("\nDataset preparation complete!")
    print(f"Dataset stored in: {paths['dataset_path']}")
    print(f"Processed images: {paths['images_path']}")
    print(f"Processed captions: {paths['captions_path']}")