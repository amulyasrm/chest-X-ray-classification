import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# Config
BASE_DIR = '/Users/amulya/Documents/capstone'
DATA_DIR = os.path.join(BASE_DIR, 'data/images')
METADATA_PATH = os.path.join(BASE_DIR, 'data/metadata_filtered.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'project_documentation/assets/augmentation_examples.png')

# Find an image with an obvious pathology
df = pd.read_csv(METADATA_PATH)
# Let's pick Cardiomegaly or something visible
sample_row = df[df['Finding Labels'].str.contains("Cardiomegaly")].iloc[5]
img_name = sample_row['Image Index']
img_path = os.path.join(DATA_DIR, img_name)

# Read and apply CLAHE (the standard preprocessing)
def apply_preprocessing(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None: return None
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(image)

base_img = apply_preprocessing(img_path)

if base_img is not None:
    # Define augmentations
    flip = transforms.RandomHorizontalFlip(p=1.0)
    rotate = transforms.RandomRotation(degrees=(15, 15)) # Fixed 15 degrees for clear visualization
    jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0)
    
    img_flip = flip(base_img)
    img_rot = rotate(base_img)
    
    # We apply specific jitter manually to ensure it's visible (e.g. much brighter)
    img_jit = transforms.functional.adjust_brightness(base_img, 1.4)
    img_jit = transforms.functional.adjust_contrast(img_jit, 0.7)

    # Plotting
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Outfit', 'Inter', 'Arial']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='#111111')
    
    imgs = [base_img, img_flip, img_rot, img_jit]
    titles = [
        "1. Original Preprocessed\n(CLAHE Applied)", 
        "2. Horizontal Flip\n(Doubles structural variations)", 
        "3. Random Rotation (15°)\n(Simulates patient posture)", 
        "4. Color Jitter\n(Simulates exposure differences)"
    ]
    colors = ['#d4af37', '#a855f7', '#3b82f6', '#10b981'] # Amber, Purple, Blue, Greenish
    
    for i, (ax, img, title, color) in enumerate(zip(axes, imgs, titles, colors)):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, color=color, fontsize=14, pad=15, fontweight='bold', ha='center')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.suptitle('Data Augmentation Pipeline Simulation', color='white', fontsize=22, fontweight='bold', y=1.05)
    
    plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
    print(f"Generated successfully: {OUTPUT_PATH}")
else:
    print("Failed to load base image.")
