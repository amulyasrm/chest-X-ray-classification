import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from PIL import Image
from tqdm import tqdm

from hybrid_384_asymmetric_experiment.hybrid_asymmetric_pipeline import HybridOptimizedModel

# --- CONFIG ---
DATA_DIR = 'data/images'
METADATA_PATH = 'data/metadata_filtered.csv'
TARGET_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- DATASET ---
class ChestDataset(Dataset):
    def __init__(self, dataframe, root_dir, img_size, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            img_name = self.dataframe.iloc[idx]['Image Index']
            img_path = os.path.join(self.root_dir, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: return torch.zeros((3, self.img_size, self.img_size)), torch.zeros(len(TARGET_CLASSES))
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            image = self.clahe.apply(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
            label = torch.tensor(self.dataframe.iloc[idx]['label_vec'], dtype=torch.float32)
            if self.transform: image = self.transform(image)
            return image, label
        except: return torch.zeros((3, self.img_size, self.img_size)), torch.zeros(len(TARGET_CLASSES))

def main():
    # Load and Split Data (Same specific 2,000 images split used in true_validation.py)
    df = pd.read_csv(METADATA_PATH)
    def encode(l): return [1 if c in str(l).split('|') else 0 for c in TARGET_CLASSES]
    df['label_vec'] = df['Finding Labels'].apply(encode)
    
    # Filter identically to true_validation.py
    df = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n--- Evaluating Hybrid Asymmetric Model (Epoch 7 Checkpoint) ---")
    model = HybridOptimizedModel().to(device)
    model.load_state_dict(torch.load('hybrid_384_asymmetric_experiment/hybrid_asymmetric_best.pth', map_location=device))
    model.eval()

    loader = DataLoader(ChestDataset(val_df, DATA_DIR, 384, transform), batch_size=16, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            out = torch.sigmoid(model(imgs.to(device)))
            all_preds.append(out.cpu().numpy())
            all_labels.append(labels.numpy())
    
    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    
    macro_auc = roc_auc_score(labels, preds, average='macro')
    print(f"\n✅ True Validation Macro AUC: {macro_auc:.6f}")
    
    # Write summary for direct comparison
    print(f"\n--- COMPARISON ---")
    print(f"Hybrid_BCE_384 (Phase 4):     0.8837")
    print(f"Hybrid_Asymmetric_384 (Current): {macro_auc:.4f}")

if __name__ == '__main__':
    main()
