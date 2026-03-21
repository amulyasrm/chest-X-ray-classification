import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F

# ---------------- CONFIG ---------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data/images")
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata_filtered.csv")
RAW_METADATA_PATH = os.path.join(BASE_DIR, "data/Data_Entry_2017_v2020.csv")

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
SSL_ENCODER_PATH = os.path.join(EXPERIMENT_DIR, "ssl_encoder_best.pth")
FINETUNE_MODEL_PATH = os.path.join(EXPERIMENT_DIR, "ssl_finetuned_best.pth")

TARGET_CLASSES = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion",
    "Emphysema","Fibrosis","Hernia","Infiltration","Mass",
    "No Finding","Nodule","Pleural_Thickening","Pneumonia","Pneumothorax"
]

IMG_SIZE = 384
BATCH_SIZE = 8
SSL_EPOCHS = 3      # Limited for fast experimentation
FINETUNE_EPOCHS = 3 
SUBSET_SIZE = 5000  # For quick iteration
NUM_WORKERS = 0 # macOS safety

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# ---------------- PREPROCESSING FUNCTION ---------------- #

def apply_clahe(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None: 
        return None
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(image)


# ---------------- MODELS (SimCLR Style Architecture) ---------------- #

class HybridEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        cnn = models.densenet121(weights="IMAGENET1K_V1")
        self.cnn = cnn.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.transformer = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0, img_size=IMG_SIZE)
        
        self.projection = nn.Sequential(
            nn.Linear(1024 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        c_feat = self.cnn(x)
        c_feat = self.pool(c_feat).view(x.size(0), -1)
        t_feat = self.transformer(x)
        feat = torch.cat([c_feat, t_feat], dim=1)
        z = self.projection(feat)
        return z

class HybridClassifier(nn.Module):
    def __init__(self, encoder, num_classes=len(TARGET_CLASSES)):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        out = self.classifier(features)
        return out

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    similarity = torch.matmul(z1, z2.T)
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss = F.cross_entropy(similarity / temperature, labels)
    return loss

# ---------------- DATASETS ---------------- #

class SSLChestXrayDataset(Dataset):
    """Returns two intensely augmented versions of the SAME image for contrastive learning."""
    def __init__(self, dataframe, root_dir):
        self.dataframe = dataframe
        self.root_dir = root_dir
        
        # Heavy augmentations crucial for SimCLR style learning
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            row = self.dataframe.iloc[idx]
            path = os.path.join(self.root_dir, row['Image Index'])
            image_pil = apply_clahe(path)
            
            if image_pil is None: 
                return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros((3, IMG_SIZE, IMG_SIZE))
            
            img1 = self.transform(image_pil)
            img2 = self.transform(image_pil)
            return img1, img2
        except:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros((3, IMG_SIZE, IMG_SIZE))


class FinetuneDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            row = self.dataframe.iloc[idx]
            path = os.path.join(self.root_dir, row['Image Index'])
            image_pil = apply_clahe(path)
            
            if image_pil is None: 
                return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))
            
            label = torch.tensor(row['label_vec'], dtype=torch.float32)
            if self.transform: 
                image_pil = self.transform(image_pil)
            
            return image_pil, label
        except:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))

def get_dataframes():
    df = pd.read_csv(METADATA_PATH)
    if 'Patient ID' not in df.columns:
        raw = pd.read_csv(RAW_METADATA_PATH, usecols=['Image Index', 'Patient ID'])
        df = df.merge(raw, on='Image Index', how='left')
    
    available = set(os.listdir(DATA_DIR))
    df = df[df['Image Index'].isin(available)].reset_index(drop=True)
    
    def encode(l): return [1 if c in str(l).split('|') else 0 for c in TARGET_CLASSES]
    df['label_vec'] = df['Finding Labels'].apply(encode)
    
    # We sample a subset for fast SSL testing
    df = df.sample(n=SUBSET_SIZE, random_state=42).reset_index(drop=True)
    
    unique_patients = df['Patient ID'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    
    train_size = int(0.8 * len(unique_patients))
    
    train_df = df[df['Patient ID'].isin(unique_patients[:train_size])].reset_index(drop=True)
    val_df = df[df['Patient ID'].isin(unique_patients[train_size:])].reset_index(drop=True)
    
    return train_df, val_df

# ---------------- PHASE 1: SELF-SUPERVISED PRETRAINING ---------------- #

def train_ssl(train_df):
    print("\n" + "="*50)
    print("🚀 PHASE 1: SELF-SUPERVISED PRETRAINING (SimCLR Style)")
    print("="*50)
    
    loader = DataLoader(SSLChestXrayDataset(train_df, DATA_DIR), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    model = HybridEncoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) # AdamW works great for hybrid
    
    for epoch in range(1, SSL_EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"SSL Epoch {epoch}/{SSL_EPOCHS}")
        
        for img1, img2 in pbar:
            img1, img2 = img1.to(device), img2.to(device)
            optimizer.zero_grad()
            
            # Forward pass twice
            z1 = model(img1)
            z2 = model(img2)
            
            loss = contrastive_loss(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(ssl_loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss/len(loader)
        print(f"✅ SSL Epoch {epoch} Avg Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), SSL_ENCODER_PATH)
    print(f"⭐ SSL Pretraining Complete. Encoder saved to: \n{SSL_ENCODER_PATH}")
    return model

# ---------------- PHASE 2: FINETUNING ---------------- #

def train_finetune(train_df, val_df, pretrained_encoder):
    print("\n" + "="*50)
    print("🚀 PHASE 2: FINE-TUNING FOR CLASSIFICATION")
    print("="*50)
    
    # Standard augmentation for finetuning
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(FinetuneDataset(train_df, DATA_DIR, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(FinetuneDataset(val_df, DATA_DIR, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = HybridClassifier(pretrained_encoder).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    best_auc = 0
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch}/{FINETUNE_EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validating"):
                out = torch.sigmoid(model(imgs.to(device)))
                all_preds.append(out.cpu().numpy())
                all_labels.append(labels.numpy())
        
        y_true, y_pred = np.vstack(all_labels), np.vstack(all_preds)
        auc = roc_auc_score(y_true, y_pred, average='macro')
        print(f"📊 Finetune Epoch {epoch} AUC: {auc:.4f} | Avg Loss: {train_loss/len(train_loader):.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), FINETUNE_MODEL_PATH)
    
    print(f"⭐ Best Classification Model Saved. Final AUC: {best_auc:.4f}")

if __name__ == '__main__':
    train_df, val_df = get_dataframes()
    
    # 1. Self-Supervised Learning (SSL)
    encoder = train_ssl(train_df)
    
    # 2. Finetuning
    train_finetune(train_df, val_df, encoder)
