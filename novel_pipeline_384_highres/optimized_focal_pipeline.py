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
from sklearn.metrics import roc_auc_score, f1_score
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'images')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'metadata_filtered.csv')
RAW_METADATA_PATH = os.path.join(BASE_DIR, 'data', 'Data_Entry_2017_v2020.csv')

TARGET_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 10
NUM_WORKERS = 0 # Set to 0 to avoid multiprocessing issues on macOS spawn

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"🚀 Device: {device} | Focal Pipeline: ON | Patient Split: ON")

# --- LOSS FUNCTION ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs should be logits
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        if self.reduction == 'mean': return torch.mean(F_loss)
        elif self.reduction == 'sum': return torch.sum(F_loss)
        else: return F_loss

# --- MODEL ARCHITECTURE ---
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv(res)
        return x * self.sigmoid(res)

class HybridOptimizedModel(nn.Module):
    def __init__(self, num_classes=len(TARGET_CLASSES)):
        super().__init__()
        cnn = models.densenet121(weights='IMAGENET1K_V1')
        self.cnn_features = cnn.features
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        self.vit = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, img_size=IMG_SIZE)
        self.spatial_att = SpatialAttention()
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.final_fc = nn.Linear(512, num_classes)

    def forward(self, x):
        c_feat = self.cnn_features(x)
        c_feat = self.spatial_att(c_feat)
        c_feat = self.cnn_pool(c_feat).view(x.size(0), -1) 
        v_feat = self.vit(x)
        combined = torch.cat([c_feat, v_feat], dim=1)
        out = self.classifier(combined)
        return self.final_fc(out)

# --- DATASET ---
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            img_name = self.dataframe.iloc[idx]['Image Index']
            img_path = os.path.join(self.root_dir, img_name)
            
            # Using cv2 for reliability, then PIL for transforms
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))
            
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            # CLAHE initialized inside worker process to avoid pickling issue
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
            
            label_vec = self.dataframe.iloc[idx]['label_vec']
            label = torch.tensor(label_vec, dtype=torch.float32)
            
            if self.transform: image = self.transform(image)
            return image, label
        except Exception as e:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))

def get_dataloaders():
    print(f"⌛ Loading metadata from {METADATA_PATH}...")
    df = pd.read_csv(METADATA_PATH)
    
    # Merge with original data for Patient ID split
    if 'Patient ID' not in df.columns:
        print("⌛ Fetching Patient IDs from raw dataset...")
        raw_df = pd.read_csv(RAW_METADATA_PATH, usecols=['Image Index', 'Patient ID'])
        df = df.merge(raw_df, on='Image Index', how='left')

    # Filter to ensure images exist
    print("⌛ Verifying image files on disk...")
    available = set(os.listdir(DATA_DIR))
    df = df[df['Image Index'].isin(available)].reset_index(drop=True)
    
    def encode(l): return [1 if c in str(l).split('|') else 0 for c in TARGET_CLASSES]
    df['label_vec'] = df['Finding Labels'].apply(encode)
    
    unique_patients = df['Patient ID'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    
    train_size = int(0.8 * len(unique_patients))
    val_size = int(0.1 * len(unique_patients))
    
    train_patients = unique_patients[:train_size]
    val_patients = unique_patients[train_size:train_size+val_size]
    
    train_df = df[df['Patient ID'].isin(train_patients)].reset_index(drop=True)
    val_df = df[df['Patient ID'].isin(val_patients)].reset_index(drop=True)
    
    print(f"📊 Dataset Split: Train Patients={len(train_patients)}, Val Patients={len(val_patients)}")
    print(f"📊 Dataset Size: Train={len(train_df)}, Val={len(val_df)}")

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(ChestXrayDataset(train_df, DATA_DIR, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(ChestXrayDataset(val_df, DATA_DIR, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    return train_loader, val_loader

# --- TRAINING ENGINE ---
def train():
    train_loader, val_loader = get_dataloaders()
    model = HybridOptimizedModel().to(device)
    criterion = FocalLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Using AMP where applicable
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validating"):
                out = torch.sigmoid(model(imgs.to(device)))
                all_preds.append(out.cpu().numpy())
                all_labels.append(labels.numpy())
        
        auc = roc_auc_score(np.vstack(all_labels), np.vstack(all_preds), average='macro')
        avg_loss = train_loss/len(train_loader)
        print(f"\n✅ Epoch {epoch} Results: AUC={auc:.4f} | Avg Loss={avg_loss:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            save_path = os.path.join(os.path.dirname(__file__), "optimized_focal_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Best Model Saved to {save_path}")
        
        scheduler.step()
        
        # Non-interactive Mode for Agent Execution
        print("-" * 30)

if __name__ == '__main__':
    train()
