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
MODEL_SAVE_PATH = os.path.join(EXPERIMENT_DIR, "hybrid_asymmetric_best.pth")
RESULTS_CSV_PATH = os.path.join(EXPERIMENT_DIR, "asymmetric_results.csv")

TARGET_CLASSES = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion",
    "Emphysema","Fibrosis","Hernia","Infiltration","Mass",
    "No Finding","Nodule","Pleural_Thickening","Pneumonia","Pneumothorax"
]

IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 10
NUM_WORKERS = 0 # macOS safety

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"🚀 Device: {device} | Hybrid Asymmetric Pipeline: ON | Patient Split: ON")

# ---------------- LOSS FUNCTION ---------------- #

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        loss_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()

# ---------------- MODEL ---------------- #

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

# ---------------- DATASET ---------------- #

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            row = self.dataframe.iloc[idx]
            img_path = os.path.join(self.root_dir, row['Image Index'])
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))
            
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
            
            label = torch.tensor(row['label_vec'], dtype=torch.float32)
            if self.transform: image = self.transform(image)
            return image, label
        except:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))

def get_loaders():
    df = pd.read_csv(METADATA_PATH)
    if 'Patient ID' not in df.columns:
        raw = pd.read_csv(RAW_METADATA_PATH, usecols=['Image Index', 'Patient ID'])
        df = df.merge(raw, on='Image Index', how='left')
    
    available = set(os.listdir(DATA_DIR))
    df = df[df['Image Index'].isin(available)].reset_index(drop=True)
    
    def encode(l): return [1 if c in str(l).split('|') else 0 for c in TARGET_CLASSES]
    df['label_vec'] = df['Finding Labels'].apply(encode)
    
    unique_patients = df['Patient ID'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    
    train_size = int(0.8 * len(unique_patients))
    val_size = int(0.1 * len(unique_patients))
    
    train_df = df[df['Patient ID'].isin(unique_patients[:train_size])].reset_index(drop=True)
    val_df = df[df['Patient ID'].isin(unique_patients[train_size:train_size+val_size])].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_l = DataLoader(ChestXrayDataset(train_df, DATA_DIR, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_l = DataLoader(ChestXrayDataset(val_df, DATA_DIR, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_l, val_l

# ---------------- TRAINING ---------------- #

def train(resume=True):
    train_loader, val_loader = get_loaders()
    model = HybridOptimizedModel().to(device)
    criterion = AsymmetricLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch = 1
    best_auc = 0

    if resume and os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 Resuming from existing checkpoint: {MODEL_SAVE_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            # Rough estimation: If we already have a best model, we'll try to calculate its AUC first
            # to avoid overwriting a better model with a worse one during the first new epoch.
            print("🔍 Validating loaded model to restore best_auc score...")
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="Restoring Metric"):
                    out = torch.sigmoid(model(imgs.to(device)))
                    all_preds.append(out.cpu().numpy())
                    all_labels.append(labels.numpy())
            best_auc = roc_auc_score(np.vstack(all_labels), np.vstack(all_preds), average='macro')
            print(f"✅ Restored Best AUC: {best_auc:.4f}")
            
            # Simple heuristic for start_epoch (or user can manually set it)
            # Since epoch 7 just finished previously, we want to start at 8.
            start_epoch = 8 
        except Exception as e:
            print(f"⚠️ Could not load checkpoint fully: {e}. Starting fresh.")

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n--- Starting Epoch {epoch}/{EPOCHS} ---")
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Training epoch {epoch}")
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
        print(f"📊 Epoch {epoch} Macro AUC: {auc:.4f} | Avg Loss: {train_loss/len(train_loader):.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # Save detailed metrics to CSV
            res = []
            for i, cls in enumerate(TARGET_CLASSES):
                res.append({'Pathology': cls, 'AUC': roc_auc_score(y_true[:, i], y_pred[:, i])})
            pd.DataFrame(res).to_csv(RESULTS_CSV_PATH, index=False)
            print(f"⭐ New Best Model Saved to {MODEL_SAVE_PATH}")
        
        scheduler.step()

        # INTERACTIVE STEP:
        if epoch < EPOCHS:
            user_input = input(f"\n✅ Epoch {epoch} complete. Proceed to next epoch? (TYPE 'yes' to continue): ").lower()
            if user_input != 'yes':
                print("🛑 Training paused. You can resume later by running the script again.")
                break

if __name__ == "__main__":
    train()
