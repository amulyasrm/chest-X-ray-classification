import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from PIL import Image
from tqdm import tqdm

# --- CONFIG ---
DATA_DIR = 'data/images'
METADATA_PATH = 'data/metadata_filtered.csv'
WEIGHTS_PATH = 'novel_full_384_best.pth'
TARGET_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
IMG_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 4

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- DATASET ---
class NovelDataset(Dataset):
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
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
            label = torch.tensor(self.dataframe.iloc[idx]['label_vec'], dtype=torch.float32)
            if self.transform: image = self.transform(image)
            return image, label
        except: return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))

# --- MODEL ---
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

class HybridCheXNet(nn.Module):
    def __init__(self, num_classes=len(TARGET_CLASSES)):
        super().__init__()
        cnn = models.densenet121(weights=None)
        self.cnn_features = cnn.features
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        self.vit = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0, img_size=IMG_SIZE)
        self.spatial_att = SpatialAttention()
        self.classifier = nn.Sequential(nn.Linear(1024 + 768, 512), nn.ReLU())
        self.final_fc = nn.Linear(512, num_classes)

    def forward(self, x):
        c_feat = self.cnn_features(x)
        c_feat = self.spatial_att(c_feat)
        c_feat = self.cnn_pool(c_feat).view(x.size(0), -1) 
        v_feat = self.vit(x)
        combined = torch.cat([c_feat, v_feat], dim=1)
        out = self.classifier(combined)
        return self.final_fc(out)

def validate():
    # Load Data
    df = pd.read_csv(METADATA_PATH)
    def encode(l): return [1 if c in str(l).split('|') else 0 for c in TARGET_CLASSES]
    df['label_vec'] = df['Finding Labels'].apply(encode)
    available = set(os.listdir(DATA_DIR))
    df = df[df['Image Index'].isin(available)].reset_index(drop=True)
    _, val_df = train_test_split(df, test_size=0.1, random_state=42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_loader = DataLoader(NovelDataset(val_df, DATA_DIR, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Load Model
    model = HybridCheXNet().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validating Benchmarks"):
            out = torch.sigmoid(model(imgs.to(device)))
            all_preds.append(out.cpu().numpy())
            all_labels.append(labels.numpy())

    # Calculate Results
    preds_stack = np.vstack(all_preds)
    labels_stack = np.vstack(all_labels)
    
    label_aucs = {}
    for i, class_name in enumerate(TARGET_CLASSES):
        try:
            auc = roc_auc_score(labels_stack[:, i], preds_stack[:, i])
            label_aucs[class_name] = auc
        except: label_aucs[class_name] = 0.5

    # Silence printing results as requested. Saving to CSV.
    results_df = pd.DataFrame(list(label_aucs.items()), columns=['Pathology', 'Measured_AUC'])
    results_df.to_csv('measured_performance_metrics.csv', index=False)

if __name__ == '__main__':
    validate()
