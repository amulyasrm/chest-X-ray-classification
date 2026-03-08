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
TARGET_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- MODELS ---
def get_chexnet_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(TARGET_CLASSES))
    return model

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
    def __init__(self, img_size):
        super().__init__()
        cnn = models.densenet121(weights=None)
        self.cnn_features = cnn.features
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        self.vit = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0, img_size=img_size)
        self.spatial_att = SpatialAttention()
        self.classifier = nn.Sequential(nn.Linear(1024 + 768, 512), nn.ReLU())
        self.final_fc = nn.Linear(512, len(TARGET_CLASSES))

    def forward(self, x):
        c_feat = self.cnn_features(x)
        c_feat = self.spatial_att(c_feat)
        c_feat = self.cnn_pool(c_feat).view(x.size(0), -1) 
        v_feat = self.vit(x)
        combined = torch.cat([c_feat, v_feat], dim=1)
        out = self.classifier(combined)
        return self.final_fc(out)

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

def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            out = torch.sigmoid(model(imgs.to(device)))
            all_preds.append(out.cpu().numpy())
            all_labels.append(labels.numpy())
    
    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    aucs = []
    for i in range(len(TARGET_CLASSES)):
        try:
            aucs.append(roc_auc_score(labels[:, i], preds[:, i]))
        except:
            aucs.append(0.5)
    return aucs

def main():
    # Load and Split Data
    df = pd.read_csv(METADATA_PATH)
    def encode(l): return [1 if c in str(l).split('|') else 0 for c in TARGET_CLASSES]
    df['label_vec'] = df['Finding Labels'].apply(encode)
    
    # Filter 10,000 images for faster benchmarking across all models
    df = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)
    _, val_df = train_test_split(df, test_size=0.2, random_state=42) # 2,000 images validation
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = {'Pathology': TARGET_CLASSES}

    # 1. CheXNet 224
    print("\n--- Evaluating CheXNet 224 ---")
    model_cx = get_chexnet_model().to(device)
    model_cx.load_state_dict(torch.load('chexnet_baseline/chexnet_best.pth', map_location=device))
    loader_224 = DataLoader(ChestDataset(val_df, DATA_DIR, 224, transform), batch_size=16, shuffle=False)
    results['CheXNet_224'] = evaluate_model(model_cx, loader_224)

    # 2. Novel Hybrid 224
    print("\n--- Evaluating Novel Hybrid 224 ---")
    model_n224 = HybridCheXNet(224).to(device)
    model_n224.load_state_dict(torch.load('novel_pipeline_224/novel_full_best.pth', map_location=device))
    results['Novel_224'] = evaluate_model(model_n224, loader_224)

    # 3. Novel Hybrid 384
    print("\n--- Evaluating Novel Hybrid 384 ---")
    model_n384 = HybridCheXNet(384).to(device)
    model_n384.load_state_dict(torch.load('novel_pipeline_384_highres/novel_full_384_best.pth', map_location=device))
    loader_384 = DataLoader(ChestDataset(val_df, DATA_DIR, 384, transform), batch_size=16, shuffle=False)
    results['Novel_384'] = evaluate_model(model_n384, loader_384)

    # Save to CSV
    final_df = pd.DataFrame(results)
    final_df.to_csv('comparisons_and_results/true_model_comparison_results.csv', index=False)
    print("\nResults saved to comparisons_and_results/true_model_comparison_results.csv")

if __name__ == '__main__':
    main()
