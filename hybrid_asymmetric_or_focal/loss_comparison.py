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

TARGET_CLASSES = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion",
    "Emphysema","Fibrosis","Hernia","Infiltration","Mass",
    "No Finding","Nodule","Pleural_Thickening","Pneumonia","Pneumothorax"
]

IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 3
SUBSET_SIZE = 5000  # Limited subset for quick comparison

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"🚀 Running on {device} | Subset Size: {SUBSET_SIZE}")

# ---------------- LOSS FUNCTIONS ---------------- #

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = True
        self.eps = eps

    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Binary Cross Entropy
        loss_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()

# ---------------- MODEL ---------------- #

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=7,padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg = torch.mean(x,dim=1,keepdim=True)
        mx,_ = torch.max(x,dim=1,keepdim=True)
        x2 = torch.cat([avg,mx],dim=1)
        x2 = self.conv(x2)
        return x * self.sigmoid(x2)

class HybridResearchModel(nn.Module):
    def __init__(self,num_classes=len(TARGET_CLASSES)):
        super().__init__()
        cnn = models.densenet121(weights="IMAGENET1K_V1")
        self.cnn_features = cnn.features
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        self.vit = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0, img_size=IMG_SIZE)
        self.spatial_att = SpatialAttention()
        self.classifier = nn.Sequential(
            nn.Linear(1024+768,512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.final_fc = nn.Linear(512,num_classes)

    def forward(self,x):
        c_feat = self.cnn_features(x)
        c_feat = self.spatial_att(c_feat)
        c_feat = self.cnn_pool(c_feat).view(x.size(0),-1)
        v_feat = self.vit(x)
        feat = torch.cat([c_feat,v_feat],dim=1)
        x = self.classifier(feat)
        return self.final_fc(x)

# ---------------- DATASET ---------------- #

class ChestXrayDataset(Dataset):
    def __init__(self,df,root,transform=None):
        self.df=df
        self.root=root
        self.transform=transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        try:
            row = self.df.iloc[idx]
            img_name=row["Image Index"]
            path=os.path.join(self.root,img_name)
            img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            if img is None: return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            clahe=cv2.createCLAHE(2.0,(8,8))
            img=clahe.apply(img)
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img=Image.fromarray(img)
            label=torch.tensor(row["label_vec"],dtype=torch.float32)
            if self.transform: img=self.transform(img)
            return img,label
        except:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))

def prepare_loaders():
    df=pd.read_csv(METADATA_PATH)
    if "Patient ID" not in df.columns:
        raw=pd.read_csv(RAW_METADATA_PATH,usecols=["Image Index","Patient ID"])
        df=df.merge(raw,on="Image Index",how="left")
    
    def encode(x):
        labels=str(x).split("|")
        return [1 if c in labels else 0 for c in TARGET_CLASSES]
    df["label_vec"]=df["Finding Labels"].apply(encode)
    
    available=set(os.listdir(DATA_DIR))
    df=df[df["Image Index"].isin(available)]
    
    # Take a small subset for quick experiment
    df = df.sample(n=SUBSET_SIZE, random_state=42).reset_index(drop=True)
    
    patients=df["Patient ID"].unique()
    np.random.seed(42)
    np.random.shuffle(patients)
    
    train_p=patients[:int(0.8*len(patients))]
    val_p=patients[int(0.8*len(patients)):]
    
    train_df=df[df["Patient ID"].isin(train_p)]
    val_df=df[df["Patient ID"].isin(val_p)]
    
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    train_loader=DataLoader(ChestXrayDataset(train_df,DATA_DIR,transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader=DataLoader(ChestXrayDataset(val_df,DATA_DIR,transform), batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

# ---------------- TRAINING LOGIC ---------------- #

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            out = torch.sigmoid(model(x)).cpu().numpy()
            preds.append(out)
            labels.append(y.numpy())
    return roc_auc_score(np.vstack(labels), np.vstack(preds), average="macro")

def run_training(loss_name, criterion):
    print(f"\n--- Training with {loss_name} Loss ---")
    train_loader, val_loader = prepare_loaders()
    model = HybridResearchModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    best_auc = 0
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"{loss_name} Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{float(loss):.4f}")
        
        val_auc = evaluate(model, val_loader)
        print(f"📊 {loss_name} Valid AUC: {val_auc:.4f}")
        best_auc = max(best_auc, val_auc)
    
    return best_auc

if __name__ == "__main__":
    results = {}
    
    # Run Focal Loss
    results['Focal'] = run_training("Focal", FocalLoss())
    
    # Run Asymmetric Loss
    results['Asymmetric'] = run_training("Asymmetric", AsymmetricLoss())
    
    print("\n" + "="*30)
    print("FINAL EXPERIMENT RESULTS:")
    for loss, auc in results.items():
        print(f"{loss} Loss: {auc:.4f} AUC")
    print("="*30)
