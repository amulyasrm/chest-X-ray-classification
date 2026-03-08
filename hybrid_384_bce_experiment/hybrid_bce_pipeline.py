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

# ---------------- CONFIG ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data/images")
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata_filtered.csv")
RAW_METADATA_PATH = os.path.join(BASE_DIR, "data/Data_Entry_2017_v2020.csv")

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_SAVE_PATH = os.path.join(EXPERIMENT_DIR, "hybrid_384_bce_best.pth")
RESULTS_CSV_PATH = os.path.join(EXPERIMENT_DIR, "experiment_results.csv")

TARGET_CLASSES = [
"Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion",
"Emphysema","Fibrosis","Hernia","Infiltration","Mass",
"No Finding","Nodule","Pleural_Thickening","Pneumonia","Pneumothorax"
]

IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 10
NUM_WORKERS = 0

device = torch.device(
"mps" if torch.backends.mps.is_available()
else ("cuda" if torch.cuda.is_available() else "cpu")
)

print("Device:", device)

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

        # Using swin_tiny_patch4_window7_224 with img_size=384 as it has robust pretrained weights
        # Note: swinv2_cr_tiny_384 does not have pretrained weights in this library version.
        self.vit = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0,
            img_size=IMG_SIZE
        )

        self.spatial_att = SpatialAttention()

        self.classifier = nn.Sequential(
            nn.Linear(1024+768,512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.final_fc = nn.Linear(512,num_classes)

    def forward(self,x):

        # CNN branch
        c_feat = self.cnn_features(x)
        c_feat = self.spatial_att(c_feat)

        c_feat = self.cnn_pool(c_feat).view(x.size(0),-1)

        # Transformer branch
        v_feat = self.vit(x)

        # Fusion
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

            if img is None:
                # Return zeros instead of failing to keep training stable
                return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))

            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            clahe=cv2.createCLAHE(2.0,(8,8))
            img=clahe.apply(img)
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img=Image.fromarray(img)
            label=torch.tensor(row["label_vec"],dtype=torch.float32)

            if self.transform:
                img=self.transform(img)

            return img,label
        except:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(len(TARGET_CLASSES))


# ---------------- DATALOADER ---------------- #

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

    patients=df["Patient ID"].unique()

    np.random.seed(42)
    np.random.shuffle(patients)

    train_p=patients[:int(0.8*len(patients))]
    val_p=patients[int(0.8*len(patients)):int(0.9*len(patients))]

    train_df=df[df["Patient ID"].isin(train_p)]
    val_df=df[df["Patient ID"].isin(val_p)]

    transform=transforms.Compose([

        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1,contrast=0.1),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    train_loader=DataLoader(
        ChestXrayDataset(train_df,DATA_DIR,transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader=DataLoader(
        ChestXrayDataset(val_df,DATA_DIR,transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader,val_loader


# ---------------- EVALUATION ---------------- #

def evaluate(model,loader):
    model.eval()
    preds=[]
    labels=[]

    with torch.no_grad():
        for x,y in tqdm(loader, desc="Validating"):
            x=x.to(device)
            out = torch.sigmoid(model(x)).cpu().numpy()
            preds.append(out)
            labels.append(y.numpy())

    y_true=np.vstack(labels)
    y_pred=np.vstack(preds)

    results = []
    for i, cls in enumerate(TARGET_CLASSES):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            results.append({'Class': cls, 'AUC': auc})
        except:
            results.append({'Class': cls, 'AUC': 0})
    
    macro_auc = roc_auc_score(y_true, y_pred, average="macro")
    return pd.DataFrame(results), macro_auc

# ---------------- TRAIN ---------------- #

def train():
    train_loader,val_loader=prepare_loaders()
    model=HybridResearchModel().to(device)
    criterion=nn.BCEWithLogitsLoss()
    optimizer=optim.AdamW(model.parameters(),lr=1e-4)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)

    best_auc=0
    for epoch in range(EPOCHS):
        model.train()
        pbar=tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x,y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out=model(x)
            loss=criterion(out,y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{float(loss):.6f}")

        res_df, val_auc = evaluate(model,val_loader)
        print(f"✅ Epoch {epoch+1} Macro AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            res_df.to_csv(RESULTS_CSV_PATH, index=False)
            print(f"⭐ Best Model and Results saved to {RESULTS_CSV_PATH}")

        scheduler.step()
        if device.type=="mps":
            torch.mps.empty_cache()


if __name__=="__main__":

    train()