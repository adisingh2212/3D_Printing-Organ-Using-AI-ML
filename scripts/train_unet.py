import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm

# -------------- Utility Functions --------------

def dice_coeff(pred, target, smooth=1e-6):
    """Computes Dice coefficient for binary masks."""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    tgt_flat = target.view(-1)
    intersection = (pred_flat * tgt_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + tgt_flat.sum() + smooth)

class DiceBCELoss(nn.Module):
    """Combines Dice loss + BCE loss."""
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = 1 - dice_coeff(inputs, targets)
        return bce_loss + dice_loss

# -------------- Dataset Class --------------

class SliceDataset(Dataset):
    """
    Loads CT and mask PNG pairs from a list file.
    Each line:
      <ct_slice.png> <mask_slice.png>
    Both are single-channel 8-bit.
    """
    def __init__(self, list_file):
        with open(list_file, 'r') as f:
            self.pairs = [line.strip().split() for line in f if line.strip()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        # To tensors [1, H, W]
        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0).clamp(0,1)
        return {'image': img_t.float(), 'mask': mask_t.float()}

# -------------- U-Net Definition --------------

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = CBR(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = CBR(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        e2 = self.pool1(enc1)
        enc2 = self.enc2(e2)
        e3 = self.pool2(enc2)
        enc3 = self.enc3(e3)
        e4 = self.pool3(enc3)
        enc4 = self.enc4(e4)
        e5 = self.pool4(enc4)
        bot = self.bottleneck(e5)
        d4 = self.up4(bot); d4 = torch.cat([d4, enc4], 1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, enc3], 1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, enc2], 1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, enc1], 1); d1 = self.dec1(d1)
        return self.final_conv(d1)

# -------------- Training Loop --------------

def train_model(
    train_list: str,
    val_list: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_ds = SliceDataset(train_list)
    val_ds = SliceDataset(val_list)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    os.makedirs('models', exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train(); train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} - Train'):
            imgs = batch['image'].to(device)
            msks = batch['mask'].to(device)
            preds = model(imgs)
            loss = criterion(preds, msks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch}/{epochs} - Val'):
                imgs = batch['image'].to(device)
                msks = batch['mask'].to(device)
                val_loss += criterion(model(imgs), msks).item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

        # Save model if improved
        if val_loss < best_val:
            best_val = val_loss
            pt_path = f'models/unet_task02_heart_best.pth'
            torch.save(model.state_dict(), pt_path)
            print(f'  [Saved best model: {pt_path}]')

    print('Training complete. Best Val Loss:', best_val)

if __name__ == '__main__':
    """
    Usage:
      python train_unet_task02_heart.py
    Update the train_list and val_list paths below to match Task02_Heart slice lists.
    """
    train_list = "E:/3D_Organs/data_preprocessed/Task02_Heart/train_list.txt"
    val_list = "E:/3D_Organs/data_preprocessed/Task02_Heart/val_list.txt"
    train_model(train_list, val_list, epochs=10, batch_size=8, lr=1e-4)
