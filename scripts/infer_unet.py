import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

def normalize_ct(arr, hu_min=-100.0, hu_max=400.0):
    """
    Clips CT intensities to [hu_min, hu_max] and scales to [0, 1].
    """
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min)
    return arr.astype(np.float32)

def postprocess_mask(mask):
    """
    Removes small connected components (<1% of total) and closes holes in 3D.
    """
    labeled, num = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    thresh = 0.01 * mask.sum()
    clean = np.zeros_like(mask)
    for i, sz in enumerate(sizes, start=1):
        if sz >= thresh:
            clean[labeled == i] = 1
    clean = ndimage.binary_closing(clean, structure=np.ones((3,3,3))).astype(np.uint8)
    return clean

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
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
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

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
        d4 = self.up4(bot)
        d4 = torch.cat([d4, enc4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, enc3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, enc2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, enc1], dim=1)
        d1 = self.dec1(d1)
        return self.final_conv(d1)


def infer_on_volume(
    ct_nifti_path: str,
    model: nn.Module,
    device: torch.device,
    output_mask_nifti: str,
    threshold: float = 0.5
):
    """
    Slice-by-slice inference for Task02_Heart.
    Saves binary heart mask as NIfTI.
    """
    # Load CT and normalize
    img_ct = sitk.ReadImage(ct_nifti_path)
    arr_ct = sitk.GetArrayFromImage(img_ct).astype(np.float32)
    arr_norm = normalize_ct(arr_ct) if arr_ct.max() > 1.0 else arr_ct

    Z, Y, X = arr_norm.shape
    pred_vol = np.zeros((Z, Y, X), dtype=np.uint8)

    model.eval()
    for z in range(Z):
        slice_img = arr_norm[z]
        inp = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            prob = torch.sigmoid(logits).cpu().numpy()[0,0]
        pred_vol[z] = (prob > threshold).astype(np.uint8)

    # Postprocess and save
    pred_clean = postprocess_mask(pred_vol)
    mask_img = sitk.GetImageFromArray(pred_clean)
    mask_img.CopyInformation(img_ct)
    os.makedirs(os.path.dirname(output_mask_nifti), exist_ok=True)
    sitk.WriteImage(mask_img, output_mask_nifti)
    print(f"Saved heart mask: {output_mask_nifti}")


if __name__ == "__main__":
    """
    Usage:
      python infer_unet_task02_heart.py \
        <ct_norm.nii.gz> <checkpoint.pth> <output_mask.nii.gz>
    """
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    ct_path    = sys.argv[1]
    checkpoint = sys.argv[2]
    out_mask   = sys.argv[3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    infer_on_volume(ct_path, model, device, out_mask)
