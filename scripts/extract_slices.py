import os
import sys
import cv2
import numpy as np
import SimpleITK as sitk

def normalize_ct(arr, hu_min=-100.0, hu_max=400.0):
    """
    Clips CT intensities to [hu_min, hu_max] and scales to [0, 1].
    Useful preprocessing for Task02_Heart volumes.
    """
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min)
    return arr.astype(np.float32)

def extract_slices(
    ct_nifti: str,      # Path to normalized CT (float32 [0,1])
    mask_nifti: str,    # Path to label (.nii.gz) with class IDs {0,1,2,3}
    out_img_dir: str,
    out_mask_dir: str
):
    """
    Reads a 3D CT and corresponding mask volume from Task02_Heart.
    For each axial slice (Z dimension):
      - saves CT slice as 8-bit PNG (values 0–255)
      - saves mask slice as binary PNG where any heart label >0 becomes 255
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # Load CT volume
    img_ct = sitk.ReadImage(ct_nifti)
    arr_ct = sitk.GetArrayFromImage(img_ct).astype(np.float32)  # shape (Z, Y, X)

    # If CT is not already normalized, normalize now
    if arr_ct.max() > 1.0:
        arr_ct = normalize_ct(arr_ct)

    # Load mask volume
    img_mask = sitk.ReadImage(mask_nifti)
    arr_mask = sitk.GetArrayFromImage(img_mask).astype(np.uint8)   # shape (Z, Y, X)

    Z, Y, X = arr_ct.shape
    for z in range(Z):
        # Convert CT slice to 8-bit
        ct_slice = (arr_ct[z] * 255.0).astype(np.uint8)
        # Binarize mask: any heart structure >0 -> 255
        mask_slice = (arr_mask[z] > 0).astype(np.uint8) * 255

        # Construct filenames: patientID_sliceXXX.png
        base = os.path.basename(ct_nifti)
        # Remove .nii or _norm suffix
        name_root = base.replace('_norm.nii.gz', '').replace('.nii.gz', '')
        img_name = f"{name_root}_slice{z:03d}.png"

        # Write PNGs
        cv2.imwrite(os.path.join(out_img_dir, img_name), ct_slice)
        cv2.imwrite(os.path.join(out_mask_dir, img_name), mask_slice)

    print(f"Extracted {Z} axial slices from {ct_nifti} to {out_img_dir} + masks to {out_mask_dir}")

if __name__ == "__main__":
    """
    Usage:
      python extract_slices_task02_heart.py \
        <ct_norm.nii.gz> <label.nii.gz> \
        <output_image_folder> <output_mask_folder>
    """
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    ct_path    = sys.argv[1]
    mask_path  = sys.argv[2]
    out_img    = sys.argv[3]
    out_mask   = sys.argv[4]
    extract_slices(ct_path, mask_path, out_img, out_mask)
