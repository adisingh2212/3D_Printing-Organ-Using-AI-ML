# scripts/preprocess_nifti.py

import os
import sys
import numpy as np
import SimpleITK as sitk

def inspect_volume(nifti_path):
    """Prints file info: voxel size, spacing, dtype, intensity range, etc."""
    img = sitk.ReadImage(nifti_path)
    arr = sitk.GetArrayFromImage(img)
    print(f"File: {nifti_path}")
    print(f"  Size (voxels):   {img.GetSize()} (X, Y, Z)")
    print(f"  Spacing (mm):    {img.GetSpacing()} (X, Y, Z)")
    print(f"  Data type:       {arr.dtype}")
    print(f"  Intensity range: [{arr.min()}, {arr.max()}]")
    print(f"  Num slices (Z):  {arr.shape[0]}")
    print(f"  Slice shape:     {arr.shape[1]} × {arr.shape[2]}")
    if "labelsTr" in nifti_path or "labelsTs" in nifti_path:
        print(f"  Unique label values: {np.unique(arr)}")

def normalize_ct(arr, hu_min=-100, hu_max=400):
    """Clips CT array to [hu_min, hu_max] and scales to [0,1]."""
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min)
    return arr.astype(np.float32)

def resample_to_isotropic(itk_image, new_spacing=(1.0,1.0,1.0)):
    """Resamples an SITK image to isotropic spacing (default 1×1×1 mm)."""
    orig_spacing = itk_image.GetSpacing()
    orig_size = itk_image.GetSize()
    new_size = [
        int(round(orig_size[i] * (orig_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(itk_image)

def process_ct(input_nifti, output_nifti, do_resample=False):
    """
    1) Read a CT volume.
    2) (Optional) Resample to 1×1×1 mm.
    3) Clip intensities to [–100,400] HU and scale to [0,1].
    4) Save result as a new NIfTI.
    """
    img = sitk.ReadImage(input_nifti)
    if do_resample:
        img = resample_to_isotropic(img, new_spacing=(1.0,1.0,1.0))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr_norm = normalize_ct(arr)
    img_norm = sitk.GetImageFromArray(arr_norm)      # convert array back to SITK image
    img_norm.CopyInformation(img)                   # preserve metadata (spacing, origin)
    os.makedirs(os.path.dirname(output_nifti), exist_ok=True)
    sitk.WriteImage(img_norm, output_nifti)
    print(f"[CT] Saved normalized: {output_nifti}")

def resample_mask(input_nifti, output_nifti, do_resample=False):
    """
    1) Read a label mask volume.
    2) (Optional) Resample to 1×1×1 mm using nearest-neighbor interpolation.
    3) Save result as a new NIfTI.
    """
    img = sitk.ReadImage(input_nifti)
    if do_resample:
        orig_spacing = img.GetSpacing()
        orig_size = img.GetSize()
        new_spacing = (1.0,1.0,1.0)
        new_size = [
            int(round(orig_size[i] * (orig_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        img = resampler.Execute(img)
    os.makedirs(os.path.dirname(output_nifti), exist_ok=True)
    sitk.WriteImage(img, output_nifti)
    print(f"[Mask] Saved resampled: {output_nifti}")

if __name__ == "__main__":
    # Basic command parsing:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "inspect":
        nifti_path = sys.argv[2]
        inspect_volume(nifti_path)
    elif cmd == "ct_norm":
        if len(sys.argv) < 4:
            print("Usage: python preprocess_nifti.py ct_norm <in_ct.nii.gz> <out_ct.nii.gz> [--resample]")
            sys.exit(1)
        inp = sys.argv[2]
        outp = sys.argv[3]
        do_res = ("--resample" in sys.argv)
        process_ct(inp, outp, do_resample=do_res)
    elif cmd == "mask_resample":
        if len(sys.argv) < 4:
            print("Usage: python preprocess_nifti.py mask_resample <in_mask.nii.gz> <out_mask.nii.gz> [--resample]")
            sys.exit(1)
        inp = sys.argv[2]
        outp = sys.argv[3]
        do_res = ("--resample" in sys.argv)
        resample_mask(inp, outp, do_resample=do_res)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
