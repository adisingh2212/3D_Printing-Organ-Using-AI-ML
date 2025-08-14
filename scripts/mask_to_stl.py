import os
import sys
import numpy as np
import SimpleITK as sitk
from skimage import measure
import trimesh

def mask_to_stl(
    mask_nifti_path: str,      # e.g., outputs/preds/heart_0050_pred.nii.gz (0/1 mask)
    reference_nifti_path: str, # e.g., data_preprocessed/Task02_Heart/imagesTs/heart_0050.nii.gz
    output_stl_path: str,      # e.g., outputs/meshes/heart_0050.stl
    iso_level: float = 0.5,
    target_face_count: int = None
):
    """
    Converts a binary Task02_Heart mask to a 3D STL mesh:
    1. Reads the 3D mask volume (Z,Y,X).
    2. Reads the reference CT for spacing/origin metadata.
    3. Applies marching_cubes to extract vertices/faces.
    4. Builds a Trimesh object, cleans and deduplicates.
    5. Optionally decimates to <target_face_count> faces.
    6. Exports a watertight STL file.
    """
    # Load mask volume
    mask_img = sitk.ReadImage(mask_nifti_path)
    mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)

    # Load CT for spacing information
    ref_ct = sitk.ReadImage(reference_nifti_path)
    spacing = ref_ct.GetSpacing()  # (x_mm, y_mm, z_mm)
    # marching_cubes expects (z_spacing, y_spacing, x_spacing)
    spacing_mc = (spacing[2], spacing[1], spacing[0])

    # 1) Extract surface via marching cubes
    print(f"Running marching_cubes on mask: {mask_nifti_path}")
    verts, faces, normals, _ = measure.marching_cubes(
        volume=mask_arr,
        level=iso_level,
        spacing=spacing_mc
    )
    print(f"  Got {len(verts)} vertices and {len(faces)} faces")

    # 2) Build mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # 3) Cleanup topology
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()

    # 4) Decimate if requested
    if target_face_count and len(mesh.faces) > target_face_count:
        print(f"Decimating mesh from {len(mesh.faces)} to {target_face_count} faces...")
        mesh = mesh.simplify_quadratic_decimation(target_face_count)
        print(f"  Decimated to {len(mesh.faces)} faces")

    # 5) Ensure watertight
    if not mesh.is_watertight:
        print("Warning: mesh not watertight after cleanup. Attempting hole fill...")
        mesh = mesh.fill_holes()
        if not mesh.is_watertight:
            print("  Still not watertight, exporting anyway.")

    # 6) Export STL
    os.makedirs(os.path.dirname(output_stl_path), exist_ok=True)
    mesh.export(output_stl_path)
    print(f"Saved STL mesh to: {output_stl_path}")

if __name__ == '__main__':
    """
    Usage:
      python mask_to_stl_task02_heart.py \
        <mask.nii.gz> <reference_ct.nii.gz> <output.stl> [<target_face_count>]

    Example:
      python mask_to_stl_task02_heart.py \
        outputs/preds/heart_0050_pred.nii.gz \
        data_preprocessed/Task02_Heart/imagesTs/heart_0050.nii.gz \
        outputs/meshes/heart_0050.stl 50000
    """
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    mask_nifti = sys.argv[1]
    ref_ct = sys.argv[2]
    out_stl = sys.argv[3]
    tgt = int(sys.argv[4]) if len(sys.argv) == 5 else None

    mask_to_stl(mask_nifti, ref_ct, out_stl, iso_level=0.5, target_face_count=tgt)
