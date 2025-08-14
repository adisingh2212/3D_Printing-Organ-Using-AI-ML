import os
import sys

def create_png_list(images_dir: str, masks_dir: str, out_list_path: str):
    """
    For Task02_Heart (MSD), writes a list file where each line is:
      <abs_path_to_ct_slice.png> <abs_path_to_mask_slice.png>
    images_dir: folder containing pre-extracted CT slices (PNG)
    masks_dir:  folder containing corresponding mask slices (PNG)
    out_list_path: path to write the list (train/val/test .txt)
    """
    images_dir = os.path.abspath(images_dir)
    masks_dir = os.path.abspath(masks_dir)

    png_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
    with open(out_list_path, 'w') as f:
        for fname in png_files:
            img_path = os.path.join(images_dir, fname)
            mask_path = os.path.join(masks_dir, fname)
            if not os.path.exists(mask_path):
                # Skip slices without a mask (e.g., background-only)
                continue
            f.write(f"{img_path} {mask_path}\n")

    print(f"Wrote {len(png_files)} entries to {out_list_path}")

if __name__ == '__main__':
    """
    Usage:
      python make_png_list_task02_heart.py \
        <images_folder> <masks_folder> <output_list.txt>
    """
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    imgs = sys.argv[1]
    masks = sys.argv[2]
    out_txt = sys.argv[3]
    create_png_list(imgs, masks, out_txt)
