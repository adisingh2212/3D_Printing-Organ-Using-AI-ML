import json
import os
import random
import sys

def create_splits(task_folder, val_ratio=0.20, seed=42):
    """
    Reads dataset.json in <task_folder>, shuffles the training entries,
    then writes train_list.txt & val_list.txt under <task_folder>.

    Each line in train_list.txt / val_list.txt is:
      <path_to_normed_CT.nii.gz> <path_to_label.nii.gz>
    """
    ds_json = os.path.join(task_folder, "dataset.json")
    with open(ds_json, "r") as f:
        ds = json.load(f)

    # Get training entries (Task02_Heart uses 'training')
    train_entries = ds.get("training", [])
    random.seed(seed)
    random.shuffle(train_entries)

    num_total = len(train_entries)
    num_val = int(num_total * val_ratio)
    val_split = train_entries[:num_val]
    train_split = train_entries[num_val:]

    # Prepare output paths
    train_txt = os.path.join(task_folder, "train_list.txt")
    val_txt   = os.path.join(task_folder, "val_list.txt")
    test_txt  = os.path.join(task_folder, "test_list.txt")

    # Write training split
    with open(train_txt, "w") as ft:
        for entry in train_split:
            img_rel = entry["image"]  # e.g., "imagesTr/heart_000.nii.gz"
            lbl_rel = entry["label"]  # e.g., "labelsTr/heart_000.nii.gz"
            img_norm = os.path.join(task_folder, img_rel.replace("imagesTr", "imagesTr_norm").replace(".nii.gz", "_norm.nii.gz"))
            lbl_p = os.path.join(task_folder, lbl_rel)
            ft.write(f"{img_norm} {lbl_p}\n")

    # Write validation split
    with open(val_txt, "w") as fv:
        for entry in val_split:
            img_rel = entry["image"]
            lbl_rel = entry["label"]
            img_norm = os.path.join(task_folder, img_rel.replace("imagesTr", "imagesTr_norm").replace(".nii.gz", "_norm.nii.gz"))
            lbl_p = os.path.join(task_folder, lbl_rel)
            fv.write(f"{img_norm} {lbl_p}\n")

    # Write test list (just CT volumes, no labels)
    with open(test_txt, "w") as ftst:
        for entry in ds.get("test", []):
            img_rel = entry.get("image")
            if img_rel:
                abs_p = os.path.join(task_folder, img_rel)
                ftst.write(f"{abs_p}\n")

    print(f"Wrote {len(train_split)} pairs to {train_txt}")
    print(f"Wrote {len(val_split)} pairs to {val_txt}")
    print(f"Wrote {len(ds.get('test', []))} lines to {test_txt}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_split_lists.py <path_to_Task02_Heart_folder>")
        sys.exit(1)
    create_splits(sys.argv[1])
