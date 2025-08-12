# 3D Heart Reconstruction and Printing using AI/ML

## 📌 Project Overview
This project demonstrates the complete pipeline for creating a 3D printable model of the human heart using MRI scans from the **Medical Segmentation Decathlon (MSD) Task_02 Heart** dataset.  
It leverages **AI/ML-based image segmentation**, medical image processing, and 3D model generation to create accurate anatomical models for biomedical education, reducing dependency on real organs.

---

## 🛠 Workflow

1. **Dataset Acquisition**  
   - Source: MSD Task_02 Heart  
   - Format: NIfTI (.nii.gz)  
   - Includes labeled MRI scans for healthy human hearts.

2. **Preprocessing**  
   - NIfTI to NumPy array conversion  
   - Image normalization and resizing  
   - Data augmentation for better model generalization  

3. **Segmentation Model**  
   - Model: **3D U-Net** for volumetric segmentation  
   - Input: 3D MRI volumes  
   - Output: Binary heart masks  

4. **3D Reconstruction**  
   - Segmentation masks converted into meshes using **Marching Cubes**  
   - Exported as `.stl` for 3D printing  

5. **3D Printing**  
   - STL files sliced using printer software (e.g., Cura, PrusaSlicer)  
   - Physical heart model printed for study and demonstration  

---

## 📊 Example Results

| Step | Output |
|------|--------|
| MRI Slice | ![MRI Slice](images/mri_slice.png) |
| Segmentation | ![Segmentation](images/segmentation.png) |
| 3D Model | ![3D Model](images/3d_model.png) |
| Printed Heart | ![Printed Heart](images/printed_heart.png) |

---

## 📂 Dataset
- [Medical Segmentation Decathlon - Task_02 Heart](http://medicaldecathlon.com/)  
- License: CC-BY-SA 4.0  

---

## 🚀 How to Run
```bash
git clone https://github.com/yourusername/3d-heart-printing.git
cd 3d-heart-printing
pip install -r requirements.txt
python train.py
python reconstruct_3d.py
