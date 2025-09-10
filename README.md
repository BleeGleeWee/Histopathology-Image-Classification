# Histopathology Image Classification | CNN & Grad-CAM

*"That which is measured, improves." – Karl Pearson*

This repository demonstrates a Convolutional Neural Network (CNN) model for **histopathology image classification**, trained on the **PatchCamelyon (PCam) dataset**. The goal is to classify small image patches of lymph node tissue as **tumor (metastatic)** or **normal**.

---

## **Dataset: PatchCamelyon (PCam)**

- PCam is a benchmark dataset for binary classification of histopathology images.  
- It contains **327,680 color images (96x96 px)** extracted from lymph node scans.  
- Each image has a **binary label** indicating the presence of metastatic tissue.  
- Green boxes in example images indicate **tumor tissue**, which determines a positive label.  
![pcam](https://github.com/user-attachments/assets/b8ef1762-de38-4747-a648-914b075b1b35)

**Dataset link:** [PCam on GitHub](https://github.com/basveeling/pcam)

> PCam provides a challenging benchmark: larger than CIFAR-10 but smaller than ImageNet, and it is trainable on a single GPU.

---

## **Project Overview**

- **Model:** CNN-based architecture with `ResNet50` as the backbone.  
- **Transfer Learning:** Applied to speed up training.  
- **Interpretability:** Grad-CAM heatmaps visualize which regions influence predictions.  

**Current Implementation Notes:**
- For educational purposes, a **smaller dataset** and **fewer epochs** were used.  
- This results in **moderate accuracy, precision, and F1-scores**.  
- With **larger datasets and longer training**, the model would likely achieve higher performance.

---

## **Installation**

```bash
# Clone the repository
git clone https://github.com/BleeGleeWee/Histopathology-Image-Classification.git
cd Histopathology-Image-Classification

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
````

---

## **Usage**

1. Download PCam `.h5` files from [PCam dataset](https://github.com/basveeling/pcam).
2. Place them inside the `data/` folder.
3. Open Jupyter Notebook `notebooks/Histopathology_CNN_GradCAM.ipynb`.
4. Run the notebook step-by-step to:

   * Load and preprocess the data
   * Train the CNN model
   * Evaluate model metrics
   * Visualize Grad-CAM overlays

---

## **Handling `.h5` files**

* Use `h5py` or `HDF5Matrix` from Keras to efficiently load large datasets.
* **Important:** Loading the entire dataset in memory may cause **MemoryError**. Consider **streaming batches** using generators for large datasets.

---

## **Evaluation Metrics**

Example results using smaller dataset:

| Metric    | Class 0 (Normal) | Class 1 (Tumor) | Overall |
| --------- | ---------------- | --------------- | ------- |
| Precision | 0.65             | 0.82            | 0.74    |
| Recall    | 0.88             | 0.54            | 0.71    |
| F1-score  | 0.75             | 0.65            | 0.70    |
| Accuracy  | -                | -               | 0.71    |

* **Interpretation:** Model detects normal patches better than tumor patches.
* **Reason:** Smaller dataset and fewer epochs limit learning capacity.
* **Expectation:** Using **full PCam dataset** and **more training epochs** should produce better performance.

---

## **Grad-CAM Visualization**

* Highlights regions in the images that the CNN focuses on.
* Overlayed heatmaps help interpret the model’s decisions.

Example Grid:

| Original Images                          | Grad-CAM Overlays                      |
| ---------------------------------------- | -------------------------------------- |
| ![Original](images/original_example.png) | ![Overlay](images/gradcam_example.png) |

---

## **Repository Structure**

```
Histopathology-Image-Classification/
│
├── data/                         # PCam dataset .h5 files
│   ├── camelyonpatch_level_2_split_train_x.h5
│   ├── camelyonpatch_level_2_split_train_y.h5
│   ├── camelyonpatch_level_2_split_valid_x.h5
│   └── camelyonpatch_level_2_split_valid_y.h5
│
├── notebooks/                     # Jupyter notebooks
│   └── Histopathology_CNN_GradCAM.ipynb
│
├── src/                           # Helper scripts (optional)
│   ├── model.py                   # CNN/ResNet model building
│   ├── data_loader.py             # Functions to load h5 data
│   └── gradcam.py                 # Grad-CAM functions
│
├── images/                        # Example images and Grad-CAM overlays
│
├── requirements.txt               # Dependencies
├── README.md
└── .gitignore
```

---

## **Contributing**

* This repository is for **educational purposes only**.
* Everyone is encouraged to **submit pull requests** or **issues** to improve the project.
* Suggestions for **optimizing the model**, adding **more visualizations**, or **better data handling** are welcome.

---

## **References**

* [PatchCamelyon (PCam) GitHub](https://github.com/basveeling/pcam)


---

