# Classification-Segmentation-Project

This repository contains our **CSE428** course project on **brain tumor detection**, combining:
## 🌐 Live Demo (Streamlit)
We deployed the multitask U-Net model online using Streamlit + Hugging Face model hosting:

🔗 BriscApp: https://classification-segmentation-project-soicmh9wwa5dqkbwswufrq.streamlit.app/ 

---
-  **Tumor type classification (4-class)**
-  **Tumor segmentation (pixel-level mask prediction)**

We built a multitask deep learning pipeline using **U-Net** and evaluated it against **Attention U-Net** for segmentation.

---

##  Dataset
We used the **BRISC 2025 Brain Tumor Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/briscdataset/brisc2025

Classes:
- No Tumor  
- Glioma  
- Meningioma  
- Pituitary  

---

##  Models Used

### 1) Multitask U-Net (Segmentation + Classification)
A standard U-Net backbone with:
- Segmentation head → predicts tumor mask
- Classification head → predicts tumor type (4-class)

### 2) Attention U-Net (Segmentation only)
Used as a segmentation baseline to compare performance.


---

🔗 Hugging Face Model: https://huggingface.co/MihirDas/brisc-unet  
(contains `unet_multitask_4cls_best.pth`)


---

##  Team Members
- **Mihir Das** - https://github.com/MIhirDas10      (me)

- **Digonta Das** - https://github.com/DigontaDas  

- **Hasnain Ashraf** - https://github.com/Hasu121  

---
