# First Homework - AN2DL

## Description
Image classification of blood cell samples (96x96 RGB) into **8 classes** using deep neural networks.  
The main challenges were **class imbalance**, **noisy data**, and **overfitting**.

## Structure
- `Final Delivery/`  
  - `AN2DL_report.pdf`: final report with methodology and results  
  - Final notebook and trained model  
- `notebooks/`  
  - Experiments, baseline models, and intermediate tests

## Approach
- Dataset cleaning (removing mislabeled images, outliers, and duplicates)
- Data augmentation (flip, rotation, brightness, MixUp, CutMix, etc.)
- Transfer learning with:
  - MobileNetV3Small (baseline)
  - EfficientNetV2B3
  - ConvNeXtXLarge (best-performing model)
- Regularization: L2, dropout, early stopping
- Class weights to handle imbalance

## Results
- **MobileNetV3Small:** ~75% accuracy  
- **EfficientNetV2B3:** ~86% accuracy  
- **ConvNeXtXLarge:** up to **90% accuracy** on Codabench  

📄 Full details available in `Final Delivery/AN2DL_report.pdf`
