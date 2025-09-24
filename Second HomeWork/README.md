# Second Homework - AN2DL

## Description
Second image classification task on blood cell samples (96x96 RGB, 8 classes), with a more complex dataset and stronger focus on **robustness** and **generalization**.  
Our solution ranked **3rd place in the final leaderboard**.

## Structure
- `Final Delivery/`  
  - `AN2DL_report.pdf`: final report with methodology and results  
  - Final notebook and trained model  
- `notebooks/`  
  - Experiments, baseline models, and additional tests

## Approach
- Advanced preprocessing (including K-means clustering for noise reduction)
- Extensive data augmentation with KerasCV (light, moderate, heavy)
- Transfer learning and fine-tuning with:
  - MobileNetV3Small
  - EfficientNetV2B3
  - ConvNeXtXLarge (best-performing model)
- Additional techniques explored:
  - Filtering (high-pass, nucleus segmentation, etc.)
  - Hyperparameter tuning with KerasTuner
- Adaptive learning rate scheduling

## Results
- **MobileNetV3Small:** ~75% accuracy  
- **EfficientNetV2B3:** ~87% accuracy  
- **ConvNeXtXLarge:** up to **90% accuracy** on Codabench  

📄 Full details available in `Final Delivery/AN2DL_report.pdf`
