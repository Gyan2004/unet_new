# Chest X-ray Classification (4-Class)

Description:
This project uses a CNN model to classify chest X-ray images into
4 categories using a custom TensorFlow/Keras CNN.

Dataset:
Chest X-ray dataset

Model:
Custom CNN implemented using TensorFlow/Keras.
- Input: 224x224 RGB images
- Output: 4-class classification
- Total parameters: 11.17M

Note:
The model was trained on Google Colab and the trained
weights (`best_model.weights.h5`) are used for evaluation
to avoid retraining.

Setup:
```bash
pip install -r requirements.txt
```

Model Weights:
**Important**: The file `best_model.weights.h5` (127.86 MB) is required but not included in this repo due to GitHub's file size limits.

To use this project, place the weights file in the `models/` directory:
```
models/
  ├── best_model.weights.h5   ← Place your weights file here
  └── model_architecture.py
```

If you don't have the weights file:
- Download from your Google Colab export / backup location
- Or train the model: `python src/train.py` (requires training data in `data/train/` and `data/val/`)

How to run:
- **Test the model**: `python test_model.py` (requires weights file)
- **Evaluate on test set**: `python src/evaluate.py` (requires test data in `data/test/`)
- **Predict on a single image**: `python src/predict.py` (requires weights file)

Architecture:
- Conv2D (32 filters) + MaxPooling
- Conv2D (64 filters) + MaxPooling  
- Conv2D (128 filters) + MaxPooling
- Flatten + Dense(128) + Dense(4-softmax)
# unet_new