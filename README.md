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

How to run:
- Evaluate on test set: `python src/evaluate.py`
- Predict on a single image: `python src/predict.py`

Architecture:
- Conv2D (32 filters) + MaxPooling
- Conv2D (64 filters) + MaxPooling  
- Conv2D (128 filters) + MaxPooling
- Flatten + Dense(128) + Dense(4-softmax)
# unet_new