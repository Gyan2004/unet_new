#!/usr/bin/env python3
"""
Quick test to verify the model loads and works correctly.
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from models.model_architecture import build_model
from config import *

def main():
    print("=" * 60)
    print("Model Loading Test")
    print("=" * 60)
    
    # Build and load model
    print(f"\n1. Building model with input shape (224, 224, 3)...")
    model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    print(f"   ✓ Model built: {len(model.layers)} layers")
    
    print(f"\n2. Loading weights from {MODEL_WEIGHTS}...")
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"   ✗ ERROR: Weights file not found at {MODEL_WEIGHTS}")
        return False
    
    model.load_weights(MODEL_WEIGHTS)
    print(f"   ✓ Weights loaded successfully")
    print(f"   ✓ Total parameters: {model.count_params():,}")
    
    # Compile model
    print(f"\n3. Compiling model...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"   ✓ Model compiled")
    
    # Test prediction with random data
    print(f"\n4. Testing prediction with random input...")
    test_input = np.random.rand(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32)
    prediction = model.predict(test_input, verbose=0)
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {prediction.shape}")
    print(f"   Predicted class: {np.argmax(prediction[0])}")
    print(f"   Confidence: {np.max(prediction[0]):.4f}")
    print(f"   ✓ Prediction successful")
    
    # Print model summary
    print(f"\n5. Model Summary:")
    print(f"   {'-' * 56}")
    model.summary()
    
    # Print class information
    print(f"\n6. Classification Classes:")
    print(f"   {'-' * 56}")
    for idx, cls in enumerate(CLASS_NAMES):
        print(f"   Class {idx}: {cls}")
    
    print(f"\n{'=' * 60}")
    print("✓ All tests passed! Model is ready for evaluation.")
    print(f"{'=' * 60}\n")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
