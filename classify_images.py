#!/usr/bin/env python3
"""
Classify all images in the images_to_test folder
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image
from models.model_architecture import build_model
from config import *

def classify_image(model, img_path):
    """Classify a single image"""
    try:
        # Load image
        img = Image.open(img_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((IMAGE_SIZE[0], IMAGE_SIZE[1]))
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_batch, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        return predicted_class, confidence, prediction[0]
    except Exception as e:
        return None, None, None

def main():
    print("\n" + "=" * 90)
    print("IMAGE CLASSIFICATION - CHEST X-RAY DIAGNOSIS")
    print("=" * 90)
    
    # Check if model weights exist
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"\n✗ ERROR: Model weights not found at {MODEL_WEIGHTS}")
        return False
    
    # Build and load model
    print(f"\n📦 Loading model...")
    model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.load_weights(MODEL_WEIGHTS)
    print(f"   ✓ Model loaded: {model.count_params():,} parameters")
    
    # Get image directory
    img_dir = 'images_to_test'
    
    if not os.path.exists(img_dir):
        print(f"\n✗ ERROR: Directory {img_dir} not found")
        return False
    
    # Get all image files
    image_files = [f for f in os.listdir(img_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"\n✗ ERROR: No images found in {img_dir}")
        return False
    
    print(f"\n📁 Found {len(image_files)} image(s) to classify")
    print("\n" + "─" * 90)
    print(f"{'FILENAME':<30} {'PREDICTED CLASS':<20} {'CONFIDENCE':<15} {'PROBABILITIES':<25}")
    print("─" * 90)
    
    results = []
    
    for img_file in sorted(image_files):
        img_path = os.path.join(img_dir, img_file)
        predicted_class, confidence, probs = classify_image(model, img_path)
        
        if predicted_class is None:
            print(f"{img_file:<30} {'ERROR':<20} {'-':<15} {'Failed to load':<25}")
            continue
        
        results.append({
            'filename': img_file,
            'class': predicted_class,
            'confidence': confidence,
            'probs': probs
        })
        
        # Format probability display
        prob_str = f"{confidence:.2%}"
        
        print(f"{img_file:<30} {predicted_class:<20} {prob_str:<15}", end="")
        
        # Show other probabilities
        other_probs = []
        for i, cls in enumerate(CLASS_NAMES):
            if cls != predicted_class:
                other_probs.append(f"{cls}:{probs[i]:.1%}")
        print(" | ".join(other_probs[:2]))
    
    # Print detailed results
    print("\n" + "=" * 90)
    print("DETAILED RESULTS")
    print("=" * 90)
    
    for result in results:
        print(f"\n📄 {result['filename']}")
        print(f"   {'-' * 80}")
        print(f"   Predicted Class: {result['class']}")
        print(f"   Confidence:      {result['confidence']:.2%}")
        print(f"\n   Class Probabilities:")
        for i, cls in enumerate(CLASS_NAMES):
            prob = result['probs'][i]
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            marker = " ← PREDICTED" if cls == result['class'] else ""
            print(f"      {cls:15s} │ {bar} │ {prob:.4f} ({prob*100:6.2f}%){marker}")
    
    print(f"\n{'=' * 90}")
    print(f"✓ Classification completed for {len(results)}/{len(image_files)} images")
    print(f"{'=' * 90}\n")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
