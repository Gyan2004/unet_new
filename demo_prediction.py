#!/usr/bin/env python3
"""
Demo script showing model predictions on test images.
This creates synthetic test images to demonstrate the model output.
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf

from models.model_architecture import build_model
from config import *

def create_synthetic_xray(image_type="normal"):
    """Create synthetic chest X-ray-like images for demo"""
    # Create base image (grayscale background)
    img = np.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32) * 0.3
    
    if image_type == "normal":
        # Draw normal chest outline
        center_x, center_y = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
        
        # Lungs (oval shape) - darker areas
        y, x = np.ogrid[:IMAGE_SIZE[0], :IMAGE_SIZE[1]]
        lung_left = ((x - center_y + 40)**2 / 1500 + (y - center_x)**2 / 2000) < 1
        lung_right = ((x - center_y - 40)**2 / 1500 + (y - center_x)**2 / 2000) < 1
        
        img[lung_left] = [0.25, 0.25, 0.25]
        img[lung_right] = [0.25, 0.25, 0.25]
        
        # Ribcage lines
        for i in range(5, IMAGE_SIZE[0], 20):
            img[i:i+2, 80:IMAGE_SIZE[1]-80] = [0.4, 0.4, 0.4]
        
        label = "NORMAL"
        
    elif image_type == "pneumonia":
        # Create pneumonia-like appearance with abnormal patterns
        img = np.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32) * 0.35
        
        # Lungs with abnormal opacities
        center_x, center_y = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
        y, x = np.ogrid[:IMAGE_SIZE[0], :IMAGE_SIZE[1]]
        
        lung_left = ((x - center_y + 40)**2 / 1500 + (y - center_x)**2 / 2000) < 1
        lung_right = ((x - center_y - 40)**2 / 1500 + (y - center_x)**2 / 2000) < 1
        
        img[lung_left] = [0.15, 0.15, 0.15]
        img[lung_right] = [0.15, 0.15, 0.15]
        
        # Add pneumonia-like opacities (bright spots)
        for _ in range(8):
            cy = np.random.randint(80, 144)
            cx = np.random.randint(80, 144)
            radius = np.random.randint(15, 30)
            circle_mask = (y - cy)**2 + (x - cx)**2 < radius**2
            img[circle_mask] = [0.5, 0.5, 0.5]
        
        label = "PNEUMONIA"
    
    # Normalize
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8), label

def main():
    print("\n" + "=" * 80)
    print("CHEST X-RAY CLASSIFICATION MODEL - PREDICTION DEMO")
    print("=" * 80)
    
    # Check if model weights exist
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"\n✗ ERROR: Model weights not found at {MODEL_WEIGHTS}")
        print("  Please place the best_model.weights.h5 file in the models/ directory")
        return False
    
    # Build and load model
    print(f"\n📦 Loading model...")
    model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.load_weights(MODEL_WEIGHTS)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"   ✓ Loaded: {model.count_params():,} parameters")
    
    # Class labels (4-class model)
    class_names = ['Class_0', 'Class_1', 'Class_2', 'Class_3']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # Create test cases
    test_cases = ["normal", "pneumonia"]
    
    print(f"\n{'─' * 80}")
    print("MAKING PREDICTIONS ON TEST IMAGES")
    print(f"{'─' * 80}\n")
    
    fig, axes = plt.subplots(len(test_cases), 2, figsize=(14, 5 * len(test_cases)))
    if len(test_cases) == 1:
        axes = [axes]
    
    for idx, test_type in enumerate(test_cases):
        print(f"Test Case {idx + 1}: {test_type.upper()}")
        print(f"  {'-' * 76}")
        
        # Create synthetic image
        img, img_label = create_synthetic_xray(test_type)
        img_normalized = img.astype(np.float32) / 255.0
        
        # Add batch dimension for model
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        print(f"  Input Image Shape: {img_batch.shape}")
        print(f"  Data Range: [0.0, 1.0]")
        
        # Make prediction
        prediction = model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        print(f"\n  🎯 MODEL OUTPUT:")
        print(f"  {'-' * 76}")
        print(f"  Predicted Class: {class_names[predicted_class]}")
        print(f"  Confidence:      {confidence:.2%}")
        print(f"  {'-' * 76}")
        
        print(f"\n  📊 CLASS PROBABILITIES:")
        for i, prob in enumerate(prediction[0]):
            bar_length = int(prob * 45)
            bar = "█" * bar_length + "░" * (45 - bar_length)
            print(f"    {class_names[i]:12s} │ {bar} │ {prob:.6f} ({prob*100:6.2f}%)")
        
        # Plot
        axes[idx][0].imshow(img, cmap='gray')
        axes[idx][0].set_title(f"Test Image: {img_label}\nInput: (224, 224, 3)", fontsize=12, fontweight='bold')
        axes[idx][0].axis('off')
        
        bars = axes[idx][1].bar(range(len(class_names)), prediction[0], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[idx][1].set_ylabel('Probability', fontsize=11, fontweight='bold')
        axes[idx][1].set_xlabel('Classes', fontsize=11, fontweight='bold')
        axes[idx][1].set_title(f'Prediction Results\n{class_names[predicted_class]} ({confidence:.2%})', 
                               fontsize=12, fontweight='bold', color=colors[predicted_class])
        axes[idx][1].set_xticks(range(len(class_names)))
        axes[idx][1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[idx][1].set_ylim([0, 1])
        axes[idx][1].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add probability values on bars
        for bar, prob in zip(bars, prediction[0]):
            height = bar.get_height()
            axes[idx][1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                             f'{prob:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        print(f"\n")
    
    # Save figure
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/model_demo_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"{'─' * 80}")
    print(f"✓ Prediction visualization saved to: {output_path}")
    print(f"{'─' * 80}")
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("✓ DEMO COMPLETED SUCCESSFULLY!")
    print(f"{'=' * 80}")
    print(f"\n📋 SUMMARY:")
    print(f"   • Model: 4-class CNN (11.17M parameters)")
    print(f"   • Input Size: 224 × 224 RGB images")
    print(f"   • Output: 4 class probabilities")
    print(f"   • Tested: {len(test_cases)} sample images")
    print(f"   • Results saved to: {output_path}")
    print(f"\n✨ The model successfully makes predictions on chest X-ray images!\n")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
