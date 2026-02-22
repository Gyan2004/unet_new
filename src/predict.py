import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.dirname(__file__))

import numpy as np
from tensorflow.keras.preprocessing import image
from models.model_architecture import build_model
from config import *

def main(img_path="sample_image.jpg", verbose=True):
    model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1],3))
    model.load_weights(MODEL_WEIGHTS)

    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Prediction Results:")
        print(f"Image: {img_path}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"{'='*60}")
        print(f"\nClass Probabilities:")
        for cls, prob in zip(CLASS_NAMES, prediction[0]):
            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            print(f"  {cls:15s} │ {bar} │ {prob:.4f} ({prob*100:.2f}%)")
        print(f"{'='*60}\n")
    
    return predicted_class, confidence

if __name__ == '__main__':
    main()
