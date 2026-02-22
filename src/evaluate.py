import os
import sys
# Ensure project root is on sys.path so models and config import correctly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.dirname(__file__))

from models.model_architecture import build_model
from config import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1],3))

    model.load_weights(MODEL_WEIGHTS)

    results = model.evaluate(test_generator)

    print("Test Loss:", results[0])
    print("Test Accuracy:", results[1])

if __name__ == '__main__':
    main()
