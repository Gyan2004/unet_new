import os
import sys
# Make project root importable so sibling packages (models, config) resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.dirname(__file__))

from models.model_architecture import build_model
from data_loader import load_data
from config import *
import tensorflow as tf

def main():
    train_gen, val_gen = load_data()

    model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    model.save_weights(MODEL_WEIGHTS)

if __name__ == '__main__':
    main()
