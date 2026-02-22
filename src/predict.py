import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.dirname(__file__))

import numpy as np
from tensorflow.keras.preprocessing import image
from models.model_architecture import build_model
from config import *

def main(img_path="sample_image.jpg"):
    model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1],3))
    model.load_weights(MODEL_WEIGHTS)

    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    print(prediction)

if __name__ == '__main__':
    main()
