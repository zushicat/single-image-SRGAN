from data_loader import DataLoader

import numpy as np
from PIL import Image
from tensorflow import keras


MODEL = "../model/image_generator_model.h5"

FILE_INPUT_PATH = "/Users/karin/programming/SRGAN/Keras-Gan/train_test_images/images/val_LR"
FILE_OUTPUT_PATH = "../test_predictions/model_predictions"


if __name__ == "__main__":
    model = keras.models.load_model(MODEL)  # load the trained model (generator)
    data_loader = DataLoader()  # makes your life easier 

    # ****
    # my fixed test image collection
    # ***
    file_names = [
        "354400_5643700_354500_5643800.png",
        "354600_5643700_354700_5643800.png",
        "354900_5642600_355000_5642700.png",
        "355100_5643700_355200_5643800.png",
        "355300_5642700_355400_5642800.png"
    ]

    for file_name in file_names:
        # load image with a fixed input size (of 400x400 pixel)
        input_image = data_loader.load_single_image(f"{FILE_INPUT_PATH}/{file_name}", 400)
        
        output_image = model.predict(input_image)  # predict image
        output_image = 0.5 * output_image + 0.5  # de-normalize image array
        
        output_image = Image.fromarray((np.uint8(output_image*255)[0]))  # change data format
        output_image.save(f"{FILE_OUTPUT_PATH}/{file_name}")  # save image
