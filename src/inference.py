import os

from data_loader import DataLoader

import numpy as np
from PIL import Image
from tensorflow import keras


MODEL = "../model/image_generator_model.h5"

# FILE_INPUT_PATH = "/Users/karin/programming/SRGAN/Keras-Gan/train_test_images/images/default_test_images/lr"
FILE_INPUT_PATH = "/Users/karin/programming/data/ortho-images/ortho_400/2013_altstadt_nord"
FILE_OUTPUT_PATH = "../test_predictions/model_predictions"

IMAGE_IN_SIZE = 400
IMAGE_OUT_SIZE = 400


if __name__ == "__main__":
    model = keras.models.load_model(MODEL)  # load the trained model (generator)
    data_loader = DataLoader()  # makes your life easier 

    # ****
    # a fixed collection of controle images
    # ***
    # file_names = [
    #     "354400_5643700_354500_5643800.png",
    #     "354600_5643700_354700_5643800.png",
    #     "354900_5642600_355000_5642700.png",
    #     "355100_5643700_355200_5643800.png",
    #     "355300_5642700_355400_5642800.png"
    # ]

    file_names = [
        "355174_5644507_355274_5644607.png"
    ]

    def create_img_dir():
        save_img_dir = f"{FILE_OUTPUT_PATH}/{IMAGE_IN_SIZE}_{IMAGE_OUT_SIZE}"
        if not os.path.exists(save_img_dir):
            os.makedirs(f"{save_img_dir}/in_{IMAGE_IN_SIZE}")
            os.makedirs(f"{save_img_dir}/out_{IMAGE_OUT_SIZE}")
        return save_img_dir
    

    def resolve_single_image(file_name):
        input_image = data_loader.load_single_image(f"{FILE_INPUT_PATH}/{file_name}", IMAGE_IN_SIZE)
        
        output_image = model.predict(input_image)  # predict image
       
        # ***
        # for both in/out image: de-normalize image colors and change data format
        # ***
        input_image = 0.5 * input_image + 0.5
        input_image = Image.fromarray((np.uint8(input_image*255)[0]))
        
        output_image = 0.5 * output_image + 0.5
        output_image = Image.fromarray((np.uint8(output_image*255)[0]))
        
        # ***
        # resize out image
        # ***
        output_image = output_image.resize((IMAGE_OUT_SIZE, IMAGE_OUT_SIZE), Image.BICUBIC)
        
        return input_image, output_image


    save_img_dir = create_img_dir()
    
    for file_name in file_names:
        input_image, output_image = resolve_single_image(file_name)
        
        input_image.save(f"{save_img_dir}/in_{IMAGE_IN_SIZE}/{file_name}")  # save image
        output_image.save(f"{save_img_dir}/out_{IMAGE_OUT_SIZE}/{file_name}")  # save image

    # for file_name in file_names:
    #     # load image with a fixed input size (of 400x400 pixel)
    #     input_image = data_loader.load_single_image(f"{FILE_INPUT_PATH}/{file_name}", IMAGE_IN_SIZE)
    #     #output_image = output_image.resize((IMAGE_IN_SIZE, IMAGE_IN_SIZE), Image.BICUBIC)
        
    #     output_image = model.predict(input_image)  # predict image
    #     output_image = 0.5 * output_image + 0.5  # de-normalize image colors
        
    #     output_image = Image.fromarray((np.uint8(output_image*255)[0]))  # change data format
    #     output_image = output_image.resize((IMAGE_OUT_SIZE, IMAGE_OUT_SIZE), Image.BICUBIC)
        
    #     output_image.save(f"{FILE_OUTPUT_PATH}/{file_name}")  # save image
