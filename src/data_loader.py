'''
Get random images from path defined in BASE_PATH
- augment image
- take random crop from image (parameter "crop_size"): hr (original) image crop
- downscale by factor (parameter "scale_factor"): lr version of hr image crop
- apply on number of images (parameter "batch_size")
- normalize batch images ([0, 255] -> [-1, 1])
'''
from glob import glob
import os

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


BASE_PATH = "/Users/karin/programming/data/ortho-images/cologne_2019_400_400"


class DataLoader():
    def __init__(self, crop_size=96, scale_factor=4):
        self.datagen = ImageDataGenerator(
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode="reflect"
        )

        self.crop_size = crop_size
        self.scale_factor = scale_factor

    def crop_image(self, img):
        dist_to_image_border = 20
        upper_left = np.random.randint(low=dist_to_image_border, high=img.width-dist_to_image_border-self.crop_size)
        lower_right = upper_left+self.crop_size
        return img.crop((upper_left, upper_left, lower_right, lower_right))


    def load_data(self, batch_size=1, is_testing=False):
        image_dir = "val_HR"
        if is_testing is False:
            image_dir = "train_HR"
        
        images_path = glob(f'{BASE_PATH}/**/*.png', recursive=True)  # all image paths from BASE_PATH or it's subdirs
        batch_images = np.random.choice(images_path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_hr_path in batch_images:
            img_hr = load_img(img_hr_path)  # type: PIL image
            
            # ***
            # image augmentation
            # ***
            if is_testing is False:
                data = np.expand_dims(img_to_array(img_hr), 0)
                it = self.datagen.flow(data, batch_size=1)
                augmented_img_np_array = it.next()[0].astype('uint8')
                img_hr = Image.fromarray(augmented_img_np_array)

                img_hr = self.crop_image(img_hr)
                # img_hr.show()  # debug
            
            img_lr = img_hr.resize((self.crop_size//self.scale_factor, self.crop_size//self.scale_factor), Image.BICUBIC)
            # img_lr.show()  # debug

            imgs_hr.append(np.asarray(img_hr))
            imgs_lr.append(np.asarray(img_lr))

        # ***
        # normalize: -1, 1
        # ***
        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr


    def load_single_image(self, file_path, size):
        img = Image.open(file_path)
        img = img.resize((size, size))
        img_np_array = [np.asarray(img)]
        return np.array(img_np_array) / 127.5 - 1.



if __name__ == "__main__":
    data_loader = DataLoader(crop_size=72, scale_factor=4)
    imgs_hr, imgs_lr = data_loader.load_data()
    print(imgs_lr[0].shape, imgs_hr[0].shape)
