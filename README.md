# single-image-SRGAN
Enhance single images with super-resolution GAN.

The stored (trained) model (/model/image_generator_model.h5) aims to improve aerial photographs of average/poor quality.   
Since this is an universal approach, you can train your own model on image types you intend to improve.

### Results
The input images in this example have a size of400x400 pixel. The output of the generated high-res images is 1600x1600 pixel.    
(The high-res images in the following 2 examples are downscaled to 400x400 pixel.)

input examples             |  output examples
:-------------------------:|:-------------------------:
![input image 1](readme_images/input_1.png?raw=true) | ![output image 1](readme_images/output_1.png?raw=true)
![input image 2](readme_images/input_2.png?raw=true) | ![output image 2](readme_images/output_2.png?raw=true)



### Usage
Install environment and change into shell
```
$ pipenv install
```
```
$ pipenv shell
```
(Leave shell with "exit".)    

Change into directory /src.    

The model is trained on aerial images (with 144000 steps for pretraining, 410000 steps for training on 72x72 pixel image crops). 

- Use inference.py if you like to enhance images of this type (change the script accordingly to your needs).
- If you like to train your own image types
    - change the high-res image path (global variable BASE_PATH) in data_loader.py
    - define the global variables in srgan.py accordingly to your needs
    - pre-train the model
    - train the model

