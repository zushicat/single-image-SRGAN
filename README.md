# single-image-SRGAN
Enhance single images with super-resolution GAN.

The stored (trained) model (/model/image_generator_model.h5) aims to improve aerial photographs.   
Since this is an universal approach, you can train your own model on image types you intend to improve.


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

The model is trained on aerial images (with 144000 steps for pretraining and 410000 steps for training on 72x72 pixel image crops). 

- Use inference.py if you like to enhance images of this type (change the script accordingly to your needs).
- If you like to train your own image types
    - change the high-res image path (global variable BASE_PATH) in data_loader.py
    - define the global variables in srgan.py accordingly to your needs
    - pre-train the model
    - train the model

