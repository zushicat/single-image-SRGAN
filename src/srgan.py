"""
SRGAN (Keras) implementations on github: 
https://github.com/eriklindernoren/Keras-GAN/blob/master/srgan/srgan.py
https://github.com/krasserm/super-resolution
https://github.com/MathiasGruber/SRGAN-Keras
https://github.com/deepak112/Keras-SRGAN/blob/master/simplified/
https://github.com/HasnainRaz/Fast-SRGAN
https://github.com/JGuillaumin/SuperResGAN-keras
https://github.com/AvivSham/SRGAN-Keras-Implementation
https://github.com/Aqsa-K/SRGAN-Keras

Some tips:
https://github.com/soumith/ganhacks (from around 2016; check for relevance)

Regarding checkerboard artifacts:
https://distill.pub/2016/deconv-checkerboard/
https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9F3A72B4581D101881B4A08C09150914/S2048770319000027a.pdf/checkerboard_artifacts_free_convolutional_neural_networks.pdf
https://medium.com/@hanspinckaers/artifacts-in-semantic-segmentation-networks-5d6826eeb431
https://arxiv.org/pdf/1707.02937.pdf
https://arxiv.org/abs/1609.05158
https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/

Regarding @tf.function
https://www.machinelearningplus.com/deep-learning/how-use-tf-function-to-speed-up-python-code-tensorflow/

Regarding learning rate / optimizer / decay / scheduler (i.e. ExponentialDecay):
https://stackoverflow.com/questions/56414605/keras-how-to-resume-training-with-adam-optimizer
https://stackoverflow.com/questions/57531409/how-is-learning-rate-decay-implemented-by-adam-in-keras
https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
https://stackoverflow.com/a/63089235 "... not using model.compile and instead performing automatic differentiation to apply the gradients manually with optimizer.apply_gradients ..."

Regarding GradientTape:
https://www.pyimagesearch.com/2020/03/23/using-tensorflow-and-gradienttape-to-train-a-keras-model/

Regarding VGG:
http://parneetk.github.io/blog/CNN-TransferLearning1/

Regarding activation:
https://datascience.stackexchange.com/questions/18583/what-is-the-difference-between-leakyrelu-and-prelu
"""

import datetime
import os
import pickle

from data_loader import DataLoader

import tensorflow as tf
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Input, LeakyReLU, Lambda, PReLU, UpSampling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

import numpy as np
from PIL import Image


PRETRAINED_GENERATOR_WEIGHTS = "../model/weights/pretrained_generator.h5"
GENERATOR_MODEL = "../model/image_generator_model.h5"

PRETRAINED_GENERATOR_CHECKPOINT_DIR = "../model/checkpoints/pre_train"
FINE_TUNE_CHECKPOINT_DIR = "../model/checkpoints/fine_tune"

IMG_DIR_VAL_LR = "/Users/karin/programming/SRGAN/Keras-Gan/train_test_images/images/default_test_images/lr_2"
IMG_DIR_PREDICTED = "../test_predictions"

HR_CROPPED_IMG_SIZE = 96
SCALE_FACTOR = 4

LEARNING_RATE = 1e-4


# **************************************
#
# **************************************
class Utils():
    # ***************
    # inference
    # ***************
    def test_predict(self, model, data_loader, lr_dir_path, trained_steps, sub_dir):
        ''' Create prediction example of selected epoch '''
        def create_img_dir(trained_steps):
            ''' My default test dump '''
            save_img_dir = f"{IMG_DIR_PREDICTED}/{sub_dir}/{trained_steps}"
            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir)
            return save_img_dir
        
        def resolve_single_image(lr_file_path):
            lr_img = data_loader.load_single_image(lr_file_path, size=None)
            generated_hr = model.generator.predict(lr_img)
            generated_hr = 0.5 * generated_hr + 0.5
            return Image.fromarray((np.uint8(generated_hr*255)[0])) 

        save_img_dir = create_img_dir(trained_steps)
        file_names = os.listdir(lr_dir_path)
        
        for i, file_name in enumerate(file_names[:5]):  # here: 5 prediction examples
            lr_file_path = f"{lr_dir_path}/{file_name}"
            img = resolve_single_image(lr_file_path)
            img.save(f"{save_img_dir}/{file_name}")


# **************************************
#
# **************************************
class SRGANModel(tf.Module):  # regarding parameter: https://stackoverflow.com/a/60509193
    def __init__(self, hr_shape, lr_shape, channels):
        self.channels = channels
        self.hr_shape = hr_shape
        self.lr_shape = lr_shape

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.vgg = self.build_vgg()


    def build_vgg(self):
        vgg = VGG19(
            weights="imagenet", 
            input_shape=self.hr_shape,
            include_top=False
        )

        output_layer = 20
        model = Model(inputs=vgg.input, outputs=vgg.layers[output_layer].output)
        model.trainable = False

        return model


    def build_generator(self, num_filters=64, num_res_blocks=16):
        def upsample(x_in):
            x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x_in)
            x = UpSampling2D(size=2)(x)  # instead of pixel shuffle as seen in some examples
            x = PReLU(shared_axes=[1, 2])(x)  # instead of Activation('relu') layer as seen in some examples
            return x
            

        def residual_block(x_in, filters):
            x = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x_in)
            x = PReLU(shared_axes=[1, 2])(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Add()([x, x_in])
            return x

        x_in = Input(shape=self.lr_shape)
        
        # add preresidual block
        x = Conv2D(64, kernel_size=9, strides=1, padding='same')(x_in)
        x = x_1 = PReLU(shared_axes=[1, 2])(x)

        # add residual blocks
        for _ in range(num_res_blocks):
            x = residual_block(x, num_filters)

        # add postresidual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, x_1])

        x = upsample(x)
        x = upsample(x)

        x_out = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(x)

        return Model(x_in, x_out)


    def build_discriminator(self, num_filters=64):
        def discriminator_block(x_in, filters, strides=1, batch_normalization=True):
            x = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(x_in)
            if batch_normalization:
                x = BatchNormalization(momentum=0.8)(x)
            return LeakyReLU(alpha=0.2)(x)

        x_in = Input(shape=self.hr_shape)

        x = discriminator_block(x_in, num_filters, batch_normalization=False)
        x = discriminator_block(x, num_filters, strides=2)
        
        x = discriminator_block(x, num_filters*2)
        x = discriminator_block(x, num_filters*2, strides=2)
        
        x = discriminator_block(x, num_filters*4)
        x = discriminator_block(x, num_filters*4, strides=2)
        
        x = discriminator_block(x, num_filters*8)
        x = discriminator_block(x, num_filters*8, strides=2)

        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(x_in, x)


# **************************************
#
# **************************************
class Pretrainer():
    def __init__(self):
        # ***
        # Input shape
        # ***
        self.channels = 3

        self.hr_height = self.hr_width = HR_CROPPED_IMG_SIZE  # assuming same dimensions for height and width
        self.lr_height = self.lr_width = HR_CROPPED_IMG_SIZE//SCALE_FACTOR
        
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # ***
        # data loader and utils
        # ***
        self.data_loader = DataLoader(HR_CROPPED_IMG_SIZE, SCALE_FACTOR)
        self.utils = Utils()

        # ***
        # model, loss and optimizer
        # ***
        self.mse_loss = MeanSquaredError()
        
        # ***
        # save training in checkpoint
        # necessary to keep all values (i.e. optimizer) when interrupting & resuming training
        # ***
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            optimizer=Adam(learning_rate=LEARNING_RATE),
            model=SRGANModel(self.hr_shape, self.lr_shape, self.channels)
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=PRETRAINED_GENERATOR_CHECKPOINT_DIR,
            max_to_keep=1
        )

        self.restore_checkpoint()

    
    # ***
    # get the latest checkpoint (if existing)
    # ***
    def restore_checkpoint(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restore checkpoint at step {self.checkpoint.step.numpy()}.")


    # ***************
    # train model (generator only)
    # ***************
    @tf.function
    def pretrain_step(self, lr_img, hr_img):
        with tf.GradientTape() as tape:
            hr_generated = self.checkpoint.model.generator(lr_img, training=True)
            mse_loss = self.mse_loss(hr_img, hr_generated)

        gradients = tape.gradient(mse_loss, self.checkpoint.model.generator.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.generator.trainable_variables))

        return mse_loss


    def pretrain(self, epochs=10, batch_size=1, sample_interval=2):
        start_time = datetime.datetime.now()  # for controle dump only

        for epoch in range(epochs):
            self.checkpoint.step.assign_add(batch_size)  # update steps in checkpoint
            trained_steps = self.checkpoint.step.numpy() # overall trained steps: for controle dump only
            
            # ***
            # train on batch
            # ***
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=batch_size)
            mse_loss = self.pretrain_step(imgs_lr, imgs_hr)

            elapsed_time = datetime.datetime.now() - start_time  # for controle dump/log only

            # ***
            # save and/or dump
            # ***
            if (epoch + 1) % 10 == 0:
                print(f"{epoch + 1} | steps: {trained_steps} | loss: {mse_loss} | time: {elapsed_time}")
            if (epoch + 1) % sample_interval == 0:
                print("   |---> save and make image sample")
                self.checkpoint_manager.save()  # save checkpoint

                # save weights for usage with class Trainer on first training iteration
                self.checkpoint.model.generator.save_weights(PRETRAINED_GENERATOR_WEIGHTS)

                # controle dump of predicted images: save in dirs named by trained_steps
                self.utils.test_predict(self.checkpoint.model, self.data_loader, IMG_DIR_VAL_LR, trained_steps, "pre_train")


# **************************************
#
# **************************************
class Trainer():
    def __init__(self, use_pretrain_weights=False):
        # ***
        # Input shape
        # ***
        self.channels = 3

        self.hr_height = self.hr_width = HR_CROPPED_IMG_SIZE  # assuming same dimensions for height and width
        self.lr_height = self.lr_width = HR_CROPPED_IMG_SIZE//SCALE_FACTOR
        
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # ***
        # data loader and utils
        # ***
        self.data_loader = DataLoader(HR_CROPPED_IMG_SIZE, SCALE_FACTOR)
        self.utils = Utils()

        # ***
        # loss
        # ***
        self.loss_binary_crossentropy = BinaryCrossentropy(from_logits=False)
        self.loss_mse = MeanSquaredError()

        # ***
        # save training in checkpoint
        # necessary to keep all values (i.e. optimizer) when interrupting & resuming training
        # ***
        # using a scheduler with the optimizer might be a good idea (not used right now)
        # learning_rate = PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])
        # ***

        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            optimizer_generator=Adam(learning_rate=LEARNING_RATE),
            optimizer_discriminator=Adam(learning_rate=LEARNING_RATE),
            model=SRGANModel(self.hr_shape, self.lr_shape, self.channels)
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=FINE_TUNE_CHECKPOINT_DIR,
            max_to_keep=1
        )

        # ***
        # either: start training with pretrained generator weights a) if parameter is True and b) if weights are available
        # or: get latest checkpoint of training  a) if parameter is False and b) if checkpoint is available
        # ***
        if use_pretrain_weights is False:
            self.restore_checkpoint()
        else:
            try:
                self.checkpoint.model.generator.load_weights(PRETRAINED_GENERATOR_WEIGHTS)
                print(f"Load pretrained generator weights.")
            except:
               print("No pre-trained weights available.")
               pass

    
    # ***
    # get the latest checkpoint (if existing)
    # ***
    def restore_checkpoint(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restore checkpoint at step {self.checkpoint.step.numpy()}.")

    # ***
    # loss functions, used in training step
    # ***
    @tf.function
    def content_loss(self, hr_img, hr_generated):
        def preprocess_vgg(x):
            '''
            Input img RGB [-1, 1] -> BGR [0,255] plus subtracting mean BGR values: (103.939, 116.779, 123.68)
            https://stackoverflow.com/a/46623958
            '''
            if isinstance(x, np.ndarray):
                return preprocess_input((x+1)*127.5)
            else:            
                return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5), autocast=False)(x)  # autocast: tf 2.something / float32 vs. float64 with Lambda
        
        hr_generated = preprocess_vgg(hr_generated)
        hr_img = preprocess_vgg(hr_img)

        # ***
        # normalize feature values by divide / 12.75 (Usually a good idea, but is it here? Well... I guess?)
        # Compare with: http://krasserm.github.io/2019/09/04/super-resolution/ (train.py)
        # ***
        hr_generated_features = self.checkpoint.model.vgg(hr_generated)/12.75 
        hr_features = self.checkpoint.model.vgg(hr_img)/12.75 

        return self.loss_mse(hr_features, hr_generated_features)


    def generator_loss(self, hr_generated_output):
        return self.loss_binary_crossentropy(tf.ones_like(hr_generated_output), hr_generated_output)


    def discriminator_loss(self, hr_output, hr_generated_output):
        hr_loss = self.loss_binary_crossentropy(tf.ones_like(hr_output), hr_output)
        hr_generated_loss = self.loss_binary_crossentropy(tf.zeros_like(hr_generated_output), hr_generated_output)
        
        return hr_loss + hr_generated_loss

    # ***
    #
    # ***
    @tf.function
    def train_step(self, lr_img, hr_img):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            hr_generated = self.checkpoint.model.generator(lr_img, training=True)

            hr_output = self.checkpoint.model.discriminator(hr_img, training=True)
            hr_generated_output = self.checkpoint.model.discriminator(hr_generated, training=True)

            content_loss = self.content_loss(hr_img, hr_generated)
            generator_loss = self.generator_loss(hr_generated_output)
            
            # ***
            # see also: feature normalization in content_loss
            # ***
            perceptual_loss = content_loss + 0.001 * generator_loss  # i.e. 0.136050597 + (0.001*12.2107553 ->) 0.0122107556
            
            discriminator_loss = self.discriminator_loss(hr_output, hr_generated_output)
            

        gradients_generator = gen_tape.gradient(perceptual_loss, self.checkpoint.model.generator.trainable_variables)
        gradients_discriminator = disc_tape.gradient(discriminator_loss, self.checkpoint.model.discriminator.trainable_variables)

        self.checkpoint.optimizer_generator.apply_gradients(zip(gradients_generator, self.checkpoint.model.generator.trainable_variables))
        self.checkpoint.optimizer_discriminator.apply_gradients(zip(gradients_discriminator, self.checkpoint.model.discriminator.trainable_variables))

        return perceptual_loss, discriminator_loss

    
    # ***
    #
    # ***
    def train(self, epochs=10, batch_size=1, sample_interval=2):
        start_time = datetime.datetime.now()  # for controle dump only

        for epoch in range(epochs):
            self.checkpoint.step.assign_add(batch_size)  # update steps in checkpoint
            trained_steps = self.checkpoint.step.numpy() # overall trained steps: for controle dump only
            
            # ***
            # train on batch
            # ***
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=batch_size)
            perceptual_loss, discriminator_loss = self.train_step(imgs_lr, imgs_hr)

            elapsed_time = datetime.datetime.now() - start_time  # for controle dump only

            # ***
            # save and/or dump
            # ***
            if (epoch + 1) % 10 == 0:
                print(f"{epoch + 1} | steps: {trained_steps} | g_loss: {perceptual_loss} | d_loss: {discriminator_loss} | time: {elapsed_time}")
            if (epoch + 1) % sample_interval == 0:
                print("   |---> save and make image sample")
                self.checkpoint_manager.save()  # save checkpoint
                self.checkpoint.model.generator.save(GENERATOR_MODEL)  # save complete model for actual usage

                # controle dump of predicted images: save in dirs named by trained_steps
                self.utils.test_predict(self.checkpoint.model, self.data_loader, IMG_DIR_VAL_LR, trained_steps, "fine_tune")
        


if __name__ == '__main__':
    '''
    Note: 
    Technically, what I call epochs in the training function calls are not epochs since this would mean to use all 
    stored image crops from the training set, when in fact I call for a newly augmented batch of randomly 
    selected image crops in every training round. 
    But for now, I'll just leave it like that.
    '''
    # ***
    # 1. pre-train generator (only)
    # ***
    # pretrainer = Pretrainer()
    # pretrainer.pretrain(epochs=100000, batch_size=4, sample_interval=20000)

    # ***
    # 2. train generator and discriminator
    # ***
    trainer = Trainer(use_pretrain_weights=True)  # use this parameter on very first training run (default: False)
    # trainer = Trainer()  # use this if you continue training (i.e. after interruption)
    trainer.train(epochs=50000, batch_size=4, sample_interval=10000)
