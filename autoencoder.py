from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import numpy as np
from os import listdir
from random import randint
from skimage import io,transform
from sklearn.model_selection import train_test_split
from math import ceil

# random bleach stack picker
class ImageStackRandomizer(Sequence):
    def __init__(self, x_set, batch_size,augment=False):
        self.x_set = x_set
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, item):
        raw_x = self.x_set.take(range(item * self.batch_size, (item + 1) * self.batch_size),mode="wrap") # get file names
        #batch size can vary, but we only want 224x224 single channel images.
        batch_x = np.empty((self.batch_size,512,512,1))
        batch_y = np.empty((self.batch_size,512,512,1))

        for i,f in enumerate(raw_x):
            img = io.imread("../data/processed/"+f).astype('float32') / 255.
            top = np.copy(img[0])
            bleach_id = 10
            if self.augment:
                bleach_id = randint(1,20)
            bleached = np.copy(img[bleach_id]) # random bleach between 0-10 (easiest)
            img = None # hopefully this will aide the garbage collection and reduce memory.

            if self.augment:
                rotation = randint(-20,20) #rotation amount
                top = transform.rotate(top,rotation)
                bleached = transform.rotate(bleached,rotation)

                if randint(0,1) == 1: #should we flip it left to right?
                    top = np.fliplr(top)
                    bleached = np.fliplr(bleached)

            batch_x[i,] = bleached.reshape(512,512,1)
            batch_y[i,] = top.reshape((512,512,1))
        return batch_x, batch_y


#MODEL CREATION
def simpleUNet():
    input_img = keras.Input(shape=(512, 512, 1))

    con1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.BatchNormalization()(con1)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    con2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(con2)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    con3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(con3)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    con4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(con4)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(layers.Add()([x,con4]))
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(layers.Add()([x,con3]))
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(layers.Add()([x,con2]))
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), padding='same')(layers.Add()([x,con1]))

    model = keras.Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

def convBlock(input,filters):
    x = layers.Conv2D(filters, (3, 3), padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def upConvBlock(input,skip,filters):
    x = layers.UpSampling2D((2, 2))(input)
    x = layers.Conv2D(filters, (3, 3), padding='same')(layers.Concatenate()([x,skip]))
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters/2, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def advancedUNet():
    input_img = keras.Input(shape=(512, 512, 1))

    block1 = convBlock(input_img,32)
    x = layers.MaxPooling2D((2, 2), padding='same')(block1)
    block2 = convBlock(x,32)
    x = layers.MaxPooling2D((2, 2), padding='same')(block2)
    block3 = convBlock(x,32)
    x = layers.MaxPooling2D((2, 2), padding='same')(block3)
    block4 = convBlock(x,32)
    x = layers.MaxPooling2D((2, 2), padding='same')(block4)
    x = convBlock(x,32)
    x = upConvBlock(x,block4,32)
    x = upConvBlock(x,block3,32)
    x = upConvBlock(x,block2,32)
    x = upConvBlock(x,block1,32)
    decoded = layers.Conv2D(1, (3, 3), padding='same')(x)

    model = keras.Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

#TRAINING DATA SETUP

dataset = np.array(listdir("../data/processed")) # we use np.take
X_train,X_val = train_test_split(dataset,train_size=0.85,random_state=1)

batch_size = 10

trainingGen = ImageStackRandomizer(X_train, batch_size,augment=True)
validationGen = ImageStackRandomizer(X_val, batch_size)



#TRAINING

train_checkpoint = ModelCheckpoint("trained_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', save_freq='epoch')
val_checkpoint = ModelCheckpoint("validated_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', save_freq='epoch')

model = advancedUNet()
history = model.fit(trainingGen,
                epochs=100,
                batch_size=batch_size,
                shuffle=True,
                validation_data=validationGen,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),train_checkpoint,val_checkpoint])

def plot_data(history):
    plt.title('Loss')
    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='red')

    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()

plot_data(history)
