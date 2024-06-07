import os
from turtle import pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (64, 64) 
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 5

# Data preprocessing and augmentation using LeNet-5 approach
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    brightness_range=[0.5, 1.5],  # Adjust brightness
    channel_shift_range=50.0,  # Adjust channel intensity
    vertical_flip=True,  # Flip vertically
    featurewise_center=False,
    featurewise_std_normalization=False,
    zca_whitening=False
)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/monasser/Desktop/nural_project/dataset/train',
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'C:/Users/monasser/Desktop/nural_project/dataset/test',
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    batch_size=1,
    shuffle=False,
    class_mode=None
)

# Data preparation for LeNet-5
def create_train_data():
    original_count = 0
    augmented_count = 0
    training_data = []
    for folders in tqdm(os.listdir('C:/Users/monasser/Desktop/nural_project/dataset/train')):
        num_of_folder = 'C:/Users/monasser/Desktop/nural_project/dataset/train' + "/" + str(folders)
        for img in tqdm(os.listdir(num_of_folder)):
            original_count += 1  # Increment original count for each image
            path = os.path.join(num_of_folder, img)
            img_data = cv2.imread(path, 0)
            img_data = cv2.resize(img_data, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
            img_data = img_data.reshape((1,) + img_data.shape + (1,))  # Reshape for augmentation

            # Generate augmented images
            i = 0
            for batch in train_datagen.flow(img_data, batch_size=1):
                augmented_count += 1  # Increment augmented count for each augmentation
                image = batch[0].reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
                label = np.zeros(NUM_CLASSES)  # Create an array of zeros
                label[int(folders) - 1] = 1  # Set the appropriate index to 1
                training_data.append([np.array(image), label])
                i += 1
                if i >= 5:  # Number of augmented images per original image
                    break

    shuffle(training_data)
    print("Original data count:", original_count)
    print("Augmented data count:", augmented_count)
    return training_data

import joblib # save model with joblib
from keras.models import load_model
import h5py


if (os.path.exists('train_data.npy')): # If you have already created the dataset:
    train_data =joblib.load('train_data.npy')
else: # If dataset is not created:
    train_data = create_train_data()
    joblib.dump(train_data, "train_data.npy")


# Splitting data into train and test sets
train, test = train_test_split(train_data, test_size=0.15, random_state=42)

X_train = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
Y_train = np.array([i[1] for i in train])

X_test = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
Y_test = np.array([i[1] for i in test])


from keras.layers import MultiHeadAttention, LayerNormalization,GlobalAveragePooling2D, Dropout, Embedding, Input, GlobalAveragePooling1D, Dense
import warnings
from keras.models import Model

vocab_size = 50000
maxlen = 20
embed_dim = 32
num_heads = 2
ff_dim = 32


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Transformer Block
def transformer_block(inputs, embed_dim, num_heads, ff_dim, rate=0.1):
    att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    att = Dropout(rate)(att)
    att = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + att)

    ffn_output = tf.keras.Sequential([
        Dense(ff_dim, activation="relu"),
        Dense(embed_dim),
    ])(att)
    ffn_output = Dropout(rate)(ffn_output)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(att + ffn_output)

# Token and Position Embedding
def token_and_position_embedding(x, maxlen, vocab_size, embed_dim):
    positions = tf.range(start=0, limit=maxlen, delta=1)
    positions = Embedding(input_dim=maxlen, output_dim=embed_dim)(positions)
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(x)
    return x + positions

inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
x = token_and_position_embedding(inputs, maxlen, vocab_size, embed_dim)
x = transformer_block(x, embed_dim, num_heads, ff_dim)
# x = GlobalAveragePooling2D()(x)  # Use GlobalAveragePooling2D for image data
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)


model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, Y_train,
                    batch_size=32, epochs=10,
                    validation_data=(X_test, Y_test))







# importing required module
import csv

# opening the file
with open("model.csv", "w", newline="") as f:
    # creating the writer
    writer = csv.writer(f)
    # using writerow to write individual record one by one
    # writer.writerow(["Image", "Label", "Index"])
    writer.writerow(["image_id", "label"])
    # Make predictions
    label = ["apple", "banana", "grapes", "mango", "stra"]
    for image in tqdm(os.listdir('C:/Users/monasser/Desktop/nural_project/dataset/test')):
        img = cv2.imread(os.path.join('C:/Users/monasser/Desktop/nural_project/dataset/test', image), 0)
        img_test = cv2.resize(img, (IMAGE_SIZE))
        img_test = img_test.reshape(1, IMAGE_SIZE[0],IMAGE_SIZE[1], 1)  # Reshape for model input
        prediction = model.predict(img_test)[0]
        max_index = np.argmax(prediction)
        print(image.split('.')[0], "  ", label[max_index])
        # writer.writerow([image.split('.')[0],  label[max_index], max_index+1])
        writer.writerow([image.split('.')[0], max_index+1])


