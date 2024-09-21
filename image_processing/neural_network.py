import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import sys

def fix_data(image):
    image = image.reshape(28, 28)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

# fetch first argument from command line
def find_predicted_letters(use_pretrained, letters):
    training_letters = pd.read_csv('image_processing/data/emnist-letters-train.csv')
    labels = np.array(training_letters.iloc[:,0].values) - 1
    images = np.array(training_letters.iloc[:,1:].values)
    images = np.apply_along_axis(fix_data, 1, images)
    images = images / 255.0
    images = images.reshape(-1, 28, 28, 1)

    number_of_classes = 26
    labels = tf.keras.utils.to_categorical(labels, number_of_classes)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPool2D(strides=2))
    model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(84, activation='relu'))
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    checkpoint_path = "training_1/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=False,
                                                    verbose=1)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if use_pretrained:
        model.load_weights(checkpoint_path)
    else:
        model.fit(images, labels, epochs=15, batch_size=32, validation_split=0.1, callbacks=[cp_callback])


    letter_dict = {i: chr(65 + i) for i in range(26)}
    output_string = ""

    for letter in letters:
        letter = cv2.resize(letter, (20, 20), interpolation= cv2.INTER_AREA)
        letter = np.bitwise_not(letter)
        letter = np.pad(letter, ((4, 4), (4, 4)), "constant", constant_values= 0)
        letter = np.expand_dims(letter, axis=-1) / 255.0
        predictions = model.predict(np.expand_dims(letter, axis=0), verbose=0)
        predicted_letter = letter_dict[np.argmax(predictions)]
        output_string += predicted_letter

    print("Predicted input string: ", output_string)
    return output_string
