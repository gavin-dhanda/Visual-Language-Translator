import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from segmentation import segment_image
from segmentation import test_fn
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,3,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(26,activation='softmax')
    ])

def main():
    base_dir = '/Users/karim/Desktop/s24/cs1430/google-tranSLAYte/image_processing/data/Latin/'
    train_datagen = ImageDataGenerator()

    # Set up generators
    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=(28, 28),
        batch_size=10,
        class_mode='categorical',
        subset='training'
    )

    print(train_generator.class_indices)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=10)

main()
letter_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

letters, boxes = test_fn('/Users/karim/Desktop/s24/cs1430/google-tranSLAYte/image_processing/images/nntest.png', False)
output_string = ""
for letter in letters:
    if letter.ndim == 2:  # Only if it's grayscale
        letter = np.stack((letter,) * 3, axis=-1)
    in_img = letter[np.newaxis, ...]
    predictions = model.predict(in_img)
    print("Prediction: ", letter_dict[np.argmax(predictions, axis=1)[0]])
    output_string += letter_dict[np.argmax(predictions, axis=1)[0]]

print("Predicted input string: ", output_string)



