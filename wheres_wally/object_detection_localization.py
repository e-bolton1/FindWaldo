# Object Detection and Localization using TensorFlow
# Converted from Jupyter Notebook to Python script for supercomputer batch execution

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import time

# Set up image directories (adjust as needed)
image_dir = os.getcwd() + '/images/finding_waldo'
background_dir = image_dir + '/wheres_wally.jpg'
waldo_dir = image_dir + '/waldo.png'
wilma_dir = image_dir + '/wilma.png'

def generate_sample_image():
    background_im = Image.open(background_dir)
    background_im = background_im.resize((500, 350))
    waldo_im = Image.open(waldo_dir).resize((60, 100))
    wilma_im = Image.open(wilma_dir).resize((60, 100))
    col = np.random.randint(0, 410)
    row = np.random.randint(0, 230)
    rand_person = np.random.choice([0, 1], p=[0.5, 0.5])
    if rand_person == 1:
        background_im.paste(waldo_im, (col, row), mask=waldo_im)
        cat = 'Waldo'
    else:
        background_im.paste(wilma_im, (col, row), mask=wilma_im)
        cat = 'Wilma'
    return np.array(background_im).astype('uint8'), (col, row), rand_person, cat

def plot_bounding_box(image, gt_coords, pred_coords=None):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle((gt_coords[0], gt_coords[1], gt_coords[0] + 60, gt_coords[1] + 100), outline='green', width=5)
    if pred_coords:
        draw.rectangle((pred_coords[0], pred_coords[1], pred_coords[0] + 60, pred_coords[1] + 100), outline='red', width=5)
    return image

def generate_data(batch_size=16):
    while True:
        x_batch = np.zeros((batch_size, 350, 500, 3))
        y_batch = np.zeros((batch_size, 1))
        boundary_box = np.zeros((batch_size, 2))
        for i in range(batch_size):
            sample_im, pos, person, _ = generate_sample_image()
            x_batch[i] = sample_im / 255
            y_batch[i] = person
            boundary_box[i, 0] = pos[0]
            boundary_box[i, 1] = pos[1]
        yield x_batch, {'class': y_batch, 'box': boundary_box}

def convolutional_block(inputs):
    x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 6, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 6, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    return x

def regression_block(x):
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(2, name='box')(x)
    return x

def classification_block(x):
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='class')(x)
    return x

def test_model(model):
    for i in range(3):
        sample_im, pos, _, cat = generate_sample_image()
        sample_image_normalized = sample_im.reshape(1, 350, 500, 3) / 255
        predicted_class, predicted_box = model.predict(sample_image_normalized)
        pred_label = 'Waldo' if predicted_class > 0.5 else 'Wilma'
        col = 'green' if (pred_label == cat) else 'red'
        im = plot_bounding_box(sample_im, pos, (predicted_box[0][0], predicted_box[0][1]))
        plt.imshow(im)
        plt.title(f'True: {cat} | Predicted: {pred_label}', color=col)
        plt.axis('off')
        plt.show()

def lr_schedule(epoch, lr):
    if (epoch + 1) % 5 == 0:
        lr *= 0.2
    return max(lr, 3e-7)

class VisCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:
            test_model(self.model)

def main():
    inputs = tf.keras.Input((350, 500, 3))
    x = convolutional_block(inputs)
    box_output = regression_block(x)
    class_output = classification_block(x)
    model = tf.keras.Model(inputs=inputs, outputs=[class_output, box_output])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss={'class': 'binary_crossentropy', 'box': 'mse'},
                  metrics={'class': 'accuracy', 'box': 'mse'})
    tick = time.time()
    hist = model.fit(
        generate_data(batch_size=8),
        epochs=10,
        steps_per_epoch=50,
        callbacks=[VisCallback(model), tf.keras.callbacks.LearningRateScheduler(lr_schedule)]
    )
    tock = time.time()
    print(f'Took {np.round((tock - tick)/60, 2)} minutes to finish training 10 epochs')
    test_model(model)
    for i in range(10):
        test_model(model)

if __name__ == "__main__":
    main()
