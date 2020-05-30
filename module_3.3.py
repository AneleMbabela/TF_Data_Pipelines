import  matplotlib.pyplot as plt
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images.shape


def plot_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)

plot_image(train_images[0])
plot_image(train_images[1])

train_images = train_images / 255.0
plot_image(train_images[0])

test_images = test_images / 255.0
