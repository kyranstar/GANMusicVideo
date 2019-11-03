import numpy as np
import os
import PIL
import time
from IPython import display
import imageio
import glob
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tqdm import tqdm
import if_to_latent_svm

def convert_array_to_image(array):
    array = tf.reshape(array, [28, 28])
    """Converts a numpy array to a PIL Image and undoes any rescaling."""
    img = PIL.Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='L')
    return img

def generate_and_save_images(model, epoch, test_input):
    #
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        predi = convert_array_to_image(predictions[i])
        plt.imshow(predi)
        plt.axis('off')

    plt.savefig("images/" + 'sample{:04d}.png'.format(epoch))
    plt.clf()

def process_and_save_images(start_latent, mf, mf_to_if_model, if_to_latent_model, if_to_image_model):
    """
    The main algorithm for the entire project. Takes in music features for a given song,
    generates image features using the given transformation, transforms this to
    a path in the GAN's latent space, and then transforms this into frames using
    the if_to_image_model.

    mf: Music feature samples. For 30 fps footage, have 30 samples per second.
    mf_to_if_model:
    if_to_latent_model:
    if_to_image_model:
    """
    image_features = mf_to_if_model(mf)
    latent_points = if_to_latent_model(image_features, mf.shape[0], start_latent)
    images = if_to_image_model(latent_points, training=False)

    for i in range(images.shape[0]):
        img = images[i, :]
        filename = "renders/" + 'reconstruction{}.jpg'.format(i)
        convert_array_to_image(img).save(filename, "JPEG")

Z_DIM = 64
generator_dir = 'models/generator.h5'
inverter_dir = 'models/inverter.h5'

generator = tf.keras.models.load_model(generator_dir)
inverter = tf.keras.models.load_model(inverter_dir)

# Load dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images[:500]
train_labels = train_labels[:500]
train_images = train_images.reshape(
    train_images.shape[0], 28, 28, 1).astype('float32')
#train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
train_images = train_images * (2.0 / 255) - 1.0
train_images = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(50000).batch(1)

# Calculate any necessary image features
image_features = {'number_label': train_labels}
def ave_brightness(img):
    return 0 if tf.reduce_mean(img) < -0.75 else 1
print('Calculating ave_brightness...')

image_features['ave_brightness'] = np.array([ave_brightness(img) for img in tqdm(train_images)])
print('Calculating latent representations...')
latent_points = np.array([inverter(img, training=False) for img in tqdm(train_images)]).reshape(-1, Z_DIM)
unit_vectors = if_to_latent_svm.get_image_unit_vectors(latent_points, image_features, generator, inverter)

mf = np.sin(np.linspace(-np.pi, np.pi, 30*5))
def mf_to_if_model(mf):
    im_feats = {'ave_brightness': mf*8}
    return im_feats

def if_to_latent_model(image_features, size, start_latent):
    """
    Converts image features to the latent space using the svm unit vectors
    if: {feature_name: np array of features}
    size: the number of resulting latent space points
    start_latent: the first latent point in the series
    """
    sum = np.zeros((size, Z_DIM))
    for (feature_name, feature_strength) in image_features.items():
        feat_vec = unit_vectors[feature_name]
        sum += np.multiply(feat_vec.transpose(), feature_strength).transpose()
    print(sum)
    #Generate walks along svm normal
    #feature_unit = unit_vectors['ave_brightness']
    #distances = np.arange(0.0, 16, 0.01)
    #imgs = [(img_0 + sum[i, :]) for i in range(size)]
    return np.add(sum, np.full((mf.shape[0], Z_DIM), fill_value=start_latent))

process_and_save_images(latent_points[0], mf, mf_to_if_model, if_to_latent_model, generator)
