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
from sklearn import svm


def get_image_unit_vectors(latent_points, image_features, generator, inverter):
    """
    Gets unit vectors in latent space corresponding to image features

    Parameters:
        X: an image dataset
        Y: a dictionary where keys are names of image features and values are
        arrays of those features corresponding to X
    Returns:
        A dictionary of image feature names to unit vectors
    """

    # Fit a SVM for each image feature
    svms = {}
    for (feature_name, feature_labels) in image_features.items():
        print("Creating svm for feature {}...".format(feature_name))
        X = latent_points
        Y = feature_labels
        clf = svm.LinearSVC()
        clf.fit(X, Y)
        svms[feature_name] = clf

    def generate_and_save_images(model, epoch, inputs):
        #
        # make sure the training parameter is set to False because we
        # don't want to train the batchnorm layer when doing inference.

        predictions = [model(test_input, training=False) for test_input in inputs]

        fig = plt.figure(figsize=(8, 8))

        for i in range(len(predictions)):
            plt.subplot(4, 4, i+1)
            predi = convert_array_to_image(predictions[i])
            plt.imshow(predi)
            plt.axis('off')

        plt.savefig("images/" + 'sample{:04d}.png'.format(epoch))
        plt.clf()

    def convert_array_to_image(array):
        array = tf.reshape(array, [28, 28])
        """Converts a numpy array to a PIL Image and undoes any rescaling."""
        img = PIL.Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='L')
        return img

    orth_vec =  svms['ave_brightness'].coef_
    feature_unit = normalize(orth_vec)

    # Generate walks along svm normal
    img_0 = latent_points[0]
    distances = np.arange(0.0, 16, 1.0)
    imgs = [(img_0 + d*feature_unit) for d in distances]

    generate_and_save_images(generator, 0, imgs)
    # Return unit vectors corresponding to image features
    return uncorrelate_norms({name: normalize(svm.coef_) for (name, svm) in svms.items()})

def normalize(v):
    return v / np.linalg.norm(v)

def uncorrelate_norms(norms):
    """
    For any two hyperplanes with normals n1 and n2, to find the projection direction of n1 uncorrelated with n2
    we just need to find n1 - (n1^Tn2)n2 (Conditional Manipulation in section 2.3 of https://arxiv.org/pdf/1907.10786.pdf)
    """
    for (feature_name, n1) in norms.items():
        other_norms = [n for (f, n) in norms.items() if f != feature_name]
        for n2 in other_norms:
            norms[feature_name] -= np.vdot(n1, n2) / np.vdot(n2, n2) * n2
    return {name: normalize(n) for (name, n) in norms.items()}
