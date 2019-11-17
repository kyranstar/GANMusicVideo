import data_loader
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import latent_explorer
import math
import frames_to_video
import argparse

def create_music_video(start_image, music_features, mf_to_if_model, dataset):
    generator = tf.keras.models.load_model('models/generator.h5')
    inverter = tf.keras.models.load_model('models/inverter.h5')
    start_latent = inverter(tf.reshape(start_image, (1, 128, 128, 3)))
    latent_explorer.process_and_save_images(start_latent, mf, mf_to_if_model, inverter, generator, dataset.take(2000), [5, 6, -1], 128)


def generate_features(img, lab):
    return ave_brightness(img)


# Converts features of music over time to features of images over time
def create_mf_to_if_model(strength):
    def mf_to_if_model(mf):
        im_feats = {5: strength*np.maximum(0, mf), 6: strength*np.minimum(0, mf), -1: strength*mf}
        return im_feats
    return mf_to_if_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-strength", type=float, default=5)
    args = parser.parse_args()
    mf = np.sin(np.linspace(-np.pi, np.pi, 30*5))
    mf_to_if_model = create_mf_to_if_model(args.strength)
    dataset = data_loader.load_data()
    start_image, lab = dataset.take(1).make_one_shot_iterator().get_next()
    print(lab)
    print(lab.shape)
    create_music_video(start_image, mf, mf_to_if_model, dataset)
    frames_to_video.create_video_from_frames()
