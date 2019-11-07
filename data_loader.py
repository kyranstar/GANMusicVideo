import json
import os
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy as np

GENRE_OPTIONS = ['portrait', 'landscape', 'genre painting', 'abstract', 'religious painting', 'sketch and study', 'cityscape', 'figurative', 'illustration',
    'nude painting (nu)', 'still life', 'design', 'symbolic painting', 'sculpture', 'mythological painting', 'flower painting', 'self-portrait', 'animal painting', 'marina', 'installation']

YEAR_QUANTILES = (1879, 1915, 1963)
STYLE_OPTIONS = ['lyrical abstraction', 'concretism', 'ink and wash painting', 'fauvism', 'magic realism', 'neo-expressionism', 'op art', 'conceptual art', 'pop art',
    'contemporary realism', 'early renaissance', 'color field painting', 'academicism', 'high renaissance', 'art deco', 'minimalism', 'mannerism (late renaissance)',
    'neoclassicism', 'abstract art', 'art informel', 'cubism', 'ukiyo-e', 'na\xc3\xafve art (primitivism)', 'abstract expressionism', 'rococo', 'symbolism',
    'northern renaissance', 'art nouveau (modern)', 'romanticism', 'baroque', 'surrealism', 'post-impressionism', 'expressionism', 'realism', 'impressionism']

def read_img(file_path, img_size):
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Conver to [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Get [-1, 1] range
    img = img * 2.0 - 1.0
    return tf.image.resize(img, [img_size, img_size])

def get_features(json_dict):
    """
    """
    # divide years into quantiles
    feats = np.zeros(len(YEAR_QUANTILES) + 1 + len(GENRE_OPTIONS) + len(STYLE_OPTIONS))
    if not json_dict['style'] is None:
        styles = [s.lower().strip() for s in json_dict['style'].split(',')]
        for style in styles:
            if style in STYLE_OPTIONS:
                style_index = len(YEAR_QUANTILES) + 1 + len(GENRE_OPTIONS) + STYLE_OPTIONS.index(style)
                feats[style_index] = 1
            else:
                print("Style {} does not exist!".format(style))
    if not json_dict['genre'] is None:
        genres = [s.lower().strip() for s in json_dict['genre'].split(',')]
        for genre in genres:
            if genre in GENRE_OPTIONS:
                genre_index = len(YEAR_QUANTILES) + 1 + GENRE_OPTIONS.index(genre)
                feats[genre_index] = 1
            else:
                print("Genre {} does not exist!".format(genre))
    year = json_dict['completitionYear']
    if year <= YEAR_QUANTILES[0]:
        feats[0] = 1
    elif year <= YEAR_QUANTILES[1]:
        feats[1] = 1
    elif year <= YEAR_QUANTILES[2]:
        feats[2] = 1
    else:
        feats[3] = 1
    return feats


    # Interesting keys: tags, genre, style, period, completitionYear, title,

def load_data(image_folder='data/wikiart-saved/images/', meta_folder='data/wikiart-saved/meta/', img_size=128):
    """
    Returns:
        tf.data.Dataset: gives (image, label) pairs, where the images are preprocessed and sized to img_size x img_size,
            and the labels are binary numpy arrays where a 1 corresponds to a certain feature.
    """

    # Load json files from data/wikiart-saved/meta/artistname.json
    meta_files = [join(meta_folder, f) for f in listdir(meta_folder) if isfile(join(meta_folder, f))]
    meta_files = meta_files[:10]
    def datapoint_gen():
        for meta_file_dir in meta_files:
            print("Loading {}...".format(meta_file_dir))
            with open(meta_file_dir, encoding='utf-8') as meta_file:
                meta_json = json.load(meta_file)
                for meta_entry in meta_json:
                    # Get key year and contentId
                    artist_name = meta_entry.get('artistUrl')
                    year = meta_entry.get('completitionYear')
                    id = meta_entry.get('contentId')
                    if artist_name is None or year is None or id is None:
                        continue
                    # Load image from data/wikiart-saved/images/year/contentId.jpg
                    img_path = join(image_folder, artist_name, str(year), str(id) + '.jpg')
                    if not isfile(img_path):
                        print("Image {} does not exist.".format(img_path))
                        continue
                    try:
                        img = read_img(img_path, img_size)
                        feats = get_features(meta_entry)
                    except Exception as e:
                        print("EXCEPTION: " + str(e))
                        continue
                    yield img, feats

    #print("Preprocessing images...")
    #datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #    featurewise_center=True,
    #    featurewise_std_normalization=True)
    #datagen.fit(images)
    return tf.data.Dataset.from_generator(datapoint_gen,
                                        (tf.float32, tf.int32))

    # Preprocess images

    """
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    print(train_images.shape, train_labels.shape)

    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')
    #train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    train_images = train_images * (2.0 / 255) - 1.0
    return (train_images, train_labels)
    """
