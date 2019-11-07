# -*- coding: utf-8 -*-

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
import data_loader

"""### Import TensorFlow and enable eager execution"""

print(tf.__version__)

"""### Start Tensorboard Logging
"""
train_log_dir = 'summaries/mnist_gan/train'
test_log_dir = 'summaries/mnist_gan/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

"""### Load the dataset
We are going to use the MNIST dataset to train the generator and the discriminator. The generator will generate handwritten digits resembling the MNIST data.
"""


# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
(train_images, train_labels) = data_loader.load_data()
# print(train_images[0])


BUFFER_SIZE = 50000
BATCH_SIZE = 512
Z_DIM = 64
LATENT_DIM = 64
DIVERGENCE_LAMBDA = 0.1
GRAD_PENALTY_FACTOR = 10.0

"""### Use tf.data to create batches and shuffle the dataset"""

num_test_images = 8
test_dataset = train_images[:num_test_images]
train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images[num_test_images:]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

"""## Create the models
We will use tf.keras [Sequential API](https://www.tensorflow.org/guide/keras#sequential_model) to define the generator and discriminator models.
### The Generator Model
The generator is responsible for creating convincing images that are good enough to fool the discriminator. The network architecture for the generator consists of [Conv2DTranspose](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose) (Upsampling) layers. We start with a fully connected layer and upsample the image two times in order to reach the desired image size of 28x28x1. We increase the width and height, and reduce the depth as we move through the layers in the network. We use [Leaky ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LeakyReLU) activation for each layer except for the last one where we use a tanh activation.
"""


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(LATENT_DIM * 7*7*4, use_bias=False, input_shape=(Z_DIM,), name='Generator_Input'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Reshape((7, 7, LATENT_DIM*4)))
    print("Model shape should be (None, 7, 7, 256) -", model.output_shape)
    # Note: None is the batch size
    assert model.output_shape == (None, 7, 7, 256)

    model.add(tf.keras.layers.Conv2DTranspose(
        LATENT_DIM * 2, (5, 5), strides=(2, 2), padding='same', use_bias=False, name='Generator_1'))
    print("Model shape should be (None, 14, 14, 128) -", model.output_shape)
    assert model.output_shape == (None, 14, 14, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    #model.add(tf.keras.layers.Conv2DTranspose(
    #    64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2),
        padding='same', use_bias=False, activation='tanh', name='Generator_2'))
    print("Model shape should be (None, 28, 28, 1) -", model.output_shape)
    assert model.output_shape == (None, 28, 28, 1)
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.LeakyReLU())

    return model

"""
def kACGANGenerator(n_samples, numClasses=0, labels, noise=None, dim=DIM, bn=True, nonlinearity=tf.nn.relu, condition=None):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    labels = tf.cast(labels, tf.float32)
    noise = tf.concat([noise, labels], 1)

    #model.add(tf.keras.layers.Dense(8*4*4*LATENT_DIM*2, use_bias=False, input_shape=(Z_DIM,), name='Generator_Input'))
    output = lib.ops.linear.Linear('Generator.Input', 128+numClasses, 8*4*4*dim*2, noise) #probs need to recalculate dimensions

    #model.add(tf.keras.layers.Reshape((-1, 8*LATENT_DIM*2, 4, 4)))
    output = tf.reshape(output, [-1, 8*dim*2, 4, 4])

    #model.add(tf.keras.layers.BatchNormalization())
    if bn:
        output = Batchnorm('Generator.BN1', [0,2,3], output)
        
    condition = lib.ops.linear.Linear('Generator.cond1', numClasses, 8*4*4*dim*2, labels,biases=False)
    condition = tf.reshape(condition, [-1, 8*dim*2, 4, 4])
    output = pixcnn_gated_nonlinearity('Generator.nl1', 8*dim, output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])


    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond2', numClasses, 4*8*8*dim*2, labels)
    condition = tf.reshape(condition, [-1, 4*dim*2, 8, 8])
    output = pixcnn_gated_nonlinearity('Generator.nl2', 4*dim,output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond3', numClasses, 2*16*16*dim*2, labels)
    condition = tf.reshape(condition, [-1, 2*dim*2, 16, 16])
    output = pixcnn_gated_nonlinearity('Generator.nl3', 2*dim,output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond4', numClasses, 32*32*dim*2, labels)
    condition = tf.reshape(condition, [-1, dim*2, 32, 32])
    output = pixcnn_gated_nonlinearity('Generator.nl4', dim, output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)

    output = tf.tanh(output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, OUTPUT_DIM]), labels
"""

"""### The Discriminator model
The discriminator is responsible for distinguishing fake images from real images. It's similar to a regular CNN-based image classifier.
"""

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        LATENT_DIM, (5, 5), strides=(2, 2), padding='same', name='Discriminator_Input'))
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(
        LATENT_DIM*2, (5, 5), strides=(2, 2), padding='same', name='Discriminator_1'))
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(
        LATENT_DIM*4, (5, 5), strides=(2, 2), padding='same', name='Discriminator_2'))
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, name='Discriminator_3'))

    return model

"""
def kACGANDiscriminator(inputs, numClasses, dim=DIM, bn=True, nonlinearity=LeakyReLU):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = nonlinearity(output)


    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = nonlinearity(output)
    finalLayer = tf.reshape(output, [-1, 4*4*8*dim])

    sourceOutput = lib.ops.linear.Linear('Discriminator.sourceOutput', 4*4*8*dim, 1, finalLayer)

    classOutput = lib.ops.linear.Linear('Discriminator.classOutput', 4*4*8*dim, numClasses, finalLayer)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()



    return (tf.reshape(sourceOutput, [-1]), tf.reshape(classOutput, [-1, numClasses]))
"""

def make_inverter_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        LATENT_DIM, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1), name='Inverter_Input'))
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(
        LATENT_DIM*2, (5, 5), strides=(2, 2), padding='same', name='Inverter_1'))
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(
        LATENT_DIM*4, (5, 5), strides=(2, 2), padding='same', name='Inverter_2'))
    model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(LATENT_DIM * 8, name='Inverter_3'))
    model.add(tf.keras.layers.Dense(Z_DIM, name='Inverter_4'))

    return model

generator = make_generator_model()
discriminator = make_discriminator_model()
inverter = make_inverter_model()

"""## Define the loss functions and the optimizer
Let's define the loss functions and the optimizers for the generator and the discriminator.
### Generator loss
The generator loss is a sigmoid cross entropy loss of the generated images and an array of ones, since the generator is trying to generate fake images that resemble the real images.
"""

def generator_loss(generated_output, gradient_penalty):
    return tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output) + gradient_penalty

"""### Discriminator loss
The discriminator loss function takes two inputs: real images, and generated images. Here is how to calculate the discriminator loss:
1. Calculate real_loss which is a sigmoid cross entropy loss of the real images and an array of ones (since these are the real images).
2. Calculate generated_loss which is a sigmoid cross entropy loss of the generated images and an array of zeros (since these are the fake images).
3. Calculate the total_loss as the sum of real_loss and generated_loss.
"""

def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

"""### Inverter Loss
The inverter loss takes four inputs: noise z, invert(generate(z)), image x, and generate(invert(x)).
"""
def inverter_loss(real_noise, rec_noise, real_image, rec_image):
    divergence = DIVERGENCE_LAMBDA * tf.reduce_mean(tf.square(real_noise - rec_noise))
    reconstruction_err = tf.reduce_mean(tf.square(real_image - rec_image))

    return divergence + reconstruction_err

"""The discriminator and the generator optimizers are different since we will train two networks separately."""

generator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
inverter_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

"""**Checkpoints (Object-based saving)**"""

checkpoint_dir = './training_checkpoints_mnist'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

"""## Set up GANs for Training
Now it's time to put together the generator and discriminator to set up the Generative Adversarial Networks, as you see in the diagam at the beginning of the tutorial.
**Define training parameters**
"""

EPOCHS = 200
num_examples_to_generate = 16

# We'll re-use this random vector used to seed the generator so
# it will be easier to see the improvement over time.
random_vector_for_generation = tf.random.normal([num_examples_to_generate,
                                                 Z_DIM])

"""**Define training method**
We start by iterating over the dataset. The generator is given a random vector as an input which is processed to  output an image looking like a handwritten digit. The discriminator is then shown the real MNIST images as well as the generated images.
Next, we calculate the generator and the discriminator loss. Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables.
"""


def train_step(images, epoch):
    #print("Train step %d" % epoch)
    if images.shape[0] != BATCH_SIZE:
        print("training images were not full batch, was " + str(images.shape[0]))
        return
   # generating noise from a normal distribution
    noise = tf.random.normal([BATCH_SIZE, Z_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as gp_tape, tf.GradientTape() as inv_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        rec_image = generator(inverter(images))
        rec_noise = inverter(generator(noise))

        # Calculate WGAN gradient penalty
        alpha = tf.random.uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.)
        x_p = tf.reshape(generated_images, [-1, 28*28])
        x = tf.reshape(images, [-1, 28*28])
        difference = x_p - x
        interpolate = tf.reshape(x + alpha * difference, [-1, 28, 28, 1])
        gradient = gp_tape.gradient(discriminator(interpolate), [interpolate])[0]
        slope = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=1))
        gradient_penalty = GRAD_PENALTY_FACTOR * tf.reduce_mean((slope - 1.) ** 2)

        gen_loss = generator_loss(generated_output, gradient_penalty)
        disc_loss = discriminator_loss(real_output, generated_output)
        inv_loss = inverter_loss(noise, rec_noise, images, rec_image)

        with train_summary_writer.as_default():
            tf.summary.scalar('gen_loss', gen_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)
            tf.summary.scalar('inv_loss', inv_loss, step=epoch)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.variables)
    gradients_of_inverter = inv_tape.gradient(inv_loss, inverter.variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.variables))
    inverter_optimizer.apply_gradients(
        zip(gradients_of_inverter, inverter.variables))


"""This model takes about ~30 seconds per epoch to train on a single Tesla K80 on Colab, as of October 2018.
Eager execution can be slower than executing the equivalent graph as it can't benefit from whole-program optimizations on the graph, and also incurs overheads of interpreting Python code. By using [tf.contrib.eager.defun](https://www.tensorflow.org/api_docs/python/tf/contrib/eager/defun) to create graph functions, we get a ~20 secs/epoch performance boost (from ~50 secs/epoch down to ~30 secs/epoch). This way we get the best of both eager execution (easier for debugging) and graph mode (better performance).
"""


def train(dataset, epochs):
    generate_and_save_images(generator, 1, random_vector_for_generation)
    print("Training")
    for epoch in range(epochs):
        start = time.time()

        for current_step, images in tqdm(enumerate(dataset)):
            train_step(images, current_step)

        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 random_vector_for_generation)
        reconstruct_and_save_images(generator, inverter, epoch+1, test_dataset)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            generator.save('models/generator.h5')
            inverter.save('models/inverter.h5')

        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                         time.time()-start))
    # generating after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             random_vector_for_generation)
    reconstruct_and_save_images(generator, inverter, epochs, test_dataset)
    print("Saving models!")
    generator.save('models/generator.h5')
    inverter.save('models/inverter.h5')


"""**Generate and save images**"""


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

    plt.savefig("images/" + 'sample_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    plt.clf()

def reconstruct_and_save_images(generator, inverter, epoch, test_images):
    predictions = generator(inverter(test_images, training=False), training=False)

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, 2*i+1)
        predi = convert_array_to_image(test_images[i])
        plt.imshow(predi)
        plt.axis('off')
        plt.subplot(4, 4, 2*i+2)
        predi = convert_array_to_image(predictions[i])
        plt.imshow(predi)
        plt.axis('off')

    plt.savefig("images/" + 'reconstruction_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()
    plt.clf()

def convert_array_to_image(array):
    array = tf.reshape(array, [28, 28])
    """Converts a numpy array to a PIL Image and undoes any rescaling."""
    img = PIL.Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='L')
    return img

"""## Train the GANs
We will call the train() method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).
At the beginning of the training, the generated images look like random noise. As training progresses, you can see the generated digits look increasingly real. After 50 epochs, they look very much like the MNIST digits.
**Restore the latest checkpoint**
"""

if __name__ == "__main__":
    # restoring the latest checkpoint in checkpoint_dir
    print("Restoring from", tf.train.latest_checkpoint(checkpoint_dir))
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    generator.save('models/generator.h5')
    inverter.save('models/inverter.h5')
    # save the architecture string to a file somehow, the below will work
    with open('models/generator_arch.json', 'w') as arch_file:
        arch_file.write(generator.to_json())
    with open('models/inverter_arch.json', 'w') as arch_file:
        arch_file.write(inverter.to_json())

    print("Num epochs", EPOCHS)
    train(train_dataset, EPOCHS)
