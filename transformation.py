import tensorflow as tf
import random


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    # This is the method that the article uses to normalize the dataset
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


@tf.function
def rand_flip_image_horizontally(input_image: tf.Tensor, input_mask: tf.Tensor):
    """Randomly flip an image and label horizontally"""
    uniform_random = tf.random.uniform([], 0, 1.0)  # select a rand number
    flip_cond = tf.less(uniform_random, .5)  # if number is less than .5, then we will preform a flip
    image = tf.cond(flip_cond, lambda: tf.image.flip_left_right(input_image), lambda: input_image)
    mask = tf.cond(flip_cond, lambda: tf.image.flip_left_right(input_mask), lambda: input_mask)
    return image, mask


@tf.function
def rand_flip_image_vertically(input_image: tf.Tensor, input_mask: tf.Tensor):
    """Randomly flip an image and label vertically (upside down)."""
    uniform_random = tf.random.uniform([], 0, 1.0)  # select a rand number

    flip_cond = tf.less(uniform_random, .5)  # if number is less than .5, then we will preform a flip
    image = tf.cond(flip_cond, lambda: tf.image.flip_up_down(input_image), lambda: input_image)
    mask = tf.cond(flip_cond, lambda: tf.image.flip_up_down(input_mask), lambda: input_mask)
    return image, mask


@tf.function
def rand_rotate_image(input_image: tf.Tensor, input_mask: tf.Tensor):
    uniform_random = tf.random.uniform([], 0, 1.0)  # select a rand number
    flip_cond = tf.less(uniform_random, .5)  # if number is less than .5, then we will preform a rotation
    # if a rotation occurs, it will either be by 90, 180, or 270
    rotation = random.randint(1, 3)
    image = tf.cond(flip_cond, lambda: tf.image.rot90(input_image, k=rotation), lambda: input_image)
    mask = tf.cond(flip_cond, lambda: tf.image.rot90(input_mask, k=rotation), lambda: input_mask)
    return image, mask
