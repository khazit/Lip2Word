"""
Create the data pipeline to import the model inputs.
Code inspired by Stanford's CS230 project code examples.
"""

import tensorflow as tf


def input_fn(is_training, filenames, labels, batch_size=None, num_epochs=1):
    """
    Input function
    Files names have format "{label}_{word}_{id}.jpg"
    Args :
        - is_training: whether to use the train or evaluation pipeline
        - num_epochs: number of epochs
        - filenames: list of the filenames
        - labels: corresponding list of labels
        - batch_size: size of the batch
    """
    num_samples = len(filenames)
    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .repeat(num_epochs)
            .shuffle(num_samples)
            .map(_import_fn, num_parallel_calls=8)
            .map(_preprocess_fn, num_parallel_calls=8)
            .batch(batch_size)
            .prefetch(1)
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(_import_fn, num_parallel_calls=8)
            .batch(32)
            .prefetch(1)
        )
    return dataset.make_one_shot_iterator().get_next()

def _import_fn(filename, label):
    """
    Import the image using the filename
    Args :
        - filename: string
        - label: label of the video [0-499]
    Returns :
        - video: decoded video from the jpeg format
        - label: label of the video [0-499]
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    video = [tf.slice(image, begin=[i*64, 0, 0], size=[64, 64, 1]) for i in range(29)]
    video = tf.concat(video, axis=2)
    return video, label

def _preprocess_fn(video, label) :
    """
    Data augmentation function
    Args :
        - video:
        - label: label of the video [0-499]
    Returns :
        - video: same as input but with random contrast and brightness applied
        - label: label of the video [0-499]
    """
    video = tf.image.random_contrast(
        image=video,
        lower=0.3,
        upper=0.7,
    )
    video = tf.image.random_brightness(
        imahe=video,
        max_delta=32.0 / 255.0
    )
    # Make sure the values are still in [0, 1]
    video = tf.clip_by_value(
        image=video,
        0.0,
        1.0
    )
    return video, label
