"""
Create the data pipeline to import the model inputs.
Code inspired by Stanford's CS230 project code examples. 
"""

import tensorflow as tf

def import_image(filename, label):
    '''
    Import the image using the filename
    Args :
        - filename: string
        - label: label of the image[0-499]
    Returns :
        - video: decoded video from the jpeg format
        - label: label of the image [0-499]
    '''
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    # Convert to range [0,1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    video = tf.reshape(image, shape=(64, 64, 29))
    return video, label

def input_fn(is_training, filenames, labels, batch_size):
    '''
    Input function
    Files names have format "{label}_{word}_{id}.jpg"
    Args :
        - is_training: whether to use the train or evaluation pipeline
        - filenames: list of the filenames
        - labels: corresponding list of labels
        - batch_size: size of the batch
    '''
    num_samples = len(filenames)
    import_fn = lambda f, l: import_image(f, l)
    
    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .shuffle(num_samples)
                   .map(import_fn, num_parallel_calls=1)
                   .batch(batch_size)
                   .prefetch(1)
                  )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .map(parse_fn)
                   .batch(num_samples)
                  )
    return dataset.make_one_shot_iterator().get_next() 