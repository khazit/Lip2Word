import tensorflow as tf

################################################################################
# General utility functions.
################################################################################

def batch_norm(inputs, is_training, layer_name, reuse=False) :
    '''
    Utility function. BatchNorm + ReLu
    Args :
        - inputs : a tensor of inputs
        - is_training : a bool
    Return :
        - Tensor after BatchNorm + ReLu. Same shape as inputs
    '''
    if reuse :
        inputs =  tf.layers.batch_normalization(
            inputs=inputs,
            reuse=tf.AUTO_REUSE,
            name=layer_name,
            training=is_training
        )
    else :
        inputs =  tf.layers.batch_normalization(
            inputs=inputs,
            name=layer_name,
            training=is_training
        )
    return tf.nn.relu(inputs)

################################################################################
# Inception modules.
# The layers are named as follows :
# [block type]_b[bifurcation number][branch number]?_[layer type][layer number]
################################################################################

def _single_frame_stem(input, is_training) :
    # Reshape to 4D tensor
    input = tf.reshape(
        input,
        [-1, 64, 64, 1]
    )

    # Convolutional Layer #1 :
    stem_conv1 = tf.layers.conv2d(
        inputs=input,
        filters=8,
        kernel_size=[3, 3],
        strides=2,
        padding="valid",
        reuse=tf.AUTO_REUSE,
        name="stem_conv1"
    )
    stem_conv1 = batch_norm(
        inputs=stem_conv1,
        is_training=is_training,
		reuse=True,
		layer_name="bn_stem_conv1")

    # Convolutional Layer #2 :
    stem_conv2 = tf.layers.conv2d(
        inputs=stem_conv1,
        filters=8,
        kernel_size=[3, 3],
        strides=1,
        padding="valid",
        reuse=tf.AUTO_REUSE,
        name="stem_conv2"
    )
    stem_conv2 = batch_norm(
        inputs=stem_conv2,
        is_training=is_training,
		reuse=True,
		layer_name="bn_stem_conv2")

    # Convolutional Layer #3 :
    stem_conv3 = tf.layers.conv2d(
        inputs=stem_conv2,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        reuse=tf.AUTO_REUSE,
        name="stem_conv3"
    )
    stem_conv3 = batch_norm(
        inputs=stem_conv3,
        is_training=is_training,
		reuse=True,
		layer_name="bn_stem_conv3")

    # Bifurcation #1
    # Branch #1
    # MaxPooling Layer #1
    stem_b11_pool1 = tf.layers.max_pooling2d(
        inputs=stem_conv3,
        pool_size=[3, 3],
        strides=2,
        name="stem_b11_pool1"
    )
    # Branch #2
    # Convolutional Layer #1
    stem_b12_conv1 = tf.layers.conv2d(
        inputs=stem_conv3,
        filters=24,
        kernel_size=[3, 3],
        strides=2,
        padding="valid",
        reuse=tf.AUTO_REUSE,
        name="stem_b12_conv1"
    )
    stem_b12_conv1 = batch_norm(
        inputs=stem_b12_conv1,
        is_training=is_training,
		reuse=True,
		layer_name="bn_stem_b12_conv1")
    # Junction #1
    stem_junction1 = tf.concat(
        values=[stem_b11_pool1, stem_b12_conv1],
        axis=3,
        name="stem_junction1"
    )

    # Bifurcation #2
    # Branch #1
    # Convolutional Layer #1
    stem_b21_conv1 = tf.layers.conv2d(
        inputs=stem_junction1,
        filters=16,
        kernel_size=[1, 3],
        padding="same",
        reuse=tf.AUTO_REUSE,
        name="stem_b21_conv1"
    )
    stem_b21_conv1 = batch_norm(
        inputs=stem_b21_conv1,
        is_training=is_training,
		reuse=True,
		layer_name="bn_stem_b21_conv1")
    # Convolutional Layer #2
    stem_b21_conv2 = tf.layers.conv2d(
        inputs=stem_b21_conv1,
        filters=24,
        kernel_size=[3, 3],
        padding="valid",
        reuse=tf.AUTO_REUSE,
        name="stem_b21_conv2"
    )
    stem_b21_conv2 = batch_norm(
        inputs=stem_b21_conv2,
        is_training=is_training,
		reuse=True,
		layer_name="bn_stem_b21_conv2")
    # Branch #2
    # Convolutional Layer #1
    stem_b22_conv1 = tf.layers.conv2d(
        inputs=stem_junction1,
        filters=16,
        kernel_size=[1, 3],
        padding="same",
        reuse=tf.AUTO_REUSE,
        name="stem_b22_conv1"
    )
    stem_b22_conv1 = batch_norm(
        inputs=stem_b22_conv1,
        is_training=is_training,
		reuse=True,
		layer_name="bn_stem_b22_conv1")
    # Convolutional Layer #2
    stem_b22_conv2 = tf.layers.conv2d(
        inputs=stem_b22_conv1,
        filters=16,
        kernel_size=[1, 5],
        padding="same",
        reuse=tf.AUTO_REUSE,
        name="stem_b22_conv2"
    )
    stem_b22_conv2 = batch_norm(
        inputs=stem_b22_conv2,
        is_training=is_training,
		reuse=True,
		layer_name="bn_stem_b22_conv2")
    # Convolutional Layer #3
    stem_b22_conv3 = tf.layers.conv2d(
        inputs=stem_b22_conv2,
        filters=16,
        kernel_size=[5, 1],
        padding="same",
        reuse=tf.AUTO_REUSE,
        name="stem_b22_conv3"
    )
    stem_b22_conv3 = batch_norm(
        inputs=stem_b22_conv3,
        is_training=is_training,
		reuse=True,
		layer_name="bn_stem_b22_conv3")
    # Convolutional Layer #4
    stem_b22_conv4 = tf.layers.conv2d(
        inputs=stem_b22_conv3,
        filters=24,
        kernel_size=[3, 3],
        strides=1,
        padding="valid",
        reuse=tf.AUTO_REUSE,
        name="stem_b22_conv4"
    )
    stem_b22_conv4 = batch_norm(
        inputs=stem_b22_conv4,
        is_training=is_training,
		reuse=True,
		layer_name="bn_stem_b22_conv4")
    # Junction #2
    stem_junction2 = tf.concat(
        values=[stem_b21_conv2, stem_b22_conv4],
        axis=3,
        name="stem_junction2"
    )
    return stem_junction2

def stem(inputs, is_training) :
    # Stem on every frame
    stem_output = [_single_frame_stem(inputs[:, :, :, i], is_training) for i in range(29)]
    # Concatenate all outputs into a single tensor
    stem_output = tf.concat(
        values=stem_output,
        axis=3,
        name="stem_output_concat"
    )
    # Convolutional Layer to reduce dimension
    stem_dim = tf.layers.conv2d(
        inputs=stem_output,
        filters=192,
        kernel_size=[1, 1],
        name="stem_dim"
    )
    stem_dim = batch_norm(
        inputs=stem_dim,
        is_training=is_training,
		layer_name="bn_stem_dim")
    return stem_dim

def inception_A(input, is_training, block_number) :
    # Bifurcation #1
    # Branch #1
    # Average Pooling Layer #1
    incepA_b11_pool1 = tf.layers.average_pooling2d(
        inputs=input,
        pool_size=2,
        strides=1,
        padding="same",
        name=block_number+"incepA_b11_pool1"
    )
    # Convolutional Layer #1
    incepA_b11_conv1 = tf.layers.conv2d(
        inputs=incepA_b11_pool1,
        filters=48,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepA_b11_conv1"
    )
    incepA_b11_conv1 = batch_norm(
        inputs=incepA_b11_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepA_b11_conv1"
    )

    # Branch #2
    # Convolutional Layer #1
    incepA_b12_conv1 = tf.layers.conv2d(
        inputs=input,
        filters=48,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepA_b12_conv1"
    )
    incepA_b12_conv1 = batch_norm(
        inputs=incepA_b12_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepA_b12_conv1"
    )

    # Branch #3
    # Convolutional Layer #1
    incepA_b13_conv1 = tf.layers.conv2d(
        inputs=input,
        filters=32,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepA_b13_conv1"
    )
    incepA_b13_conv1 = batch_norm(
        inputs=incepA_b13_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepA_b13_conv1"
    )
    # Convolutional Layer #2
    incepA_b13_conv2 = tf.layers.conv2d(
        inputs=incepA_b13_conv1,
        filters=48,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        name=block_number+"incepA_b13_conv2"
    )
    incepA_b13_conv2 = batch_norm(
        inputs=incepA_b13_conv2,
        is_training=is_training,
		layer_name=block_number+"bn_incepA_b13_conv2"
    )

    # Branch #4
    # Convolutional Layer #1
    incepA_b14_conv1 = tf.layers.conv2d(
        inputs=input,
        filters=32,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepA_b14_conv1"
    )
    incepA_b14_conv1 = batch_norm(
        inputs=incepA_b14_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepA_b14_conv1"
    )
    # Convolutional Layer #1
    incepA_b14_conv2 = tf.layers.conv2d(
        inputs=incepA_b14_conv1,
        filters=48,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        name=block_number+"incepA_b14_conv2"
    )
    incepA_b14_conv2 = batch_norm(
        inputs=incepA_b14_conv2,
        is_training=is_training,
		layer_name=block_number+"bn_incepA_b14_conv2"
    )
    # Convolutional Layer #3
    incepA_b14_conv3 = tf.layers.conv2d(
        inputs=incepA_b14_conv2,
        filters=48,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        name=block_number+"incepA_b14_conv3"
    )
    incepA_b14_conv3 = batch_norm(
        inputs=incepA_b14_conv3,
        is_training=is_training,
		layer_name=block_number+"bn_incepA_b14_conv3"
    )

    # Junction #1
    incepA_junction1 = tf.concat(
        values=[
            incepA_b11_conv1,
            incepA_b12_conv1,
            incepA_b13_conv2,
            incepA_b14_conv3
        ],
        axis=3,
        name=block_number+"incepA_junction1"
    )
    return incepA_junction1

def reduction_A(inception_A, is_training) :
    # Bifurcation #1
    # Branch #1
    # MaxPooling Layer #1
    reducA_b11_pool1 = tf.layers.max_pooling2d(
        inputs=inception_A,
        pool_size=[3, 3],
        strides=1,
        name="reducA_b11_pool1"
    )

    # Branch #2
    # Convolutional Layer #1
    reducA_b12_conv1 = tf.layers.conv2d(
        inputs=inception_A,
        filters=192,
        kernel_size=[3, 3],
        strides=1,
        padding="valid",
        name="reducA_b12_conv1"
    )
    reducA_b12_conv1 = batch_norm(
        inputs=reducA_b12_conv1,
        is_training=is_training,
		layer_name="bn_reducA_b12_conv1"
    )

    # Branch #3
    # Convolutional Layer #1
    reducA_b13_conv1 = tf.layers.conv2d(
        inputs=inception_A,
        filters=96,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name="reducA_b13_conv1"
    )
    reducA_b13_conv1 = batch_norm(
        inputs=reducA_b13_conv1,
        is_training=is_training,
		layer_name="bn_reducA_b13_conv1"
    )
    # Convolutional Layer #2
    reducA_b13_conv2 = tf.layers.conv2d(
        inputs=reducA_b13_conv1,
        filters=112,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        name="reducA_b13_conv2"
    )
    reducA_b13_conv2 = batch_norm(
        inputs=reducA_b13_conv2,
        is_training=is_training,
		layer_name="bn_reducA_b13_conv2"
    )
    # Convolutional Layer #3
    reducA_b13_conv3 = tf.layers.conv2d(
        inputs=reducA_b13_conv2,
        filters=128,
        kernel_size=[3, 3],
        strides=1,
        padding="valid",
        name="reducA_b13_conv3"
    )
    reducA_b13_conv3 = batch_norm(
        inputs=reducA_b13_conv3,
        is_training=is_training,
		layer_name="bn_reducA_b13_conv3"
    )

    # Junction #1
    reducA_junction1 = tf.concat(
        values=[
            reducA_b11_pool1,
            reducA_b12_conv1,
            reducA_b13_conv3
        ],
        axis=3,
        name="reducA_junction1"
    )
    return reducA_junction1

def inception_B(input, is_training, block_number) :
    # Bifurcation #1
    # Branch #1
    # Average Pooling
    incepB_b11_pool1 = tf.layers.average_pooling2d(
        inputs=input,
        pool_size=2,
        strides=1,
        padding="same",
        name=block_number+"incepB_b11_pool1"
    )
    # Convolutional Layer #1
    incepB_b11_conv1 = tf.layers.conv2d(
        inputs=incepB_b11_pool1,
        filters=64,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepB_b11_conv1"
    )
    incepB_b11_conv1 = batch_norm(
        inputs=incepB_b11_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepB_b11_conv1"
    )

    # Branch #2
    # Convolutional Layer #1
    incepB_b12_conv1 = tf.layers.conv2d(
        inputs=input,
        filters=192,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepB_b12_conv1"
    )
    incepB_b12_conv1 = batch_norm(
        inputs=incepB_b12_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepB_b12_conv1"
    )

    # Branch #3
    # Convolutional Layer #1
    incepB_b13_conv1 = tf.layers.conv2d(
        inputs=input,
        filters=96,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepB_b13_conv1"
    )
    incepB_b13_conv1 = batch_norm(
        inputs=incepB_b13_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepB_b13_conv1"
    )
    # Convolutional Layer #2
    incepB_b13_conv2 = tf.layers.conv2d(
        inputs=incepB_b13_conv1,
        filters=112,
        kernel_size=[1, 5],
        strides=1,
        padding="same",
        name=block_number+"incepB_b13_conv2"
    )
    incepB_b13_conv2 = batch_norm(
        inputs=incepB_b13_conv2,
        is_training=is_training,
		layer_name=block_number+"bn_incepB_b13_conv2"
    )
    # Convolutional Layer #3
    incepB_b13_conv3 = tf.layers.conv2d(
        inputs=incepB_b13_conv2,
        filters=128,
        kernel_size=[5, 1],
        strides=1,
        padding="same",
        name=block_number+"incepB_b13_conv3"
    )
    incepB_b13_conv3 = batch_norm(
        inputs=incepB_b13_conv3,
        is_training=is_training,
		layer_name=block_number+"bn_incepB_b13_conv3"
    )

    # Branch 4
    # Convolutional Layer #1
    incepB_b14_conv1 = tf.layers.conv2d(
        inputs=input,
        filters=96,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepB_b14_conv1"
    )
    incepB_b14_conv1 = batch_norm(
        inputs=incepB_b14_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepB_b14_conv1"
    )
    # Convolutional Layer #2
    incepB_b14_conv2 = tf.layers.conv2d(
        inputs=incepB_b14_conv1,
        filters=96,
        kernel_size=[1, 5],
        strides=1,
        padding="same",
        name=block_number+"incepB_b14_conv2"
    )
    incepB_b14_conv2 = batch_norm(
        inputs=incepB_b14_conv2,
        is_training=is_training,
		layer_name=block_number+"bn_incepB_b14_conv2"
    )
    # Convolutional Layer #3
    incepB_b14_conv3 = tf.layers.conv2d(
        inputs=incepB_b14_conv2,
        filters=112,
        kernel_size=[5, 1],
        strides=1,
        padding="same",
        name=block_number+"incepB_b14_conv3"
    )
    incepB_b14_conv3 = batch_norm(
        inputs=incepB_b14_conv3,
        is_training=is_training,
		layer_name=block_number+"bn_incepB_b14_conv3"
    )
    # Convolutional Layer #4
    incepB_b14_conv4 = tf.layers.conv2d(
        inputs=incepB_b14_conv3,
        filters=112,
        kernel_size=[1, 5],
        strides=1,
        padding="same",
        name=block_number+"incepB_b14_conv4"
    )
    incepB_b14_conv4 = batch_norm(
        inputs=incepB_b14_conv4,
        is_training=is_training,
		layer_name=block_number+"bn_incepB_b14_conv4"
    )
    # Convolutional Layer #4
    incepB_b14_conv5 = tf.layers.conv2d(
        inputs=incepB_b14_conv4,
        filters=128,
        kernel_size=[5, 1],
        strides=1,
        padding="same",
        name=block_number+"incepB_b14_conv5"
    )
    incepB_b14_conv5 = batch_norm(
        inputs=incepB_b14_conv5,
        is_training=is_training,
		layer_name=block_number+"bn_incepB_b14_conv"
    )

    # Junction 1
    incepB_junction1 = tf.concat(
        values=[
            incepB_b11_conv1,
            incepB_b12_conv1,
            incepB_b13_conv3,
            incepB_b14_conv5
        ],
        axis=3,
        name=block_number+"incepB_junction1"
    )
    return incepB_junction1

def reduction_B(inception_B, is_training) :
    # Bifurcation #1
    # Branch #1
    # Pooling Layer #1
    reducB_b11_pool1 = tf.layers.max_pooling2d(
        inputs=inception_B,
        pool_size=[3, 3],
        strides=1,
        name="reducB_b11_pool1"
    )

    # Branch #2
    # Convolutional Layer #1
    reducB_b12_conv1 = tf.layers.conv2d(
        inputs=inception_B,
        filters=96,
        kernel_size=[1, 1],
        padding="same",
        name="reducB_b12_conv1"
    )
    reducB_b12_conv1 = batch_norm(
        inputs=reducB_b12_conv1,
        is_training=is_training,
		layer_name="bn_reducB_b12_conv1"
    )
    # Convolutional Layer #2
    reducB_b12_conv2 = tf.layers.conv2d(
        inputs=reducB_b12_conv1,
        filters=96,
        kernel_size=[3, 3],
        padding="valid",
        strides=1,
        name="reducB_b12_conv2"
    )
    reducB_b12_conv2 = batch_norm(
        inputs=reducB_b12_conv2,
        is_training=is_training,
		layer_name="bn_reducB_b12_conv2"
    )
    # Branch #3
    # Convolutional Layer #1
    reducB_b13_conv1 = tf.layers.conv2d(
        inputs=inception_B,
        filters=128,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name="reducB_b13_conv1"
    )
    reducB_b13_conv1 = batch_norm(
        inputs=reducB_b13_conv1,
        is_training=is_training,
		layer_name="bn_reducB_b13_conv1"
    )
    # Convolutional Layer #2
    reducB_b13_conv2 = tf.layers.conv2d(
        inputs=reducB_b13_conv1,
        filters=128,
        kernel_size=[1, 5],
        strides=1,
        padding="same",
        name="reducB_b13_conv2"
    )
    reducB_b13_conv2 = batch_norm(
        inputs=reducB_b13_conv2,
        is_training=is_training,
		layer_name="bn_reducB_b13_conv2"
    )
    # Convolutional Layer #3
    reducB_b13_conv3 = tf.layers.conv2d(
        inputs=reducB_b13_conv2,
        filters=160,
        kernel_size=[5, 1],
        strides=1,
        padding="same",
        name="reducB_b13_conv3"
    )
    reducB_b13_conv3 = batch_norm(
        inputs=reducB_b13_conv3,
        is_training=is_training,
		layer_name="bn_reducB_b13_conv3"
    )
    # Convolutional Layer #4
    reducB_b13_conv4 = tf.layers.conv2d(
        inputs=reducB_b13_conv3,
        filters=160,
        kernel_size=[3, 3],
        strides=1,
        padding="valid",
        name="reducB_b13_conv4"
    )
    reducB_b13_conv4 = batch_norm(
        inputs=reducB_b13_conv4,
        is_training=is_training,
		layer_name="bn_reducB_b13_conv4"
    )

    # Junction 1
    reducB_junction1 = tf.concat(
        values=[
            reducB_b11_pool1,
            reducB_b12_conv2,
            reducB_b13_conv4
        ],
        axis=3,
        name="reducB_junction1"
    )
    return reducB_junction1

def inception_C(input, is_training, block_number) :
    # Bifurcation #1
    # Branch #1
    # Average Pooling
    incepC_b11_pool1 = tf.layers.average_pooling2d(
        inputs=input,
        pool_size=2,
        strides=1,
        padding="same",
        name=block_number+"incepC_b11_pool1"
    )
    # Convolutional Layer #1
    incepC_b11_conv1 = tf.layers.conv2d(
        inputs=incepC_b11_pool1,
        filters=128,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepC_b11_conv1"
    )
    incepC_b11_conv1 = batch_norm(
        inputs=incepC_b11_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepC_b11_conv1"
    )

    # Branch #2
    # Convolutional Layer #1
    incepC_b12_conv1 = tf.layers.conv2d(
        inputs=input,
        filters=128,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepC_b12_conv1"
    )
    incepC_b12_conv1 = batch_norm(
        inputs=incepC_b12_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepC_b12_conv1"
    )

    # Branch #3
    # Convolutional Layer #1
    incepC_b13_conv1 = tf.layers.conv2d(
        inputs=input,
        filters=192,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepC_b13_conv1"
    )
    incepC_b13_conv1 = batch_norm(
        inputs=incepC_b13_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepC_b13_conv1"
    )
    # Bifurcation #1
    # Branch #1
    # Convolutional Layer #1
    incepC_b13_b11_conv1 = tf.layers.conv2d(
        inputs=incepC_b13_conv1,
        filters=128,
        kernel_size=[1, 3],
        strides=1,
        padding="same",
        name=block_number+"incepC_b13_b11_conv1"
    )
    incepC_b13_b11_conv1 = batch_norm(
        inputs=incepC_b13_b11_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepC_b13_b11_conv1"
    )
    # Branch #2
    # Convolutional Layer #1
    incepC_b13_b12_conv1 = tf.layers.conv2d(
        inputs=incepC_b13_conv1,
        filters=128,
        kernel_size=[3, 1],
        strides=1,
        padding="same",
        name=block_number+"incepC_b13_b12_conv1"
    )
    incepC_b13_b12_conv1 = batch_norm(
        inputs=incepC_b13_b12_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepC_b13_b12_conv1"
    )

    # Branch #4
    # Convolutional Layer #1
    incepC_b14_conv1 = tf.layers.conv2d(
        inputs=input,
        filters=192,
        kernel_size=[1, 1],
        strides=1,
        padding="same",
        name=block_number+"incepC_b14_conv1"
    )
    incepC_b14_conv1 = batch_norm(
        inputs=incepC_b14_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepC_b14_conv1"
    )
    # Convolutional Layer #2
    incepC_b14_conv2 = tf.layers.conv2d(
        inputs=incepC_b14_conv1,
        filters=224,
        kernel_size=[1, 3],
        strides=1,
        padding="same",
        name=block_number+"incepC_b14_conv2"
    )
    incepC_b14_conv2 = batch_norm(
        inputs=incepC_b14_conv2,
        is_training=is_training,
		layer_name=block_number+"bn_incepC_b14_conv2"
    )
    # Convolutional Layer #3
    incepC_b14_conv3 = tf.layers.conv2d(
        inputs=incepC_b14_conv2,
        filters=256,
        kernel_size=[3, 1],
        strides=1,
        padding="same",
        name=block_number+"incepC_b14_conv3"
    )
    incepC_b14_conv3 = batch_norm(
        inputs=incepC_b14_conv3,
        is_training=is_training,
		layer_name=block_number+"bn_incepC_b14_conv3"
    )
    # Bifurcation #1
    # Branch #1
    # Convolutional Layer #1
    incepC_b14_b11_conv1 = tf.layers.conv2d(
        inputs=incepC_b14_conv3,
        filters=128,
        kernel_size=[1, 3],
        strides=1,
        padding="same",
        name=block_number+"incepC_b14_b11_conv1"
    )
    incepC_b14_b11_conv1 = batch_norm(
        inputs=incepC_b14_b11_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepC_b14_b11_conv1"
    )
    # Branch #2
    # Convolutional Layer #1
    incepC_b14_b12_conv1 = tf.layers.conv2d(
        inputs=incepC_b14_conv3,
        filters=128,
        kernel_size=[3, 1],
        strides=1,
        padding="same",
        name=block_number+"incepC_b14_b12_conv1"
    )
    incepC_b14_b12_conv1 = batch_norm(
        inputs=incepC_b14_b12_conv1,
        is_training=is_training,
		layer_name=block_number+"bn_incepC_b14_b12_conv1"
    )

    # Junction 1
    incepC_junction1 = tf.concat(
        values=[
            incepC_b11_conv1,
            incepC_b12_conv1,
            incepC_b13_b11_conv1,
            incepC_b13_b12_conv1,
            incepC_b14_b11_conv1,
            incepC_b14_b12_conv1
        ],
        axis=3,
        name=block_number+"incepC_junction1"
    )
    return incepC_junction1
