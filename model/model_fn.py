"""
Define the model function.
"""

import tensorflow as tf

def batch_norm(inputs, is_training) :
    '''

    '''
    inputs =  tf.layers.batch_normalization(
        inputs=inputs,
        training=is_training)
    return tf.nn.relu(inputs)

def cnn_model_fn(features, labels, mode):
    '''
    Model function for the CNN (Multiple Towers)
    Code inspired by :
        https://www.tensorflow.org/tutorials/estimators/cnn
    Args :
        - features: a dict of the features
        - labels: numpy array containing the labels
        - mode: PREDICT, TRAIN or EVAL
    Returns :
        - Custom Estimator
    '''
    # Useful variables
    num_frames = 29
    num_classes = 500
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # tf.summary.image(
    #     tensor=tf.reshape(
    #         features[:, :, :, 1],
    #         [-1, 64, 64, 1]),
    #     name="viz")

    # Convolutional Layer #1 :
    # on every frame separately , shared weights
    conv1 = list()
    for i in range(num_frames) :
        conv1.append(
            tf.layers.conv2d(
                inputs=tf.reshape(
                    features[:, :, :, i],
                    [-1, 64, 64, 1]),
                filters=48,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu,
                name="conv1",
                reuse=tf.AUTO_REUSE
            )
        )
    # Concatenate tensors along time dimension
    conv1_concat = tf.concat(
        values=conv1,
        axis=3,
        name="concat")
    # Pooling Layer #1 :
    # 2x2, stride 2, on every frame.
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1_concat,
        pool_size=[2, 2],
        strides=2,
        name="pool1")

    # Convolutional Layer to reduce dimension
    conv_dim = tf.layers.conv2d(
        inputs=pool1,
        filters=92,
        kernel_size=[1, 1],
        name="conv_dim")
    conv_dim = batch_norm(
        inputs=conv_dim,
        is_training=is_training)

    # Convolutional and Pooling Layers #2
    conv2 = tf.layers.conv2d(
        inputs=conv_dim,
        filters=256,
        kernel_size=[3, 3],
        name="conv2")
    conv2 = batch_norm(
        inputs=conv2,
        is_training=is_training)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        name="pool2")

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs = pool2,
        filters=512,
        kernel_size=[3, 3],
        name="conv3")
    conv3 = batch_norm(
        inputs=conv3,
        is_training=is_training)

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs = conv3,
        filters=512,
        kernel_size=[3, 3],
        name="conv4")
    conv4 = batch_norm(
        inputs=conv4,
        is_training=is_training)

    # Convolutional and Pooling Layers #5
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=512,
        kernel_size=[3, 3],
        name="conv5")
    conv5 = batch_norm(
        inputs=conv5,
        is_training=is_training)
    pool5 = tf.layers.max_pooling2d(
        inputs=conv5,
        pool_size=[2, 2],
        strides=2,
        name="pool5")

    # Flatten the feature map
    pool5_flat = tf.reshape(pool5,
                            [-1, 8192],
                            name="pool5_flat")

    # Dense Layer
    dense = tf.layers.dense(
        inputs=pool5_flat,
        units=4096,
        name="dense")
    dense = batch_norm(
        inputs=dense,
        is_training=is_training)

    # IF OVERFITTING, USE DROPOUT HERE

    # Logits Layer
    logits = tf.layers.dense(
        inputs=dense,
        units=num_classes,
        name="logits")

    # Generate predictions
    predictions = {
        "classes": tf.argmax(input=logits,
                             axis=1,
                             name="argmax"),
        "probabilities": tf.nn.softmax(logits=logits,
                                       name="softmax")
    }

    # Prediction stops here
    if mode == tf.estimator.ModeKeys.PREDICT :
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    # Calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits)

    # Evaluation metrics
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predictions["classes"],
                                   name='acc_op')
    metrics = {'accuracy' : accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('loss', loss)

    # Evaluation stops here
    if mode == tf.estimator.ModeKeys.EVAL :
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=metrics)

    # Training stops here
    starter_learning_rate = 0.002
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step,
                                               20000,
                                               0.79,
                                               staircase=False)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=0.9)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)
