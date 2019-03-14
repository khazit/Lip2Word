"""
Define the model function.
"""

import tensorflow as tf

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
    
    # Input layer 
    input_layer = tf.reshape(features, [-1, 64, 64, num_frames])
    
    # Convolutional Layer #1 : 
    # on every frame separately , shared weights
    conv1 = list()
    for i in range(num_frames) :
        conv1.append(tf.layers.conv2d(
            inputs=tf.reshape(
                input_layer[:, :, :, i], 
                [-1, 64, 64, 1]),
            filters=48, 
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu, 
            name="conv1",
            reuse=tf.AUTO_REUSE))
    # Pooling Layer #1 :
    # 2x2, stride 2, on every frame.
    pool1 = list()
    for conv in conv1 :
        pool1.append(tf.layers.max_pooling2d(
            inputs=conv,
            pool_size=[2, 2],
            strides=2,
            name="pool1"))
    # Concatenate tensors along time dimension
    layer1 = tf.concat(
        values=pool1,
        axis=3,
        name="concat")
    
    # Convolutional Layer to reduce dimension
    conv_dim = tf.layers.conv2d(
        inputs=layer1,
        filters=92,
        kernel_size=[1, 1],
        name="conv_dim")
    
    # Convolutional and Pooling Layers #2
    conv2 = tf.layers.conv2d(
        inputs=conv_dim,
        filters=256,
        kernel_size=[3, 3],
        name="conv2")
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
    
    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs = conv3,
        filters=512,
        kernel_size=[3, 3],
        name="conv4")
    
    # Convolutional and Pooling Layers #5
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=512,
        kernel_size=[3, 3],
        name="conv5")
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
        activation=tf.nn.relu,
        name="dense")
    
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
    starter_learning_rate = 0.01
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(starter_learning_rate, 
                                               global_step,
                                               10000, 
                                               0.76, 
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