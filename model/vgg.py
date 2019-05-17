import tensorflow as tf
from model.utils import batch_norm

def vgg_model_fn(features, labels, mode, params):
    """
    Model function for the CNN (Multiple Towers)
    Code inspired by https://www.tensorflow.org/tutorials/estimators/cnn
    Args :
        - features: a dict of the features
        - labels: numpy array containing the labels
        - mode: PREDICT, TRAIN or EVAL
    Returns :
        - Custom Estimator
    """
    # Useful variables
    num_classes = params["num_classes"]
    if (mode == tf.estimator.ModeKeys.TRAIN) :
        is_training = tf.constant(True, dtype=tf.bool)
    else :
        is_training = tf.constant(False, dtype=tf.bool)

    # Compute the logits
    logits=_build_model(features, num_classes, is_training)

    # Generate predictions
    predictions = {
        "classes": tf.argmax(
            input=logits,
            axis=1,
            name="argmax"
        ),
        "probabilities": tf.nn.softmax(
            logits=logits,
            name="softmax"
        )
    }

    # Prediction stops here
    if mode == tf.estimator.ModeKeys.PREDICT :
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
    )

    # Calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    # Compute evaluation metrics
    accuracy = tf.metrics.accuracy(
        labels=labels,
        predictions=predictions["classes"],
        name='acc_op'
    )
    topk_accuracy = tf.metrics.mean(
        tf.nn.in_top_k(
            predictions=logits,
            targets=labels,
            k=10
        )
    )
    metrics = {
        'accuracy': accuracy,
        'topk_accuracy': topk_accuracy
    }

    # Display stuff on Tensorboard
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('topk_accuracy', topk_accuracy[1])
    tf.summary.scalar('loss', loss)
    # Uncomment to display image inputs on Tensorboard
    # tf.summary.image(
    #     tensor=tf.reshape(
    #         features[:, :, :, 1],
    #         [-1, 64, 64, 1]),
    #     name="viz")

    # Logging hook
    hook = tf.train.LoggingTensorHook(
        tensors = {
            "accuracy": accuracy[0],
            "topk_accuracy": topk_accuracy[0]
        },
        every_n_iter=100
    )

    # Evaluation stops here
    if mode == tf.estimator.ModeKeys.EVAL :
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=metrics,
            evaluation_hooks=[hook]
        )

    # Learning rate
    starter_learning_rate = params["starter_learning_rate"]
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(
        learning_rate=starter_learning_rate,
        global_step=global_step,
        decay_steps=params["decay_steps"],
        decay_rate=params["decay_rate"],
        staircase=False
    )

    # Display more stuff on Tensorboard (training related)
    tf.summary.scalar('learning_rate', learning_rate)

    # Optimizer specifications
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params["optimizer_momentum"]
    )
    train_op = optimizer.minimize(
        loss=loss,
        global_step=global_step
    )

    # Add the update ops for the moving_mean and moving_variance of batchnorm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([train_op, update_ops])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=[hook]
    )

def _build_model(features, num_classes, is_training) :
    """
    Build the graph for the Multiple Tower architecture
    Args :
        - features : Tensor of the inputs
        - num_classes : number of number of classes
        - is_training : whether it's training mode or inference mode
    Returns :
        - logits
    """
    # Useful variables
    num_frames = 29

    # Convolutional Layer #1 :
    # on every frame separately , shared weights
    conv1 = list()
    bn = list()
    for i in range(num_frames) :
        conv1.append(
            tf.layers.conv2d(
                inputs=tf.reshape(
                    features[:, :, :, i],
                    [-1, 64, 64, 1]
                ),
                filters=48,
                kernel_size=[3, 3],
                padding="valid",
                name="conv1",
                reuse=tf.AUTO_REUSE
            )
        )
    for layer in conv1 :
        bn.append(
            batch_norm(
                inputs=layer,
                is_training=is_training,
    		layer_name="bn_conv1",
                reuse=True
            )
        )
    # Concatenate tensors along time dimension
    conv1_concat = tf.concat(
        values=bn,
        axis=3,
        name="concat"
    )
    # Pooling Layer #1 :
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1_concat,
        pool_size=[2, 2],
        strides=2,
        name="pool1"
    )

    # Convolutional Layer to reduce dimension
    conv_dim = tf.layers.conv2d(
        inputs=pool1,
        filters=92,
        kernel_size=[1, 1],
        name="conv_dim"
    )
    conv_dim = batch_norm(
        inputs=conv_dim,
        is_training=is_training,
		layer_name="bn_conv_dim"
    )

    # Convolutional and Pooling Layers #2
    conv2 = tf.layers.conv2d(
        inputs=conv_dim,
        filters=256,
        kernel_size=[3, 3],
        name="conv2"
    )
    conv2 = batch_norm(
        inputs=conv2,
        is_training=is_training,
		layer_name="bn_conv2"
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        name="pool2"
    )

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs = pool2,
        filters=512,
        kernel_size=[3, 3],
        name="conv3"
    )
    conv3 = batch_norm(
        inputs=conv3,
        is_training=is_training,
		layer_name="bn_conv3"
    )

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs = conv3,
        filters=512,
        kernel_size=[3, 3],
        name="conv4"
    )
    conv4 = batch_norm(
        inputs=conv4,
        is_training=is_training,
		layer_name="bn_conv4"
    )

    # Convolutional and Pooling Layers #5
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=512,
        kernel_size=[3, 3],
        name="conv5"
    )
    conv5 = batch_norm(
        inputs=conv5,
        is_training=is_training,
		layer_name="bn_conv5"
    )
    pool5 = tf.layers.max_pooling2d(
        inputs=conv5,
        pool_size=[2, 2],
        strides=2,
        name="pool5"
    )

    # Flatten the feature map
    pool5_flat = tf.reshape(
        pool5,
        [-1, 8192],
        name="pool5_flat"
    )

    # Dense Layer
    dense = tf.layers.dense(
        inputs=pool5_flat,
        units=4096,
        name="dense"
    )
    dense = batch_norm(
        inputs=dense,
        is_training=is_training,
		layer_name="bn_dense"
    )

    # IF OVERFITTING, USE DROPOUT HERE

    # Logits
    logits = tf.layers.dense(
        inputs=dense,
        units=num_classes,
        name="logits"
    )
    return logits
