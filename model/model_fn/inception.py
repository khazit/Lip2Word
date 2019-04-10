import tensorflow as tf
from model.model_fn.inception_modules import *

def _build_model(inputs, num_classes, is_training):
    """
    Build the graph for the inception_v4 architecture
    Architecture :
        inputs          : (64 x 64 x  29)
        stem            : (12 x 12 x 192)
        inception_A     : (12 x 12 x 192)
        reduction_A     : (10 x 10 x 512)
        inception_B     : (10 x 10 x 512)
        reduction_B     : ( 8 x  8 x 768)
        inception_C     : ( 8 x  8 x 768)
        average_pooling : (          768)
    Args :
        - inputs : Tensor of the inputs
        - num_classes : number of number of classes
        - is_training : whether it's training mode or inference mode
    Returns :
        - logits
    """
    # Stem
    stem_output = stem(inputs)
    # Inception-A
    incepA_output = inception_A(stem_output)
    # Reduction-A
    reducA_output = reduction_A(incepA_output)
    # Inception-B
    incepB_output = inception_B(reducA_output)
    # Reduction-B
    reducB_output = reduction_B(incepB_output)
    # Inception-C
    incepC_output = inception_C(reducB_output)
    # Average Pooling Layer
    avg_pooling = tf.layers.average_pooling2d(
        inputs=incepC_output,
        pool_size=8,
        strides=1,
        padding="valid",
        name="average_pooling"
    )
    # Dense Layer
    dense = tf.layers.flatten(
        inputs=avg_pooling,
    )
    # Dropout (keep 0.8)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.2,
        training=is_training,
        name="dropout"
    )
    # Logits
    logits = tf.layers.dense(
        inputs=dense,
        units=num_classes,
        name="logits"
    )
    return logits

def inception_model_fn(features, labels, mode):
    """
    Model function for the CNN (Inception architecture)
    Args :
        - features: a dict of the features
        - labels: numpy array containing the labels
        - mode: PREDICT, TRAIN or EVAL
    Returns :
        - Custom Estimator
    """
    # Useful variables
    num_classes = 500
    if (mode == tf.estimator.ModeKeys.TRAIN) :
        is_training = tf.constant(True, dtype=tf.bool)
    else :
        is_training = tf.constant(False, dtype=tf.bool)

    # Compute logits
    logits = _build_model(features, num_classes, is_training)

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
            k=3
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
    starter_learning_rate = 0.002
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(
        starter_learning_rate,
        global_step,
        20000,
        0.79,
        staircase=False
    )

    # Display more stuff on Tensorboard (training related)
    tf.summary.scalar('learning_rate', learning_rate)

    # Optimizer specifications
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=0.9
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
