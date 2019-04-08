import tensorflow as tf

def batch_norm(inputs, is_training) :
    '''
    Utility function. BatchNorm + ReLu
    Args :
        - inputs : a tensor of inputs
        - is_training : a bool
    Return :
        - Tensor after BatchNorm + ReLu. Same shape as inputs
    '''
    inputs =  tf.layers.batch_normalization(
        inputs=inputs,
        training=is_training)
    return tf.nn.relu(inputs)
