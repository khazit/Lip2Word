"""
Print the total number of parameters of a Tensorflow Estimator and the number
of parameters in every layers.
Slightly modified version of this source code :
https://gist.github.com/dalgu90/a9952dfd372cbe1cdc529b204329e189
"""

import sys
import tensorflow as tf
import numpy as np
from model.vgg import vgg_model_fn
from model.inception import inception_model_fn


if len(sys.argv) == 3:
    model_dir = sys.argv[1]
else:
    print('Usage: python count_ckpt_param.py path_to_chkpt')
    sys.exit(1)

if sys.argv[2] == "vgg" :
    model_fn = vgg_model_fn
elif sys.argv[2] == "inception" :
    model_fn = inception_model_fn

cnn_classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=model_dir)

reader = tf.train.NewCheckpointReader(tf.train.latest_checkpoint(model_dir))

print('\nCount the number of parameters from {}'.format(model_dir))
param_map = reader.get_variable_to_shape_map()
total_count = 0
for k, v in param_map.items():
    if 'Momentum' not in k and 'global_step' not in k:
        temp = np.prod(v)
        total_count += temp
        print('%s: %s => %d' % (k, str(v), temp))
print('Total Param Count: %d' % total_count)
