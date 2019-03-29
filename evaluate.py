from model.model_fn import cnn_model_fn
from model.input_fn import input_fn
import argparse
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', 
                    default='', 
                    help="Directory with processed dataset")
parser.add_argument('--model_dir',
                    default=None,
                    help="Model directory")
parser.add_argument('--data_set',
                    default='test',
                    help='Name of the datafile')


if __name__ == '__main__' :
    args = parser.parse_args()
    
    # Load the dataset
    print("Loading dataset from " + args.data_dir + args.data_set)
    test_dir = os.path.join(args.data_dir, args.data_set)
    assert os.path.isdir(test_dir)
    # Training data
    test_pathlist = Path(test_dir).glob("*.jpg")
    test_filenames = [str(path) for path in test_pathlist] # doesn't generalize
    test_labels = [int(s.split("_")[1].split('/')[2]) for s in test_filenames]
    
    print(test_filenames[:3])
    print(test_labels[:3])
    print("Done loading data")
    print("Test set size {}\n".format(
        len(test_filenames)))

    # Create the estimator
    print("Creating estimator from/to " + os.path.join("experiments", args.model_dir))
    cnn_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=os.path.join("experiments", args.model_dir))
    
    print("Evaluating model")
    test_results = cnn_classifier.evaluate(
        input_fn=lambda:input_fn(False,
                                 test_filenames,
                                 test_labels))    
    print("Results : \n{}".format(test_results))
