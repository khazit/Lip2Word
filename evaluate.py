import os
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from model.vgg import vgg_model_fn
from model.inception import inception_model_fn
from model.input_fn import input_fn

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='',
    help="Directory with processed dataset"
)
parser.add_argument(
    '--model_fn',
    default="vgg",
    help="Model function (vgg or inception)"
)
parser.add_argument(
    '--params_file',
    default="",
    help="Path to the .json file containing the parameters"
)
parser.add_argument(
    '--data_set',
    default='test',
    help='Name of the datafile'
)

if __name__ == '__main__' :
    args = parser.parse_args()

    # Useful variables
    if args.model_fn == "vgg":
        model_fn = vgg_model_fn
    elif args.model_fn == "inception":
        model_fn = inception_model_fn
    model_dir = os.path.join("experiments", args.params_file)

    # Check if .json file exists, then read it
    params_file = os.path.join("hyperparameters", args.params_file + ".json")
    assert os.path.isfile(params_file), "No .json file found"
    with open(params_file) as json_file:
        params = json.load(json_file)
    print("Parameters used :\n{}".format(params))

    # Load the dataset
    print("Loading dataset from " + args.data_dir + args.data_set)
    test_dir = os.path.join(args.data_dir, args.data_set)
    assert os.path.isdir(test_dir), "No test directory found"
    # Test data
    test_pathlist = Path(test_dir).glob("*.jpg")
    test_filenames = [str(path) for path in test_pathlist]
    test_filenames = [s for s in test_filenames if int(s.split("_")[1].split('/')[2]) < params["num_classes"]]
    test_labels = [int(s.split("_")[1].split('/')[2]) for s in test_filenames]

    print("Done loading data")
    print("Test set size {}\n".format(len(test_filenames)))

    # Create the estimator
    cnn_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=params
    )

    print("Evaluating model")
    test_results = cnn_classifier.evaluate(
        input_fn=lambda:input_fn(
            is_training=False,
            filenames=test_filenames,
            labels=test_labels)
        )
    print("Results : \n{}".format(test_results))
