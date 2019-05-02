"""
Describe the training.
"""

import argparse
import os
import json
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
    '--n_steps',
    default=0,
    help="Number of steps"
)
parser.add_argument(
    '--n_epochs',
    default=0,
    help="Number of epochs"
)
parser.add_argument(
    '--debugging',
    default=False,
    help="Enable the debugging mode (infinite epochs)"
)


if __name__ == '__main__' :
    # Parse arguments
    args = parser.parse_args()

    # Useful variables
    if args.model_fn == "vgg":
        model_fn = vgg_model_fn
    elif args.model_fn == "inception":
        model_fn = inception_model_fn
    if args.debugging == "True" :
        debugging = True
        model_dir = None
    else :
        debugging = False
        model_dir = os.path.join("experiments", args.params_file)
    n_steps = int(args.n_steps)
    n_epochs = int(args.n_epochs)

    # Check if .json file exists, then read it
    params_file = os.path.join("hyperparameters", args.params_file + ".json")
    assert os.path.isfile(params_file), "No .json file found"
    with open(params_file) as json_file:
        params = json.load(json_file)
    print("Parameters used :\n{}".format(params))

    # Check if dataset exists
    print("Loading dataset from "+args.data_dir)
    train_dir = os.path.join(args.data_dir, "train")
    assert os.path.isdir(train_dir), "No training directory found"
    val_dir = os.path.join(args.data_dir, "val")
    assert os.path.isdir(val_dir), "No validation directory found"

    # Training data
    train_pathlist = Path(train_dir).glob("*.jpg")
    train_filenames = [str(path) for path in train_pathlist]
    train_filenames = [s for s in train_filenames if int(s.split("_")[1].split('/')[2]) < params["num_classes"]]
    train_labels = [int(s.split("_")[1].split('/')[2]) for s in train_filenames]

    # Validation data
    val_pathlist = Path(val_dir).glob("*.jpg")
    val_filenames = [str(path) for path in val_pathlist]
    val_filenames = [s for s in val_filenames if int(s.split("_")[1].split('/')[2]) < params["num_classes"]]
    val_labels = [int(s.split("_")[1].split('/')[2]) for s in val_filenames]

    # Data summary after loading
    print("Done loading data")
    print("Data summary :\n* Training set size {}\n* Validation set size {}".format(
        len(train_filenames),
        len(val_filenames))
    )

    # Create the estimator
    print("Creating the custom estimator")
    cnn_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=params
    )

    if debugging :
        print("DEBUGGING MODE ENABLED")
        print("Training classifier for {} steps".format(n_steps))
        cnn_classifier.train(
            input_fn=lambda:input_fn(
                is_training=True,
                num_epochs=-1,
                filenames=train_filenames[:10],
                labels=train_labels[:10],
                batch_size=32
            ),
            steps=n_steps
        )
    else :
        # If the number of epochs is not defined (= 0), then train on number of
        # steps and evaluate at the end of the training ...
        if (n_epochs == 0) :
            print("Training classifier for {} steps".format(n_steps))
            cnn_classifier.train(
                input_fn=lambda:input_fn(
                    is_training=True,
                    num_epochs=1,
                    filenames=train_filenames,
                    labels=train_labels,
                    batch_size=32
                ),
                steps=n_steps
            )
            val_results = cnn_classifier.evaluate(
                input_fn=lambda:input_fn(
                    is_training=False,
                    filenames=val_filenames,
                    labels=val_labels
                )
            )
        # else train on multiple epochs and evaluate every epoch
        else :
            for i in range(n_epochs) :
                cnn_classifier.train(
                    input_fn=lambda:input_fn(
                        is_training=True,
                        num_epochs=1,
                        filenames=train_filenames,
                        labels=train_labels,
                        batch_size=32
                    )
                )
                val_results = cnn_classifier.evaluate(
                    input_fn=lambda:input_fn(
                        is_training=False,
                        filenames=val_filenames,
                        labels=val_labels
                    )
                )
        print("Results : \n{}".format(val_results))
    print("Done training")

    if not debugging :
        # Save results to .json file
        # But first convert values from float32 to string
        for key in val_results :
            val_results[key] = str(val_results[key])
        with open(os.path.join(model_dir, "results.json"), 'w') as outfile:
            json.dump(val_results, outfile)
        print("Results saved to {}".format(model_dir))
