"""
Describe the training.
"""

from model.model_fn.vgg import vgg_model_fn
from model.model_fn.inception import inception_model_fn
from model.input_fn import input_fn
import argparse
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='',
    help="Directory with processed dataset"
)
parser.add_argument(
    '--model_dir',
    default=None,
    help="Model directory"
)
parser.add_argument(
    '--model_fn',
    default="vgg",
    help="Model function (vgg or inception)"
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


if __name__ == '__main__' :
    # Parse arguments
    args = parser.parse_args()

    # Useful variable
    if args.model_fn == "vgg":
        model_fn = vgg_model_fn
    elif args.model_fn == "inception":
        model_fn = inception_model_fn
    n_steps = int(args.n_steps)
    n_epochs = int(args.n_epochs)

    # Check if dataset exists
    print("Loading dataset from "+args.data_dir)
    train_dir = os.path.join(args.data_dir, "train")
    assert os.path.isdir(train_dir), "No training directory found"
    val_dir = os.path.join(args.data_dir, "val")
    assert os.path.isdir(val_dir), "No validation directory found"

    # Training data
    train_pathlist = Path(train_dir).glob("*.jpg")
    train_filenames = [str(path) for path in train_pathlist]
    #train_filenames = [s for s in train_filenames if int(s.split("_")[1].split('/')[2]) < 10] ## REMOVE AFTER TESTS
    train_labels = [int(s.split("_")[1].split('/')[2]) for s in train_filenames]

    # Validation data
    val_pathlist = Path(val_dir).glob("*.jpg")
    val_filenames = [str(path) for path in val_pathlist]
    #val_filenames = [s for s in val_filenames if int(s.split("_")[1].split('/')[2]) < 10] ## REMOVE AFTER TESTS
    val_labels = [int(s.split("_")[1].split('/')[2]) for s in val_filenames]

    # Data summary after loading
    print("Done loading data")
    print("Data summary :\n\tTraining set size {}\n\tValidation set size {}".format(
        len(train_filenames),
        len(val_filenames))
    )

    # Create the estimator
    print("Creating estimator from/to "
        + os.path.join("experiments", args.model_dir))
    cnn_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=os.path.join("experiments", args.model_dir)
    )

    # If the number of epochs is not defined (= 0), then train on number of
    # steps and evaluate at the end of the training ...
    if (n_epochs == 0) :
        print("Training classifier for {} steps".format(n_steps))
        cnn_classifier.train(
            input_fn=lambda:input_fn(
                True,
                train_filenames,
                train_labels,
                32
            ),
            steps=n_steps
        )
        val_results = cnn_classifier.evaluate(
            input_fn=lambda:input_fn(
                False,
                val_filenames,
                val_labels
            )
        )
    # else train on multiple epochs and evaluate every epoch
    else :
        for i in range(n_epochs) :
            cnn_classifier.train(
                input_fn=lambda:input_fn(
                    True,
                    train_filenames,
                    train_labels,
                    32
                )
            )
            val_results = cnn_classifier.evaluate(
                input_fn=lambda:input_fn(
                    False,
                    val_filenames,
                    val_labels
                )
            )
    print("Results : \n{}".format(val_results))
    print("Done training")
