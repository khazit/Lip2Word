from model.model_functions.vgg import vgg_model_fn
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
parser.add_argument('--n_steps',
                    default=1,
                    help="Number of steps")
parser.add_argument('--model_dir',
                    default=None,
                    help="Model directory")


if __name__ == '__main__' :
    args = parser.parse_args()

    # Load the training dataset
    print("Loading dataset from "+args.data_dir)
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    assert os.path.isdir(train_dir), "No training directory found"
    assert os.path.isdir(val_dir), "No validation directory found"
    # Training data
    train_pathlist = Path(train_dir).glob("*.jpg")
    train_filenames = [str(path) for path in train_pathlist]
#    train_filenames = [s for s in train_filenames if int(s.split("_")[1].split('/')[2]) < 10] ## REMOVE AFTER TESTS
    train_labels = [int(s.split("_")[1].split('/')[2]) for s in train_filenames]
    # Validation data
    val_pathlist = Path(val_dir).glob("*.jpg")
    val_filenames = [str(path) for path in val_pathlist]
#    val_filenames = [s for s in val_filenames if int(s.split("_")[1].split('/')[2]) < 10] ## REMOVE AFTER TESTS
    val_labels = [int(s.split("_")[1].split('/')[2]) for s in val_filenames]

    print("Done loading data")
    print("Data summary :\n\tTraining set size {}\n\tValidation set size {}".format(
        len(train_filenames),
        len(val_filenames)))

    # Create the estimator
    print("Creating estimator from/to " + os.path.join("experiments", args.model_dir))
    cnn_classifier = tf.estimator.Estimator(
        model_fn=vgg_model_fn,
        model_dir=os.path.join("experiments", args.model_dir))

    print("Training classifier for {} steps".format(args.n_steps))
    n_steps = int(args.n_steps)
    cnn_classifier.train(
            input_fn=lambda:input_fn(True,
                                    train_filenames,
                                    train_labels,
                                    32),
            steps=n_steps)
    val_results = cnn_classifier.evaluate(
            input_fn=lambda:input_fn(False,
                                    val_filenames,
                                    val_labels,
                                    None))
#   print("Results : \n{}".format(val_results))

    # for i in range(n_steps // 20000) :
    #     cnn_classifier.train(
    #         input_fn=lambda:input_fn(True,
    #                                  train_filenames,
    #                                  train_labels,
    #                                  32),
    #         steps=20000)
    #     val_results = cnn_classifier.evaluate(
    #         input_fn=lambda:input_fn(False,
    #                                  val_filenames,
    #                                  val_labels,
    #                                  None))
    print("Results : \n{}".format(val_results))
    print("Done training")
