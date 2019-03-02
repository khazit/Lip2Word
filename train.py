from model.model_fn import cnn_model_fn
import argparse
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
    print("Loading dataset from "+args.data_dir+"data_train.npz")
    train = np.load(file=args.data_dir+'data_train.npz')
    train_data = train.f.array1
    train_labels = train.f.array2
    randomize = np.arange(len(train_data))
    np.random.shuffle(randomize)
    train_data = train_data[randomize]
    train_labels = train_labels[randomize]

    print("Done loading data")
    print("Data summary :\n\tExamples : {}\n\tLabels : {}"
         .format(train_data.shape,
                 train_labels.shape))

    # Create the estimator
    print("Creating estimator from/to "+"experiments/"+args.model_dir)
    cnn_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="experiments/"+args.model_dir)

    # Define input function
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=32,
        num_epochs=None,
        shuffle=True)

    print("Training classifier for {} steps".format(args.n_steps))
    cnn_classifier.train(
        input_fn=train_input_fn,
        steps=int(args.n_steps))
