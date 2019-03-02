from model.model_fn import cnn_model_fn
import argparse
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
parser.add_argument('--data_name',
                    default='data_val.npz',
                    help='Name of the datafile')


if __name__ == '__main__' :
    args = parser.parse_args()
    
    # Load the dataset
    print("Loading dataset from "+args.data_dir+args.data_name)
    val = np.load(file=args.data_dir+args.data_name)
    val_data = val.f.array1
    val_labels = val.f.array2
    randomize = np.arange(len(val_data))
    np.random.shuffle(randomize)
    val_data = val_data[randomize]
    val_labels = val_labels[randomize]

    print("Done loading data")
    print("Data summary :\n\tExamples : {}\n\tLabels : {}"
         .format(val_data.shape,
                 val_labels.shape))

    # Create the estimator
    print("Creating estimator from/to "+"experiments/"+args.model_dir)
    cnn_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="experiments/"+args.model_dir)

    # Define input function
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": val_data},
        y=val_labels,
        num_epochs=1,
        shuffle=False)
    
    print("Evaluating model")
    eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
    
    print("Results : \n{}".format(eval_results))
