'''
Build the dataset by preprocessing all the videos (ie. reframed around the mouth and resized to 64x64) and adding the labels.
Save each video to a 1856x64 image.
'''

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', 
                    default='', 
                    help="Directory with the dataset")
parser.add_argument('--output_dir', 
                    default='', 
                    help="Where to write the new data")

def resize_frame(image, size, offset) :
    '''
    Resize the image from the center point
    Args :
        - image : numpy ndarray representing the image
        - size : size of the image
        - offset : shift of the middle point
    Returns :
        - a numpy ndarrray representing the resized image
    '''
    epsilon = size // 2
    mid = image.shape[0] // 2
    resized_image = image[mid-epsilon+offset:mid+epsilon+offset, mid-epsilon:mid+epsilon]
    return resized_image

def capture_process_frames(path, size) :
    '''
    Captures and processes all the frames from a video
    Args :
        - path : path to the .mp4 file
        - size : size of the image
    Returns 
        - a vector representing the video (all frames, concatenated along a third dimension (time))
    '''
    vidObj = cv2.VideoCapture(path) 
    count = 0
    success = 1
    size_frame = 256 # size of the original frame from the video
    number_frames = 29 # all videos are 29 frames
    all_frames = np.zeros((size*number_frames, size))
    while success: 
        success, image = vidObj.read()
        if success :
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = resize_frame(image, size_frame-180, offset=35)
            image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            all_frames[count*size:size*(count+1), :] = image
            count += 1
    return all_frames

def get_label_from_path(path, label_dict) :
    '''
    Find the label from the path of the .mp4
    Args :
        - path : path to the .mp4
        - label_dict : a dict to match a word to a label
    Returns :
        - label (int)
    '''
    return label_dict[path.split('/')[5]]

def create_dict_word_list(path) :
    '''
    Create a dict used to transfrom labels from str to int
    Args :
        - path : Path to the word list
    Return :
        - Python dictionnary {Word : Label}
    '''
    count = 0
    my_dict = dict()
    with open(path+'word_list.txt', 'r') as f:
        for line in f:
            my_dict.update({line[:-1] : count})
            count += 1
    return my_dict


if __name__ == '__main__':
    args = parser.parse_args()

    # Tests
    assert os.path.isdir(args.data_dir), "No data directory found"
    assert os.path.isdir(args.output_dir), "No output directory found"
    print("{} files found.".format(
        len([path for path in Path(args.data_dir).glob('**/**/*.mp4')])))
    print("Saving files to "+args.output_dir)
    input("Press the <ENTER> key to continue...")
    
    # Somes useful variables
    label_dict = create_dict_word_list(args.data_dir)
    size = 64 # size of the frames
    n_frames = 29 # number of frames
    
    # Iterate over the training, validation and test sets
    for set_type in ['train', 'val', 'test'] :
        # Useful for naming the image files
        count = 0
        
        # Create empty matrix
        image = np.zeros((n_frames*size, size)).astype(np.float32)
        
        # Iterate over .mp4 files in every sub directory (train, val, test)
        pathlist = Path(args.data_dir).glob('**/'+set_type+'/*.mp4')
        print("Processing {} data".format(set_type))
        for path in tqdm(pathlist):
            image = capture_process_frames(str(path), size) 
            cv2.imwrite(args.output_dir+'{}/{}_{}_{}.jpg'.format(set_type, 
                                                              get_label_from_path(str(path), label_dict), 
                                                              str(path).split('/')[5],   
                                                              count),
                        image)
            count += 1

    print("Done building datasets")
