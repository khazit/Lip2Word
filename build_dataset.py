'''
Build the dataset by preprocessing all the videos (ie. reframed around the mouth and resized to 64x64) and adding the labels.
Save the datasets as numpy arrays (3 .npz files, training, validation and testing)
'''

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', 
                    default='/mnt/disks/sdb/data/LRDataset/sample_lipread_mp4/', 
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
    all_frames = np.zeros((number_frames, size, size))
    while success: 
        success, image = vidObj.read()
        if success :
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = resize_frame(image, size_frame-180, offset=35)
            image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            all_frames[count] = image
            count += 1
    return all_frames

def get_label_from_path(path) :
    '''
    Find the label (ie the word) from the path of the .mp4
    Args :
        - path : path to the .mp4 (eg : /mnt/disks/sdb/data/LRDataset/lipread_mp4/ABOUT/****)
    Returns :
        - a word (str)
    '''
    return path.split('/')[7]

def create_dict_word_list() :
    '''
    Create a dict used to transfrom labels from str to int
    Args :
        - None
    Return :
        - Python dictionnary {Word : Label}
    '''
    count = 0
    my_dict = dict()
    with open('/mnt/disks/sdb/data/LRDataset/word_list.txt', 'r') as f:
        for line in f:
            my_dict.update({line[:-1] : count})
            count += 1
    return my_dict

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Create the word to label dict
    label_dict = create_dict_word_list()
    
    # Somes useful variables
    size = 64 # size of the frames
    n_frames = 29 # number of frames
    
    # Iterate over the training, validation and test sets
    for set_type in ['train', 'val', 'test'] :
        
        # really ugly, may need to change it
        n_examples = 0
        pathlist = Path(args.data_dir).glob('**/'+set_type+'/*.mp4')
        for path in pathlist:
            n_examples += 1  # number of .mp4 files
        print("{} files found in {} directory".format(n_examples, set_type))
        
        # Create empty matrices 
        data_features = np.zeros((n_examples, n_frames, size, size)).astype(np.float32)
        data_labels = np.zeros((n_examples)).astype(np.float32)
        count = 0
        pathlist = Path("/mnt/disks/sdb/data/LRDataset/sample_lipread_mp4/").glob('**/'+set_type+'/*.mp4')
        
        # Iterate over .mp4 files in every sub directory (train, val, test)
        print("Processing {} data".format(set_type))
        for path in tqdm(pathlist):
            data_features[count] = capture_process_frames(str(path), size)
            data_labels[count] = label_dict[get_label_from_path(str(path))]
            count += 1
        
        # Saves every subdirectory data into a seperate .npz file  
        print("Saving {} data to {}".format(set_type, args.output_dir))
        np.savez_compressed(file=args.output_dir+'data_'+set_type, 
                            array1=data_features, 
                            array2=data_labels)
        
    print("Done building datasets")
