"""
Given a input video (.mp4), outputs the probabilities of the k most
plausible words.
Also outputs a .avi file that shows the input of the neural network and
a summary bar graph
"""

import os
import cv2
import sys
import argparse
import face_recognition
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model.inception import inception_model_fn

tf.logging.set_verbosity(tf.logging.FATAL)
plt.style.use(['dark_background', 'presentation.mplstyle'])


def videoToArray(path) :
    """
    Capture every frame of a .mp4 file and save them to
    a 3D numpy array.
    Args :
        - path: path to the .mp4 file
    Return :
        - 4D numpy array
    """
    vidObj = cv2.VideoCapture(path)

    # Some useful info about the video
    width = int(vidObj.get(3))
    height = int(vidObj.get(4))
    fps = int(vidObj.get(5))
    n_frames = int(vidObj.get(7))
    print("Video info : {}x{}, {} frames".format(
        height,
        width,
        n_frames))
    # Create the numpy array that will host all the frames
    # Could use np.append later in the loop but this is
    # more efficient
    video = np.zeros((height, width, n_frames))
    video = video.astype(np.uint8)

    # Iterate over every frame of the video
    i = 0
    while True :
        # Capture one frame
        success, frame = vidObj.read()
        if not success :
            break;
        else :
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Save to one 4D numpy array
            video[:, :, i] = frame
            i += 1
    return video, n_frames, fps

def frameAdjust(video):
    """
    Select a fixed number of frames from the input video
    Args :
        - 3D numpy array
    Returns :
        - Adjusted numpy array
    """
    target = 29
    n_frames = video.shape[2]
    if target == n_frames :
        print("Perfect number of frames !")
        return video
    else :
        if n_frames > target :
            # If number of frames is more than 29, we select
            # 29 evenly distributed frames
            print("Adjusting number of frames")
            idx = np.linspace(0, n_frames-1, 29)
            idx = np.around(idx, 0).astype(np.int32)
            print("Indexes of the selected frames : \n{}".format(idx))
            return video[:, :, idx]
        else :
            # If number of frames is less than 29, duplicate last
            # frame at the end of the video
            output_video = np.zeros((video.shape[0], video.shape[1], 29)).astype(np.uint8)
            output_video[:, :, :n_frames] = video
            for i in range(target-n_frames+1) :
                output_video[:, :, i+n_frames-1] = output_video[:, :, n_frames-1]
            return output_video

def mouthCrop(video) :
    """
    Crop a video around the mouth of the speaker
    Args :
        - 3D numpy array
    Returns :
        - Cropped numpy array
    """
    size = 64
    n_frames = 29
    cropped_video = np.zeros((size, size, n_frames)).astype(np.uint8)

    # For every frame of the image ...
    for i in range(n_frames) :
        # Compute the face locations (right/left eye and nose tip)
        face_locations = face_recognition.face_landmarks(
            video[:, :, i],
            model="small"
        )
        if len(face_locations) == 0 :
            sys.exit("No face detected in frame {}".format(i))
        # To make sure the crop around the mouth just right (not too zoomed
        # in or zoomed out), the distance between the eyes is used as
        # a reference. The leftmost point of the left eye and the rightmost point
        # of the right eye are selected. We then use these
        # values to compute the size of the crop
        left_point = face_locations[0]["left_eye"][0][0]
        right_point = face_locations[0]["right_eye"][1][0]
        crop_size = right_point - left_point
        # The selection is centered on the x axis point of the nosetip
        crop_location_x = face_locations[0]["nose_tip"][0][0]
        crop_location_y = face_locations[0]["nose_tip"][0][1]
        selection = video[
            crop_location_y:crop_location_y+crop_size,
            crop_location_x-(crop_size//2):crop_location_x+(crop_size//2),
            i
        ]
        # Resize to target size
        cropped_video[:, :, i] = cv2.resize(
            selection,
            dsize=(size, size),
            interpolation=cv2.INTER_LINEAR)
    return cropped_video

def reshapeAndConvert(video) :
    """
    Reshape the video to a 4D array before feeding it to the model function
    Also apply normalization to go from [0-255] to [0-1]
    Args :
        - 3D numpy array
    Returns :
        - 4D numpy array
    """
    size = video.shape[0]
    n_frames = video.shape[2]
    video = np.reshape(video, (1, size, size, n_frames)).astype(np.float32)
    return video / 255.0

def create_dict_word_list(path) :
    '''
    Create a dict used to transfrom labels from int to str
    Args :
        - path : Path to the word list
    Return :
        - Python dictionnary {Word : Label}
    '''
    count = 0
    my_dict = dict()
    with open(path+'word_list.txt', 'r') as f:
        for line in f:
            my_dict.update({count : line[:-1]})
            count += 1
    return my_dict

# Debugging function
def _write_video(video, path, fps) :
    writer = cv2.VideoWriter(
        path+".avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (256,256)
    )
    video = video * 255
    for i in range(29) :
        writer.write(
            cv2.resize(
                cv2.cvtColor(
                    video[0, :, :, i].astype('uint8'),
                    cv2.COLOR_GRAY2BGR
                ),
                dsize=(256, 256),
                interpolation=cv2.INTER_LINEAR
            )
        )
    writer.release()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--file',
    default=None,
    help="Name/path of the video file"
)
parser.add_argument(
    '--checkpoint_path',
    default=None,
    help="Path to the checkpoint file"
)
parser.add_argument(
    '--output',
    default=".",
    help="Name of the ouput file"
)
parser.add_argument(
    '--k',
    default="10",
    help="Show top-k predictions"
)


if __name__ == '__main__':
    # Useful stuff
    args = parser.parse_args()
    assert os.path.isfile(args.file), "Video file not found"
    im_size = 64
    n_frames = 29
    params = {"num_classes": 500}
    word_dict = create_dict_word_list("data/")

    # Preprocessing
    print("Reading frames from {}".format(args.file))
    video, n_frames_original, fps = videoToArray(args.file)
    video = frameAdjust(video)
    print("Cropping video around the speaker's mouth (may take time)")
    video = mouthCrop(video)
    video = reshapeAndConvert(video)
    # Used for debugging, not important :
    # fps_output aligns the input video used by the model with the original video
    fps_output = int(fps * (n_frames / n_frames_original)) 
    _write_video(video, args.output, fps_output)

    # Create the classifier
    print("Creating classifier from {}".format(args.checkpoint_path))
    classifier = tf.estimator.Estimator(
        model_fn=inception_model_fn,
        params=params,
        model_dir=args.checkpoint_path
    )
    # Inference time !
    print("Computing predictions")
    predictions = classifier.predict(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            {"x": video},
            batch_size=1,
            shuffle=False
        )
    )

    # Print predictions
    predictions = list(predictions)[0]
    predicted_class = predictions["classes"]
    top_k_classes = (-predictions["probabilities"]).argsort()[:int(args.k)]
    predicted_words = list()
    print("Predictions :")
    for label in top_k_classes :
        predicted_words.append(word_dict[label])
        print("* {} : {} %".format(
            word_dict[label],
            predictions["probabilities"][label]*100
        ))

    # Draw plot and write a .png file
    print("Rendering prediction plot to {}.png".format(args.output))
    idx = [3*i for i in range(int(args.k))]
    plt.figure(figsize=(int(args.k)//2+5, 5))
    plt.bar(
        x=idx,
        height=predictions["probabilities"][top_k_classes],
        color="goldenrod"
    )
    plt.xlabel('Words')
    plt.ylabel('Probabilities')
    plt.title("Most plausible words".format(args.k))
    plt.xticks(idx, predicted_words)
    plt.savefig(
        args.output+".png",
        transparent=False,
        bbox_inches="tight"
    )

    print("Done.")
