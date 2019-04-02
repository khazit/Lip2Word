# CNN

## Lip Reading in the Wild - Joon Son Chung and Andrew Zisserman
* In lip-reading there is a fundamental limitation on performance due to homophemes (sets of words that sound different but involves identical movements of the speaker's lips). Thus they cannot be distinguished using visual information alone ("mark", "park" and "bark" for eg.) **Need to take account of this ambiguity when assessing the performance of the model**
* Challenging problem due to intra-class variations (accents, speed of speaking, etc)
* Using CNNs to recognize individual words from a sequence of lip movements
* The words are not isolated, as is the case in other lip-reading datasets; as a result, there may be co-articulation of the lips from preceding subsequent words. **Might help against the homophemes (brings more "context")**
* 4 networks architectures : Early Fusion (2D (grayscale) and 3D) and Multiple Towers (harder to implement)

#### Details about the training :
* SGD with momentum 0.9 and batch normalisation (but without dropout)
* Decreasing learning rate (from 1E-2 to 1E-4)

#### Results :
* 2D models are far more accurate. Multiple Towers are slightly more accurate than Early Fusion (+- 4%)
* Best top-1 accuracy around 60% (500 different words)

## Large-scale Classification with Convolutional Neural Networks 
https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Karpathy_Large-scale_Video_Classification_2014_CVPR_paper.pdf

#### Architecture :
 * Using shorthand notation, the full architecture is C(96,11,3)-N-P-C(256,5,1)-N-P-C(384,3,1)-C(384,3,1)-C(256,3,1)-P-FC(4096)-FC(4096), where C(d,f,s) indicates a convolutional layer with d filters of spatial size f×f, applied to the input with stride s.FC(n) is a fully connected layer with n nodes. All pooling layers P pool spatially in non-overlapping 2×2 regions and all normalization layers N use the same parameters: k= 2, n= 5, α=10−4, β= 0.5. The final layer is connected to a softmax classifier with dense connections.
 The Early Fusion extension combines in-formation across an entire time window immediately on thepixel level. This is implemented by modifying the filters onthe first convolutional layer in the single-frame model byextending them to be of size 11×11×3×T pixels, where T is some  temporal extent 
 * An effecive approach to speeding up the runtime performance of CNNs is to modify the architecture to contain two separate streams of processing : a context stream that learns features on low-resolution frames and a high-resolution fovea stream that only operates on the middle portion of the frame.
 * The multiresolution architecture aims to strike a compromise by having two seperate streams of processing over two spatial resolutions. The context stream receives the downsampled frames at half the original spatial resolution, while the fovea steam receives the center region at the original resolution.
 * Must ensure that both streams still terminate in a same size layer to be fed to the fully connected layer.

#### Training :
 * SGD, mini-batch (32), momentum of 0.9, weight decay of 0.0005. Initialized with learning rate of 1E-3 (decreasing)

## 3D Convolutional Neural Networks for Human Action Recognition
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.442.8617&rep=rep1&type=pdf

#### Architecture
 1. Hardwired kernels to generate multiple chanels of information (difficult to implement)
 2. 3D convolution with a kernel of size (7x7x3)
 3. 2x2 subsampling layer
 4. 3D convolution with a kernel of size (7x6x3) **Not a square because the input image is 60x40**
 5. 3x3 subsampling
 6. 2D convolution of 7x4
 7. fully connected layer

## Inception
Medium : https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
Paper : https://arxiv.org/pdf/1602.07261.pdf
#### Architecture 
 * Stem architecture : https://cdn-images-1.medium.com/max/1600/1*cYjhQ05zLXdHn363TsPrLQ.jpeg
 * Modules : https://cdn-images-1.medium.com/max/640/1*KrBAIZjcrlXu6JPiPQj2vQ.jpeg
 * Global architecture

## Batch Normalization
https://fr.coursera.org/lecture/deep-neural-network/normalizing-activations-in-a-network-4ptp2
 * Normalize values inside a NN to speed up the learning. 
 * We normalize the values before the activation function. It is done more often. Recommended by Andrew Ng.
 * Two trainable parameters, Gamma and Beta to avoid having a mean=0 variance=1 all the time

# Code optimization

## Data pipeline using tf.data to optimize performance
https://www.tensorflow.org/guide/performance/datasets

#### Pipelining
Pipelining overlaps the preprocessing and model execution of a training step. While the accelerator is performing training step N, the CPU is preparing the data for step N+1 (to reduce idle time)
```python
dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
return dataset
```

## Importing Data - Tensorflow
https://www.tensorflow.org/guide/datasets#basic_mechanics

# RNN
## Reccurent Neural Networks - Coursera
https://fr.coursera.org/lecture/nlp-sequence-models/recurrent-neural-network-model-ftkzt
#### RNN Model
A feature learned in t, can generalize quickly to something in t+i. It's similar to what convolutional neural networks do. Where things learned for one part of the image can generalize quickly for other parts of the image.

#### Different types of RNNs
Input the frames one at a time. Rather than having to produce an output at every time-step we can have the RNN see all the frames and prduce an output at the last time-step. Many-to-one architecture.