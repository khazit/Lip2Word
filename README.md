# lipReader

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
 * An effecive approach to speeding up the runtime performance of CNNs is to modify the architecture to contain two separate streams of processing : a context stream that learns features on low-resolution frames and a high-resolution fovea stream that only operates on the middle portion of the frame.
 * The multiresolution architecture aims to strike a compromise by having two seperate streams of processing over two spatial resolutions. The context stream receives the downsampled frames at half the original spatial resolution, while the fovea steam receives the center region at the original resolution.
 * Must ensure that both streams still terminate in a same size layer to be fed to the fully connected layer.

#### Training :
 * SGD, mini-batch (32), momentum of 0.9, weight decay of 0.0005. Initialized with learning rate of 1E-3 (decreasing)