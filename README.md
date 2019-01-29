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

