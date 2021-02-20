# Environmental-Sound-Classification

###### [Overview](#esc-50-dataset-for-environmental-sound-classification) | [Dataset](#Dataset) | [Repository content](#Repository-Content) | [Results](#Results-Achieved) | [References](#References)

<img src="example_clip.png" alt="ESC-50 clip preview" title="ESC-50 clip preview" align="right" width=350 />

The main purpose of this project will be to provide an efficient way, using machine learning techniques, to classify environmental sound clips belonging to one of the only public available dataset on the internet. <br>
Several approaches have been tested during the years, but only a few of them were able to reproduce or even overcome the human classification accuracy, that was estimated around 81.30%. <br>
The analysis will be organized in the following way: since the very first approaches were maily focused on the examination of audio features that one could extract from raw audio files, we will provide a way to collect and organize all those "vector of features" and use them to distinguish among different classes. Then, different classification architectures and techniques will be implemented and compared among each other, in order also to show how they react to different data manipulation (overfitting, numerical stability,...). 
In the end, it will be shown that all those feature classifiers, without exceptions, underperform when compared to the results provided by the use of Convolutional Neural Networks directly on audio signals and relative spectrograms (so without any kind of feature extraction), and how this new approach opened for a large number of opportunities in term of models with high accuracy in sound classification.

### Dataset

### Repository Content

### Results Achieved

### References
