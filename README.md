# Multi-Person Action Recognition
This repository contains the Computer Vision and Cognitive Systems exam project.

## Project contraints
The project must be composed of 4 parts:
1. Classical Image Processing task
2. Neural Network (creation and training)
3. Retrieval task
4. Geometry task

## Abstract
<p style="text-align: justify;">
<em>Multi-Person Action Recognition is the task of recognizing the action of multiple people present in a frame. With the following paper, we present an approach to solve this problem by combining people detection, tracking and action recognition networks. Specifically, YOLOv7 will be employed for extracting the bounding boxes, then, DeepSORT will be used to keep track of the same individuals during multiple frames and, lastly, by employing the bounding boxes obtained, a custom 3D-fused 2 stream architecture will be used to predict the actions. Furthermore, once a person has engaged in a harmful deed, its bounding box will be saved into blacklist where its face will be later extracted through a HOG face detector. Then, it will be possible to understand, thanks to face recognition, whether the dangerous people discovered have been already detected in other recordings. 
After performing multiple tests on real-world data, we concluded that, besides the architectures used, the overall approach works. In particular, we think that because it is not performed in an end-to-end way, if more accurate architectures will be unveiled in the future, they could be applied on our pipeline easily, thus, boosting the overall accuracy.</em>
</p>

## Project parts done by me
* Tracking (combination with the detector and evaluation on TrackEval)
* Designed (following the paper), written, trained all the components of the Action Recognition Network
* Adapted the HMDB51 dataset for our task
* Performed classical image processing task on the dataset (Gaussian Filter)
* Written all the code that combines the detector (YOLOv7), tracker (DeepSORT) and the Action Recognition Network (3D-fused 2 stream architecture)

My code is in 'Multi-Person Action Recognition' and 'Action Recognition Architecture-Training-Dataset' folder. I didn't take any part in the making of 'geometry' and 'retrieval' folders.

## How to test the Action Recognition model
Firstly, download the Multi-Person Action Recognition folder which contains the code that combines the detector, tracking and action recognition networks. In particular, there are 2 different implementations:
1. official implementation.py\
It represents the version that best performed on our dataset.
2. official implementation (ensemble version).py\
It represents the ensemble version.

Once you have decided which version you prefer to use, follow these instructions:
1. Open Anaconda Prompt
2. Create a conda environment: "conda create -n &lt;name&gt; python=3.8.3"
3. Activate it: "conda activate &lt;name&gt;"
4. Install the requirements in the folder: "pip install -r requirements.txt"
5. Run the code to understand the possible parameters: "python 'official implementation (ensemble version).py' --help" OR "python 'official implementation.py' --help"

where &lt;name&gt; needs to be substituted with a conda name of your choice. 

## Important Note
1. It is strictly prohibited to use our code/paper and other informations present in this repository outside of testing our task.  
2. Trained weights have been kept private.

## References
[1] Matteo Fabbri, Guillem Brasó, Gianluca Maugeri, Orcun
Cetintas, Riccardo Gasparini, Aljoša Ošep, Simone
Calderara, Laura Leal-Taixé, Rita Cucchiara - MOTSynth:
How Can Synthetic Data Help Pedestrian Detection and
Tracking?\
[2] Schroff, Florian, Dmitry Kalenichenko, and James
Philbin. "Facenet: A unified embedding for face recognition
and clustering." Proceedings of the IEEE conference on
computer vision and pattern recognition. 2015.\
[3] Karen Simonyan, Andrew Zisserman - Two-Stream
Convolutional Networks for Action Recognition in Videos\
[4] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, T. Serre -
HMDB: A Large Video Database for Human Motion
Recognition\
[5] Fabbri, M., Lanzi, F., Gasparini, R., Calderara, S.,
Baraldi, L., & Cucchiara, R. (2020). Inter-homines:
Distance-based risk estimation for human safety. arXiv
preprint arXiv:2007.10243.\
[6] Nicolai Wojke, AlexBewley, Dietrich Paulus -
DeepSORT - https://github.com/nwojke/deep_sort\
[7] Christoph Feichtenhofer, Axel Pinz, Andrew Zisserman -
Convolutional Two-Stream Network Fusion for Video Action
Recognition\
[8] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo
Torresani, Manohar Paluri - Learning Spatiotemporal
Features with 3D Convolutional Networks\
[9] Zhaofan Qiu, Ting Yao, Tao Mei - Learning Spatio-
Temporal Representation with Pseudo-3D Residual Networks\
[10] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao,
Dahua Lin, Xiaoou Tang, Luc Van Gool - Temporal Segment
Networks for Action Recognition in Videos\
[11] Yang, Shuo, et al. "Wider face: A face detection
benchmark." Proceedings of the IEEE conference on
computer vision and pattern recognition. 2016.

## Acknowledgement
I want to personally thank my University that gave me access to their GPUs to train the networks. 