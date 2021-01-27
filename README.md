# Explore tensors in videos

This projects explores the idea of using tensors in videos. It uses Tucker decomposition to transform videos into a lower dimensional space. More specifically it utilizes the core matrix of tucker decomposition  as compressed version of the video. The code in this repository takes videos, decomposes them and then it utilizes K-means clustering to group the cores. The aim is to check if this lower dimensional spaces are good representations of videos.


### Prerequisites

Before running the code some libraries have to be installed

Tensorly: a library used to compute the decompostions.

With pip 
```
pip install -U tensorly
```

With conda

```
conda install -c tensorly tensorly
```

OpenCV: Computer vision library used to manipulate videos.

With pip 
```
pip install opencv-python
```

Numpy: mathematical library
With pip 
```
pip install numpy
```

## Running decompositions

First all files must be in the same directory. Videos must be put into a separate Directory called Videos. ie. DIRECTORY_PATH\Videos.

The next step is to run the file Tensor_Decomposition.py. This file will compute the tucker decomposition for all videos. By default it performs a rankn = (32,32,32) Gray Scale Decomposition at 200 frames.

The previous file makes use of TuckerFunction.py, which is the main file at which decompositions are runned. Particulary we use the function video_tensor_decomposer()
```python

def video_tensor_decomposer(video_address,
                            tensor_length=200,
                            rank=(2, 2, 2, 2),
                            decomposer="tucker",
                            gray=True,
                            seed_state=None,
                            max_frames=None):
```
As you can see you can define several parameters mainly:

Tensor Length: First we do a dimensionality reduction on the video, so we only choose n number of randoms frames from the videos, given by this parameter

Rank: The Rank of the decomposition

Decomposer: The decompostion method to be used (Tucker or CP) 

Gray: Boolean value indicating if we want to transform the video into Gray Scale. This allows to choose between the Full colored videos, or a gray-scaled version to reduce dimensionality


After the decompostion is done the Core Tensor and Factor Matrices (Given that the decomposition is Tucker) are stored in an object of class SubTensor. One can then call self.core or self.factors respectively to get the core  and factor matrices of the tensor.

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

## Built With

* [Tensorly](http://tensorly.org/) - Simple and Fast Tensor Learning in Python
* [OpenCV](https://opencv.org/) - Open Source computer vision


## Authors

* **Jean Carlos Fernandez** - *Author* 
* **Abdelrahman Elmorsy** - *Author* 
* **Javier Jaquez** - *Author* 


