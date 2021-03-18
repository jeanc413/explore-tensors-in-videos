# Explore tensors in videos

This project explores the idea of using tensors in videos. 
It uses Tucker decomposition to transform videos into a lower dimensional space. 
In simpler words, it utilizes the core matrix of tucker decomposition  as a compressed 
version of the video. The code in this repository takes videos, 
decomposes them into Tucker Cores and Factors,
and then uses *K-means* clustering to group the cores. 
The aim is to check if this lower dimensional spaces 
are good representations of videos.


### Prerequisites

Before running the code some libraries have to be installed. We will
simply provide a list of packages to install in a format of *required*
and *optional* packages. For installation use the package manager 
in your local installation.

## Required packages
This list of packages are required to replicate the experiment:

* [Tensorly](http://tensorly.org/) - Library used to deploy 
  tensors framework and compute tensor decompositions.
* [OpenCV](https://opencv.org/) - Computer vision library for video 
  manipulation.
* [Numpy](https://numpy.org/) - Mathematical computation library 
  for multidimensional arrays.
* [Matplotlib](https://matplotlib.org) Python-Numpy Plotting library.
* [Pandas](https://pandsa.pydata.org) - Data analysis 
  and manipulation tool.
* [Seaborn](https://seaborn.pydata.org) - Matplotlib based 
  library for statistical plotting.
* [Scikit-Learn](https://scikit-learn.org) - Machine Learning tools library.

## Optional packages
These were used as optional infrastructure to accelerate certain 
computation, overall if a compatible GPU device was available:

* PyTorch: Machine Learning library used on irregular input data.

## Running decompositions

First all files must be in the same directory. Videos must be put into a separate Directory called Videos. ie. DIRECTORY_PATH\Videos.

The next step is to run the file Tensor_Decomposition.py. This file will compute the tucker decomposition for all videos. By default it performs a rankn = (32,32,32) Gray Scale Decomposition at 200 frames.

The previous file makes use of TuckerFunction.py, which is the main file at which decompositions are runned. Particulary we use the function video_tensor_decomposer()
```python

def video_tensor_decomposer(video_address,
                            tensor_length=200,
                            rank=(32, 32, 32),
                            decomposer="tucker",
                            gray=True,
                            seed_state=None,
                            max_frames=None)
    """
    This is a wrapper function to generate decompositions of videos by exploiting their structure as tensors.

    Parameters
    ----------
        video_address: string
            Video path location.
        tensor_length: int
            Integer for frames quantity to be selected to generate the tensor.
        rank: tuple or list
            tuple or list  object with the n-rank to use during the decomposition.
        decomposer: {tucker, parafac} or tensorly.decomposition fun
            Used decomposition algorithm from `tensorly` library.
        gray: bool
            Boolean telling if the video should be converted to grayscale by cv2 library.
        seed_state: numpy.random.RandomState
            Seed state used to generate the frames sample. If None, a random initial seed is to be started.
        max_frames: int
            Integer telling if there's a maximum amount of frames to be used from video length. If None,
            it will be set as the maximum number of frames of the video.

    Returns
    -------
    Returns TuckerFunction.SubTensor class object with the video decomposition for the given parameters.
    """
```

After the decomposition is done the Core Tensor and 
Factor Matrices (Given that the decomposition is Tucker) 
are stored in an object of class SubTensor. 
One can then call self.core or self.factors respectively 
to get the core  and factor matrices of the tensor. 
For more information, feel free to read the class or functions 
documentation on script, or console (by typing
`TuckerFunction.SubTensor?` after import).

## Running the tests

The file `Comparison.py` contains the *clustering* experiment 
using our defined `nranks` for the tensors decompositions. It's 
important to start this script's working directory on it's same path.
Since the used *k-means* algorithm was a self-implementation to 
work with our SubTensor object class, it's required to import 
(therefore have in the same path when starting the script)
`TuckerFunction.py` and `kmeans.py`.

This script will run the experiment and generate appropriate
summaries for each provided decomposition configuration.
This repository provides 3 defined decompositions. If users
want to redefine this, please modify the `Comparison.py` cells 
for `decompositions_features` and `decompositions_paths`.
The detailed summary is provided via console prints, and the plots 
are stored in memory.

### Break down into end-to-end tests

For example results please read the attached
paper ***Video_Analysis_using_Tensors.pdf***, where the experiments
are summarized in depth, and the interpretation of result is explained.

## Authors

* **Jean Carlos Fernandez** - *Author* 
* **Abdelrahman Elmorsy** - *Author* 
* **Javier Jaquez** - *Author* 
