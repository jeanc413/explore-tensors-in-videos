# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:59:15 2020
Updated on Mon Jan 18

@author: JC
"""
# %% module dependencies
import cv2
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import time
try:
    tl.set_backend("pytorch")
except ValueError:
    tl.set_backend("numpy")
backend_test = False


# %% define Tensor subset class
class SubTensor:
    def __init__(self, video_array, rng, video_name, rank, decomposer="tucker"):
        """
        Class object to wrap video tensor decomposition and identification parameters

        Parameters
        ----------
            video_array: ndarray
                Array containing the sampled frames.
            rng: numpy.array
                Array containing the sampled frames number for reproducibility.
            video_name: string
                String containing the name of the video
            rank: tuple or list
                tuple or list  object with the n-rank to use during the decomposition.
            decomposer: {tucker, parafac} or tensorly.decomposition function
                Used decomposition algorithm from `tensorly` library.

        Returns
        -------
        Returns TuckerFunction.SubTensor class object with the video decomposition for the given parameters.
        """
        self.decomposer = decomposer.lower()
        self.video_name = video_name
        self.rng = rng
        self.rank = rank
        if 'numpy' == tl.get_backend():
            subset = tl.tensor(video_array).astype('d')
        elif 'pytorch' == tl.get_backend():
            subset = tl.tensor(video_array).double()
        else:
            tl.set_backend("numpy")
            subset = tl.tensor(video_array).astype('d')
        self.__class__ = SubTensor

        try:
            method = getattr(tl.decomposition, self.decomposer)
            self.core, self.factors = method(subset, rank=rank)
        except AttributeError:
            print("Tensorly has no attribute "+str(self.decomposer))
            self.core, self.factors = (None, None)
            self.__class__ = None


# %% define Tucker Decomposition Function
def video_tensor_decomposer(video_address,
                            tensor_length=200,
                            rank=(2, 2, 2, 2),
                            decomposer="tucker",
                            gray=True,
                            seed_state=None,
                            max_frames=None):
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
    # Checking if decomposers are on parameter
    if decomposer not in ("tucker", "parafac"):
        print("Decomposition technic not in the common parameters.")
        print("SubTensor class will try to use this decomposer.")

    # Create video capture device
    video_capture = cv2.VideoCapture(video_address)

    # Check how long the video will be
    if max_frames is None:
        max_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Checks or sets random seed and store the sample index selection
    if seed_state is None:
        seed_state = np.random.RandomState(int(time.time()))
    rng = np.sort(seed_state.choice(range(0, max_frames), tensor_length, replace=False))

    # Initialize empty array
    frames_array = []

    # Load frames, first checks if gray-scale required
    if gray:
        for i in rng:
            video_capture.set(cv2.cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_array.append(frame)
    else:
        for i in rng:
            video_capture.set(cv2.cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video_capture.read()
            if not ret:
                break
            frames_array.append(frame)

    # release the video capture
    video_capture.release()
    cv2.destroyAllWindows()

    # release capture video from memory
    del video_capture

    # make tensor and decomposition (See class "sub_tensor" disambiguation up)
    tensor_decomposition = SubTensor(video_array=np.asarray(frames_array),
                                     rng=rng,
                                     video_name=video_address,
                                     rank=rank,
                                     decomposer="tucker")

    # return the tensor object
    return tensor_decomposition
