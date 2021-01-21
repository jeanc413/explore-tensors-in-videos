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
from tensorly.decomposition import parafac
try:
    tl.set_backend("pytorch")
except ValueError:
    tl.set_backend("numpy")
backend_test = False


# %% define Tensor subset class
class SubTensor:
    def __init__(self, video_array, rng, video_name, rank, decomposer="tucker"):
        self.decomposer = decomposer.lower()
        self.video_name = video_name
        self.rng = rng
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
                            tensor_length=50,
                            rank=(2, 2, 2, 2),
                            decomposer="tucker",
                            gray=False,
                            seed_state=None,
                            max_frames=None):
    """
    This is a wrapper function to generate tucker and cp decompositions of videos by transforming them as tensors.

    Parameters
    ----------
        video_address: string
            Path where video is located
        tensor_length: int
            Integer telling how many frames are to be selected to generate the tensor
        rank: tuple or list
            tuple or list with the n-rank to use during the decomposition
        decomposer: {tucker, parafac} or tensorly.decomposition fun
            Used decomposition algorithm from `tensorly` library
        gray: bool
            Bool telling if the video should be converted to grayscale
        seed_state: numpy.random.RandomState
            Seed state used to generate the frames sample. If None, a random initial seed is to be started.
        max_frames: int
            Integer telling if there's a maximum amount of frames to be used from video length. If None,
            it's to be set as the maximum number of frames of the video.

    Returns
    -------
    Returns system integrated points for the provided mesh.
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


# %% Testing function and class
if __name__ == '__main__':
    from os import chdir
    import time
    seed = np.random.RandomState(123)
    chdir("C:/Users/JC/Documents/MEGAsync/TrierDataScienceMaster/Research Project/Scripts/Data")

    t_i = time.time()
    video_tensor_decomposer('parking_lot.MOV', tensor_length=10, seed_state=seed, max_frames=314)
    video_tensor_decomposer('patio.MOV', tensor_length=10, seed_state=seed, max_frames=314)
    video_tensor_decomposer('commute.MOV', tensor_length=10, seed_state=seed, max_frames=314)
    duration = time.time()-t_i
    print("Three tensors decomposition lasted", duration)

    test_gray = video_tensor_decomposer('parking_lot.MOV',
                                        tensor_length=10, gray=True, seed_state=seed, max_frames=314)
    test_cp = video_tensor_decomposer('parking_lot.MOV',
                                      tensor_length=10, decomposer="parafac", seed_state=seed, max_frames=314)

    if test_gray != (None, None):
        print("Gray-scale color reduction successful run")

    if test_cp != (None, None):
        print("CP decomposer successful run")

# run of 3 videos lasted 186.35 with this approach
# run of 3 videos lasted 268.011 with paper approach
# random_frames=parking_lot_Object.rng

# %% Backend performance checker
if __name__ == '__main__' and backend_test:
    from os import chdir
    import time
    seed = np.random.RandomState(123)
    chdir("C:/Users/JC/Documents/MEGAsync/TrierDataScienceMaster/Research Project/Scripts/Data")
    backends = ('numpy', 'pytorch')
    sample_size = 3
    average_time = {}
    for backend in backends:
        t_i = time.time()
        for _ in range(sample_size):
            video_tensor_decomposer('parking_lot.MOV',
                                    tensor_length=10,
                                    seed_state=seed)
        average_time[backend] = (time.time()-t_i)/sample_size
    print(average_time)
