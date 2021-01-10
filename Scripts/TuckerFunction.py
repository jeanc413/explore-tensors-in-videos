# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:59:15 2020

@author: JC
"""
# %% import packages
import cv2
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import time


# %% define Tensor subset class
class SubTensor:
    def __init__(self, video_array, rng, video_name):
        self.video_name = video_name
        self.rng = rng
        subset = tl.tensor(video_array).astype('d')
        self.core, self.factors = tucker(subset, rank=[2, 2, 2, 2])


# %% define Tucker Decomposition Function
def tucker_function(video_address, tensor_length=50, seed_state=None, max_frames=None):
    """
    INPUTS:
    video_capture: an OpenCV VideoCapture object whose frames we want to read
    max_frames: the maximum number of frames we want to read
    
    OUTPUT:
    array of all the frames until max_frames
    """
    video_capture = cv2.VideoCapture(video_address)
    # Check how long the video will be
    if max_frames is None:
        max_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize empty array
    frames_array = []
    
    # Keep track of the frame number
    frame_nb = 0
    
    # iterate through the frames and append them to the array
    while video_capture.isOpened() and frame_nb < max_frames:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames_array.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_nb += 1
    
    # release the video capture
    video_capture.release()
    cv2.destroyAllWindows()
    
    # release video from memory 
    del video_capture

    # set fixed seed for reproducibility
    if seed_state is None:
        seed_state = np.random.RandomState(int(time.time()))
    
    # subset array by taking frame sample
    rng = seed_state.choice(range(0, max_frames), tensor_length)
    frames_array = np.asarray(frames_array)[rng, :, :, :]
    
    # make tensor and decomposition (See class "sub_tensor" disambiguation up)
    tensor_decomposition = SubTensor(video_array=frames_array, rng=rng, video_name=video_address)

    # return the tensor object 
    return tensor_decomposition


# %% Testing function and class
if __name__ == '__main__':
    from os import chdir
    seed = np.random.RandomState(123)
    chdir("C:/Users/JC/Documents/MEGAsync/TrierDataScienceMaster/Research Project/Scripts/Data")
    
    t_i = time.time()
    parking_lot_Object = tucker_function('parking_lot.MOV', seed_state=seed, max_frames=314)
    patio_Object = tucker_function('patio.MOV', seed_state=seed, max_frames=314)
    commute_Object = tucker_function('commute.MOV', seed_state=seed, max_frames=314)
    duration = time.time()-t_i
    print(duration)

# run of 3 videos lasted 186.35 with this approach
# run of 3 videos lasted 268.011 with paper approach
# random_frames=parking_lot_Object.rng










