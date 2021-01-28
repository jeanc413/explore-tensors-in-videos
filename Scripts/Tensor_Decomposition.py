#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from TuckerFunction import video_tensor_decomposer
import pickle  # For file writing
import glob  # For File searching

# %% Chage into appropiate path contaning videos samples
video_list = glob.glob('Videos/*.MOV')  # Reading all files with MOV extension
video_list2 = glob.glob('Videos/*.mp4')

# %% Loop for object creation

seed = np.random.RandomState(123)  # reproducible seed state

i = 0  # First List Index
if video_list:
    for video in video_list:
        decomposition = video_tensor_decomposer(video_list[i],
                                                tensor_length=50, 
                                                rank=(2,2,2,2),
                                                seed_state=seed)
        pickle_out = open(video_list[i].replace('.MOV', '')
                          .replace('Videos/','Decompositions/'), "wb")
        pickle.dump(decomposition, pickle_out)
        pickle_out.close()
        video_list.append(video_list[i].replace('.MOV', '')
                          .replace('Videos/','Decompositions/'))
        i = i + 1

j = 0    # Second list index
if video_list2:
    for video in video_list2:
        decomposition = video_tensor_decomposer(video_list2[j],
                                                tensor_length=50,
                                                rank=(2,2,2,2),
                                                seed_state=seed)
        pickle_out = open(video_list2[j].replace('.mp4', '')
                          .replace('Videos/','Decompositions/'), "wb")  # We use Main index for naming
        pickle.dump(decomposition, pickle_out)
        pickle_out.close() 
        video_list.append(video_list2[j].replace('.mp4', '')
                          .replace('Videos/','Decompositions/'))
        j = j + 1

# %% Saving List as an object
pickle_out = open('video_list', "wb")  # We use Main index for naming
pickle.dump(video_list, pickle_out)
pickle_out.close()


# %% Reading List

# pickle_in = open("video_list","rb")
# list = pickle.load(pickle_in)
# pickle_in.close()
