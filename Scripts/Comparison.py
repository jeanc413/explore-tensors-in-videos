# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:19:40 2020

@author: JC
"""
# %% import packages
import pickle
import glob
import TuckerFunction
import kmeans
import tensorly as tl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from os import chdir
from sklearn.metrics.cluster import normalized_mutual_info_score


# %% Create formatting and pre-processing functions
def labeling(number):
    number = int(number)
    if number <= 5:
        label = "DarkSteelSink1"
    elif number <= 8:
        label = "LightSteelSink1"
    elif number <= 11:
        label = "CeramicSink1"
    elif number <= 14:
        label = "SmallCeramicSink6"
    elif number <= 17:
        label = "LightGraniteSink1"
    elif number <= 25:
        label = "CeramicSink2"
    else:
        label = "ERROR"
    return label


def reformat_string(data):
    temp = np.empty(len(data), dtype=object)
    for k in range(len(data)):
        temp[k] = labeling(data[1][k])
    data["Type"] = temp
    data["Step"] = data[3]
    for k in range(6):
        del data[k]
    return data


def standardize(tensor_list):
    core_list = [tensor_list[i].core for i in range(len(tensor_list))]
    mean = tl.mean(core_list, axis=0)
    std = tl.sqrt(((core_list - mean)**2)/len(tensor_list)-1)
    standard_score = (core_list - mean)/std
    for i in range(len(tensor_list)):
        tensor_list[i].core = standard_score
    return tensor_list


def unit_length_scaling(tensor_list):
    core_list = [tensor_list[i].core for i in range(len(tensor_list))]
    for i in range(len(tensor_list)):
        tensor_list[i].core = core_list[i]/tl.norm(core_list[i])
    return tensor_list


def min_max_scaling(tensor_list):
    core_list = [tensor_list[i].core for i in range(len(tensor_list))]
    min_core = tl.min(core_list, axis=0)
    max_core = tl.max(core_list, axis=0)
    core_list = (core_list - min_core)/(max_core - min_core)
    for i in range(len(tensor_list)):
        tensor_list[i].core = core_list[i]
    return tensor_list


# %% WARNING Change current directory to decomposition folder
chdir("Decompositions")
# %% Load SubTensors objects
name_list = glob.glob('*')
name_list = [n for n in name_list if "HandWash_" in n]
Tensor_List = []
for tensor in name_list:
    Handler = open(tensor, "rb")
    Tensor_List.append(pickle.load(Handler))

n = len(Tensor_List)

for i in range(n):
    Tensor_List[i].core = np.asarray(Tensor_List[i].core)

# %% Contingency table
Clusters = kmeans.KMeans(Tensor_List, k=6).predict()
Table = pd.DataFrame()
Table["Type"] = reformat_string(pd.Series(name_list).str.split("_", expand=True))["Type"]
Table["Clusters"] = Clusters
Contingency = pd.crosstab(Table["Clusters"], Table["Type"], margins=True)
print("Normalized Mutual Information", normalized_mutual_info_score(Table["Clusters"], Table["Type"]))
print("Contingency Table", Contingency, sep="\n")


# %% Inverse engineering the decomposition
print(Tensor_List[0])
print(np.shape(Tensor_List[0].factors[0]))
print(np.shape(Tensor_List[0].factors[1]))
print(np.shape(Tensor_List[0].factors[2]))
print(np.shape(Tensor_List[0].factors[3]))
print(Tensor_List[0].factors[3])
print(Tensor_List[0].core)

