# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:19:40 2020

@author: JC
"""
# %% import packages
from os import chdir
import pickle
import glob
import TuckerFunction
import tensorly as tl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import kmeans
from time import time


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
    for k in range(len(tensor_list)):
        tensor_list[k].core = core_list[k]
    return tensor_list


# %% Set experiment evaluation
np.random.seed(123)
iterations = 100
digits = 2
decompositions_features = ["gray scale, rank=(32, 32, 32), frames=200",
                           "BGR, rank=(10, 10, 10, 2), frames=100",
                           "BGR, rank=(2, 2, 2, 2), frames=50"]
decompositions_paths = ["Decompositions", 
                        "Decompositions2",
                        "Decompositions3"]  # Folders where distinct decompositions are stored

print("Running a total of", iterations, "iterations")

for index, path in enumerate(decompositions_paths):
    # Change path into right folder
    chdir(path)

    # Load SubTensors objects
    name_list = glob.glob('*')
    name_list = [n for n in name_list if "HandWash_" in n]

    if not name_list:
        print("Path provided has no tensor decompositions.")
        break

    Tensor_List = []
    for tensor in name_list:
        Handler = open(tensor, "rb")
        Tensor_List.append(pickle.load(Handler))

    for j in range(len(Tensor_List)):
        Tensor_List[j].core = np.asarray(Tensor_List[j].core)

    section_break = "-------------------------------------"
    print(section_break)
    print("Results for", decompositions_features[index])

    # Confusion matrix for Type
    score = []
    convergence = []
    duration = []
    for _ in range(iterations):
        t_i = time()
        algorithm = kmeans.KMeans(Tensor_List, k=6)
        Clusters = algorithm.predict(verbose=True)
        Table = pd.DataFrame()
        Table["Type"] = reformat_string(pd.Series(name_list).str.split("_", expand=True))["Type"]
        Table["Clusters"] = Clusters
        temp_score = metrics.adjusted_mutual_info_score(Table["Type"], Table["Clusters"])
        if not score or temp_score > np.max(score):
            Contingency_Type = pd.crosstab(Table["Clusters"], Table["Type"], margins=True)
        score.append(temp_score)
        convergence.append(algorithm.iterations)
        duration.append(time()-t_i)
    # Print clustering  results
    print("Sink-Type results")
    print("Convergence average required steps", np.mean(convergence).round(digits))
    print("Average Adjusted Mutual Information", np.mean(score).round(digits))
    print("Standard Deviation AMI", np.std(score).round(digits))
    # noinspection PyUnboundLocalVariable
    print("Best contingency table", "score = "+str(np.max(score).round(digits)),
          Contingency_Type, sep="\n")
    best_score_type = np.max(score)
    print("Average duration", np.mean(duration).round(digits))
    print("Run lasted", np.sum(duration).round(digits))
    print()
    
    # Confusion matrix for Type
    score = []
    convergence = []
    duration = []
    for _ in range(iterations):
        t_i = time()
        algorithm = kmeans.KMeans(Tensor_List, k=3)
        Clusters = algorithm.predict(verbose=True)
        Table = pd.DataFrame()
        Table["Step"] = reformat_string(pd.Series(name_list).str.split("_", expand=True))["Step"]
        Table["Clusters"] = Clusters
        temp_score = metrics.adjusted_mutual_info_score(Table["Step"], Table["Clusters"])
        if not score or temp_score > np.max(score):
            Contingency_Step = pd.crosstab(Table["Clusters"], Table["Step"], margins=True)
        score.append(temp_score)
        convergence.append(algorithm.iterations)
        duration.append(time()-t_i)
    # Print clustering  results
    print("Step-number results")
    print("Convergence average required steps", np.mean(convergence).round(digits))
    print("Average Adjusted Mutual Information", np.mean(score).round(digits))
    print("Standard Deviation AMI", np.std(score).round(digits))
    print("Best contingency table", "score = "+str(np.max(score).round(digits)),
          Contingency_Step, sep="\n")
    best_score_step = np.max(score)
    print("Average duration", np.mean(duration).round(digits))
    print("Run lasted", np.sum(duration).round(digits))

    #  Adding plots respect to the confusion matrix of cluster per true label
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    # Define max values for temperature range on heatmap.
    max_type_margin = min(Contingency_Type.T.All.min(), 
                      Contingency_Type.All.min())
    max_steps_margin = min(Contingency_Step.T.All.min(),
                           Contingency_Step.All.min())
    
    # Plotting heatmaps
    sns.heatmap(Contingency_Type, 
                ax=axes[0], 
                vmin=-0, 
                vmax=max_type_margin, 
                annot=True, 
                linewidths=0.5)
    axes[0].set_title("SinkType AMI="+str(best_score_type.round(digits)))

    sns.heatmap(Contingency_Step, 
                ax=axes[1],
                vmin=0,
                vmax=max_steps_margin,
                robust=True,
                annot=True,
                linewidths=0.5)
    axes[1].set_title("StepNumber AMI="+str(best_score_step.round(digits)))
    fig.tight_layout(pad=2)
    fig.suptitle(decompositions_features[index])
    plt.show()

    # Go back on path
    chdir("..")
    print("\n\n")
    
