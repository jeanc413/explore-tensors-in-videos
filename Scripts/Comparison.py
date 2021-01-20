# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:19:40 2020

@author: JC
"""
# %% import packages
import pickle
import glob
import tensorly as tl
from TuckerFunction import video_tensor_decomposer, SubTensor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% define Tensor subset class
name_list = glob.glob('*')
name_list = [n for n in name_list if "HandWash_" in n]
Tensor_List = []
for tensor in name_list:
    Handler = open(tensor, "rb")
    Tensor_List.append(pickle.load(Handler))

n = len(Tensor_List)

Comparisons = pd.DataFrame(columns=["Left", "Right", "Difference"], dtype=float)

for i in range(n):
    for j in range(n):
        if i == j:
            Append = pd.Series([name_list[i], name_list[j], 0], index=Comparisons.columns)
            Comparisons = Comparisons.append(Append, ignore_index=True)
        else:
            Append = pd.Series([name_list[i], name_list[j], tl.norm(Tensor_List[i].core - Tensor_List[j].core)],
                               index=Comparisons.columns)
            Comparisons = Comparisons.append(Append, ignore_index=True)


def labeling(number):
    number = int(number)
    label = ""
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
        temp[k] = labeling(data[1][i])
    data["Type"] = temp
    data["Step"] = data[3]
    for k in range(6):
        del data[k]
    return data


Left = reformat_string(Comparisons["Left"].str.split("_", expand=True))
Right = reformat_string(Comparisons["Right"].str.split("_", expand=True))

Comparisons["Left Type"] = Left["Type"]
Comparisons["Left Step"] = Left["Step"]
Comparisons["Right Type"] = Right["Type"]
Comparisons["Right Step"] = Right["Step"]

# %% View Results

# Histogram of difference
plt.hist(Comparisons["Difference"], color='pink', ec='black')
plt.title("Histogram of differences between videos")
plt.show()

# Boxplot 
Filter = (Comparisons["Difference"] != 0) & (Comparisons["Right Type"] == Comparisons["Left Type"])
sns.boxplot(data=Comparisons[Filter], y="Right Type", x="Difference").set_title(
    "Boxplot for difference between same type of sink")
plt.show()

# Full comparison matrix
Comp_Matrix = pd.pivot_table(Comparisons, values='Difference', index='Left', columns='Right', aggfunc=np.mean)

# Comparing all from Step 01
Filter = (Comparisons["Left Step"] == '01') & (Comparisons["Right Step"] == '01')
Comp_Step_01 = pd.pivot_table(Comparisons[Filter], values='Difference', index='Left', columns='Right', aggfunc=np.mean)

# Comparing all from DarkSteelSink1
Filter = (Comparisons["Left Type"] == 'DarkSteelSink1') & (Comparisons["Right Type"] == 'DarkSteelSink1')
Comp_DarkSteelSink1 = pd.pivot_table(Comparisons[Filter], values='Difference', index='Left', columns='Right',
                                     aggfunc=np.mean)

# Comparing between steps
Comp_Step = pd.pivot_table(Comparisons, values='Difference', index='Left Step', columns='Right Step', aggfunc=np.mean)

# Comparing between Types
Comp_Type = pd.pivot_table(Comparisons, values='Difference', index='Left Type', columns='Right Type', aggfunc=np.mean)

# Heatmaps
sns.heatmap(Comp_Matrix, linewidths=0.5, yticklabels=True, xticklabels=True)
sns.heatmap(Comp_Type, linewidths=1)
sns.heatmap(Comp_Step, linewidths=1)

for step in ["01", "02", "04"]:
    Filter = (Comparisons["Left Step"] == step) & (Comparisons["Right Step"] == step)
    Comp_Step_Type = pd.pivot_table(Comparisons[Filter], values='Difference', index='Left Type', columns='Right Type',
                                    aggfunc=np.mean)
    sns.heatmap(Comp_Step_Type, linewidths=0.5).set(title="Type of sink average difference from step " + step)
    plt.show()

All_Types = pd.unique(Comparisons["Right Type"])

for Type in All_Types:
    Filter = (Comparisons["Left Type"] == Type) & (Comparisons["Right Type"] == Type)
    Comp_Type_Step = pd.pivot_table(Comparisons[Filter], values='Difference', index='Left Step', columns='Right Step',
                                    aggfunc=np.mean)
    sns.heatmap(Comp_Type_Step, linewidths=0.5).set(title="Step average difference from " + Type)
    plt.show()

# %% Inverse engineering the decomposition
print(Tensor_List[0])
print(np.shape(Tensor_List[0].factors[0]))
print(np.shape(Tensor_List[0].factors[1]))
print(np.shape(Tensor_List[0].factors[2]))
print(np.shape(Tensor_List[0].factors[3]))
print(Tensor_List[0].factors[3])
print(Tensor_List[0].core)

