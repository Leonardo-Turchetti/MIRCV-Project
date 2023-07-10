# TensorFlow
from ctypes import sizeof
from unittest import result
import tensorflow as tf
#import tensorflow_hub as hubA

# Utilities
import os
import math
from os import path
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import time
from flask import jsonify
from PIL import Image
import io
import base64
import pickle
from VPT import VPTree
from point import Point

#same number of pictures to show in the index.html file
PIC_NUM = 12

# KNN
def knn_search(index, query, k):
    results = []
    #finding results as list of points (features, img_id, label)
    list = VPTree.knn(index, query, k)
    list.sort(key=lambda y: y[0])
    i = 0
    for l in list:
        if i > PIC_NUM:
            break
        print("Label:" + l[1].label + " " + l[1].img_id)
        print("Distanza: ", l[0])
        filtered = (l[1].label, l[1].img_id)
        i = i + 1
        results.append(filtered)
    return results

# Range
def range_search(index, query, parameter):
    results = []
    i = 0
    list = VPTree.range_search(index, query, int(parameter))
    list.sort(key=lambda y: y[1])
    for l in list:
        if i > PIC_NUM:
            break
        print("Label:" + l[0].label + " " + l[0].img_id)
        print("Distanza: ", l[1])
        filtered = (l[0].label, l[0].img_id)
        i = i + 1
        results.append(filtered)
    return results