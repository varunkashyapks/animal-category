import keras
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score
import os
import json
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
train['classes'] = train['category_id'].apply(lambda x: classes[x])

train.head()

fig = plt.figure(figsize=(25, 60))
imgs = [np.random.choice(train.loc[train['classes'] == i, 'file_name'], 4) for i in train.classes.unique()]
imgs = [i for j in imgs for i in j]
labels = [[i] * 4 for i in train.classes.unique()]
labels = [i for j in labels for i in j]
for idx, img in enumerate(imgs):
    ax = fig.add_subplot(14, 4, idx + 1, xticks=[], yticks=[])
    im = Image.open("../input/train_images/" + img)
    plt.imshow(im)
    ax.set_title(f'Label: {labels[idx]}')
