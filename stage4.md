---
layout: post
title: Reusable tSNE
mathjax: true
permalink: /stage4/
---

## t-SNE vs PCA
Dimensionality reduction is a technique often employed for visualization purposes. The method of choice used to be Principle Component Analysis (PCA), but since its arrival on the scene, t-Distribution Stochastic Neighborhood Embedding (t-SNE) has been gaining popularity due to its improved performance in maintaining distance-based relationships between data points from the high dimensional space. 

However, while tSNE offers advantages in performance, its mode of operation prevents it from being used on train-test splits. tSNE requires the information for all data points at once, and rather than find a transformation like PCA, it simply finds an embedding in the low dimensional space which optimizes some loss function.

The methodology behind tSNE is explained extremely well in [this](https://www.youtube.com/watch?v=RJVL80Gg3lA) Google talk by its inventor Laurens van der Maaten. So I will skip into my proposed methodology to fix the issue. This blog post focuses on creating a reproducible wrapper around tSNE (using the scikit-learn implementation).


## The Methodology

tSNE works by finding an embedding in an $m$ dimensional space from an $n$ dimensional space (where $m \le n$). So let's ask ourselves, perhaps once the embedding is found, there is a mapping from the $n$ dimensional space to the $m$ dimensional space? In other words, for each dimension $i$ in the new $m$ dimensional space, is there a mapping:

$$\Phi(\vec{X}) = \vec{X}'$$

Where $\vec{X} \in {\rm I\!R}^n$ and $\vec{X}' \in {\rm I\!R}^m$

If such a mapping exists, then for a reduction to $m$ dimensional space, we could employ a linear regression to find this mapping.

To test this theory, we will first try to fit a linear model on a multi-dimensional dataset from `sklearn`. I have chosen the MNIST dataset as the example here since it has more than two classes, and requires no scaling (all dimensions are in the same scale).


We will start by importing everything we need:

```python
import numpy as np
import time
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=False)
import pandas as pd

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR


import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

%matplotlib inline

```

Let's load the dataset:

```python
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

MNIST_X = mnist.data
MNIST_Y = mnist.target
print(MNIST_X.shape)
print(MNIST_Y.shape)

np.random.seed(0)
mnist_percent = 0.02
X_index = np.random.choice(list(range(MNIST_X.shape[0])), size=int(MNIST_X.shape[0]*mnist_percent))

MNIST_X_10 = MNIST_X[X_index]
MNIST_Y_10 = MNIST_Y[X_index]

print(MNIST_X_10.shape)
print(MNIST_Y_10.shape)

```

```
(70000, 784)
(70000,)
(1400, 784)
(1400,)
```

And now finally run tSNE on the result, we will reduce the number of dimensions to 2 so we can easily visualize the data.

```python
tSNE = TSNE(n_components=2, 
            random_state=2, 
            perplexity=20, 
            early_exaggeration=4.0, 
            learning_rate=400.0,
            n_iter=1000, 
            angle=0.3,
            verbose=1)

MNIST_X_10_2D = tSNE.fit_transform(MNIST_X_10)
```

Let's define a function to easily plot the data:

```python
def plot_MNIST(MNIST_x, MNIST_y):
    plt.figure(figsize=(10,10))
    colors = {0: "red",
              1: "blue",
              2: "black",
              3: "orange",
              4: "green",
              5: "pink",
              6: "purple",
              7: "yellow",
              8: "magenta",
              9: "silver"}

    for i in range(10):
        mask = MNIST_y == i    
        plt.scatter(MNIST_x[mask][:, 0], MNIST_x[mask][:, 1], c=colors[i], label=f'{i}')
    plt.xlabel("tSNE component 1")
    plt.ylabel("tSNE component 2")
    plt.legend()
    plt.show();
```
```python
plot_MNIST(MNIST_X_10_2D, MNIST_Y_10)
```

<img src="/assets/images/rtSNE.svg" />

This is the result of tSNE when it is run on a dataset for dimensionality reduction. However, we want to ask ourselves, what if we had a train/test split? tSNE would only be fitted on the training data, and then we'd have to recreate the embedding as best we can for the test data.

## The Train/Test Split Set-up

Now let's consider our scenario of interest. We will use the MNIST data again, and split it into a train and a test datasets.

```python
MNIST_X_10_train, MNIST_X_10_test, MNIST_Y_10_train, MNIST_Y_10_test = train_test_split(MNIST_X_10, 
                                                                        MNIST_Y_10, 
                                                                        test_size=0.33, 
                                                                        random_state=1)
MNIST_X_10_train_2D, MNIST_X_10_test_2D, MNIST_Y_10_train, MNIST_Y_10_test = train_test_split(MNIST_X_10_2D, 
                                                                              MNIST_Y_10, 
                                                                              test_size=0.33, 
                                                                              random_state=1)
```

Notice we split both the original 784-dimensional dataset, as well as the transformed 2-dimensional dataset. Furthermore, we split both in the same way by fixing the `random_state` of both splitting methods. 
So now we have our original dataset split, as well as the reduced dataset. We will use the train data to try and find a mapping, first linear and then non-linear, and then test how good our fit is using the test data.
At this point it's important to define our independent variables and our dependent variables; Our independent variable is the 784-dimensional data, while our dependent variable is the 2-dimensional data. 

First, let's look at the training data, we will only use the y data to colour the three different digits

```python
plot_MNIST(MNIST_X_10_train_2D, MNIST_Y_10_train)
```

<img src="/assets/images/rtSNE2_train.svg" />
