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

Now let's plot the test data:

```python
plot_MNIST(MNIST_X_10_test_2D, MNIST_Y_10_test)
```

<img src="/assets/images/rtSNE2_test.svg" />


Now let's create our linear regression to learn the mapping function. I'm using a LASSO regression here since early experiments showed the stock linear regression performs terribly.

```python
my_lr = Lasso()
my_lr.fit(MNIST_X_10_train, MNIST_X_10_train_2D)
MNIST_X_10_pred_2D = my_lr.predict(MNIST_X_10_test) 
```

We're going to develop an equivalent measure of the $R^2$ value, but for distances. Simply put, we will calculate
$$1-\frac{u}{v}$$
Where $u$ is the distance from all calculated positions to the the true positions, and $v$ is the distance from the center of the true positions to all the individual true positions.

Much like the $R^2$ measure, this is bound by an upper limit of 1.0 if all coordinates are predicted perfectly.

```python
def distance_r2(X_pred, X_true):
    u = np.mean(np.linalg.norm(X_true-X_pred, axis=1))
    v = np.mean(np.linalg.norm(X_true-np.mean(X_true, axis=0), axis=1))
    
    return 1- (u/v)
```
```python
distance_r2(MNIST_X_10_pred_2D, MNIST_X_10_test_2D)
```
```
0.4635027691570044
```

Not bad... but there is room for improvement.
First of all, a distance-$R^2$ value of 0.46 may not seem great, but let's ask ourselves, how consistent is this performance? to test this point, let's split the data in the same proportion 1000 times, with a different seed every time, and look at the distribution of performances.

We will also re-fit tSNE every few times since its performance is also stochastic and non-deterministic.

```python
lr_distances = []

max_trials = 1000


for i in range(max_trials):
  
    if ((i % 50) == 0):
        # make a new tSNE and embedding
        tSNE = TSNE(n_components=2, 
                    perplexity=20, 
                    early_exaggeration=4.0, 
                    learning_rate=400.0,
                    n_iter=1000, 
                    angle=0.3)

        MNIST_X_2D = tSNE.fit_transform(MNIST_X_10)

    # split the source data and the embedding
    MNIST_X_train, MNIST_X_test, MNIST_Y_train, MNIST_Y_test = train_test_split(MNIST_X_10, 
                                                                                MNIST_Y_10, 
                                                                                test_size=0.2, 
                                                                                random_state=i)
    MNIST_X_train_2D, MNIST_X_test_2D, MNIST_Y_train, MNIST_Y_test = train_test_split(MNIST_X_2D, 
                                                                                      MNIST_Y_10, 
                                                                                      test_size=0.2, 
                                                                                      random_state=i)
  

    # create the embedding mapping
    my_lr.fit(MNIST_X_train, MNIST_X_train_2D)
    MNIST_X_pred_2D = my_lr.predict(MNIST_X_test)

    # calculate the quality of the embedding mapping
    dist_r2 = distance_r2(MNIST_X_pred_2D, MNIST_X_test_2D)
    lr_distances.append(dist_r2)
     
    # give some progress output
    if ((i%10) == 0):
        print(f'finished {i} rounds: {dist_r2}', end="\r")
```

```python
plt.figure(figsize=(7,7))
plt.hist(lr_distances, bins=100)
plt.show()
print(np.mean(lr_distances),"+/-",np.std(lr_distances))
```
<img src="/assets/images/LR_hist.svg" />

```
0.475123113167164 +/- 0.022504375367352846
```

So in general it seems the distance-$R^2$ value will be around 0.48 with a standard deviation of 0.02. This is our baseline, and it doesn't look ideal.

The next question is, can non-linear methods do better? Let's set up an environment to test this idea. 

We will try a few models:


*   K-Nearest Neighbour (with varying $K$ values)
*   Decision Tree (with varying depth limits)
*   Linear Support Vector Machine (with varying degrees of regularization)



## Testing Advanced Models

First, let's define a function `measure_distances` which will accept a model and a number of trials, and plot for us the result of employing this model in finding a mapping.

```python
def measure_distances(model1, model2, number_of_trials=10000):
    # The function accepts two models since SVM models cannot accept multiple targets
    distances = []

    for i in range(number_of_trials):

        if ((i % 50) == 0):
            # make a new tSNE and embedding
            tSNE = TSNE(n_components=2, 
                        perplexity=20, 
                        early_exaggeration=4.0, 
                        learning_rate=400.0,
                        n_iter=1000, 
                        angle=0.3)

            MNIST_X_10_2D = tSNE.fit_transform(MNIST_X_10)

        # split the source data and the embedding
        MNIST_X_10_train, MNIST_X_10_test, MNIST_Y_10_train, MNIST_Y_10_test = train_test_split(MNIST_X_10, 
                                                                                MNIST_Y_10, 
                                                                                test_size=0.2, 
                                                                                random_state=i)
        MNIST_X_10_train_2D, MNIST_X_10_test_2D, MNIST_Y_10_train, MNIST_Y_10_test = train_test_split(MNIST_X_10_2D, 
                                                                                      MNIST_Y_10, 
                                                                                      test_size=0.2, 
                                                                                      random_state=i)
        
        # create the embedding mapping
        try:
            model1.fit(MNIST_X_10_train, MNIST_X_10_train_2D)
            MNIST_X_10_pred_2D = model1.predict(MNIST_X_10_test)
        except ValueError:
            # go here if the regressor cannot accept multiple outputs
            
            model1.fit(MNIST_X_10_train, MNIST_X_10_train_2D[:, 0])
            model2.fit(MNIST_X_10_train, MNIST_X_10_train_2D[:, 1])

            MNIST_X_10_pred_2D = np.zeros(MNIST_X_10_test_2D.shape, MNIST_X_10_test_2D.dtype)

            MNIST_X_10_pred_2D[:, 0] = model1.predict(MNIST_X_10_test)
            MNIST_X_10_pred_2D[:, 1] = model2.predict(MNIST_X_10_test)

        dist_r2 = distance_r2(MNIST_X_10_pred_2D, MNIST_X_10_test_2D)
        distances.append(dist_r2)
        
        # provide progress output
        if ((i%10) == 0):
            print(f'finished {i} rounds: {dist_r2}', end="\r")

    return distances
```

### K-Nearest Neighbour

We'll run our simulation with different values of $K$ for the KNN method and then use plotly to plot the results

Run the process
```python
knn_distances = {}
max_k = 10
for k in range(1, max_k):
    print("k="+str(k))
    model1 = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
  
    knn_distances[k] = measure_distances(model1, None, number_of_trials=max_trials)
    print() 
```
Organize everything into a long dataframe
```python
model_type = []
observation = []
k_values = []

for distance in lr_distances:
    model_type.append("LinearRegression")
    observation.append(distance)
    k_values.append(0)
    
for k in knn_distances:
    for distance in knn_distances[k]:
        model_type.append(f'KNN({k})')
        observation.append(distance)
        k_values.append(k)
        
knn_df = pd.DataFrame({"Model_Type": model_type,
                       "Observation": observation,
                       "K_value": k_values})
```
And finally, plot
```python
unique_models = knn_df["Model_Type"].unique()

fig = go.Figure()
for model in unique_models:
    mask = knn_df["Model_Type"] == model
    k_values = knn_df["K_value"][mask]
    k = k_values.iloc[0]
    if (k == 0):
        color = f'rgb(0,255,255)'
    else:
        r = 255
        g = b = 255-int(255*(k-1)/max_k)
        color = f'rgb({r},{g},{b})'
    fig.add_trace(go.Violin(x=knn_df['Model_Type'][mask],
                            y=knn_df['Observation'][mask],
                            name=model,
                            box_visible=False,
                            meanline_visible=True,
                            fillcolor=color,
                            line_color="black",
                            opacity=1.0,
                            hoverinfo='y'))
    
fig.update_yaxes(range=[0.25, 1.0])
plotly.offline.iplot(fig)
```
<iframe width="900" height="500" frameborder="0" src="/assets/plotly/KNN_distance.html"></iframe>

This is somewhat better. The distance-$R^2$ values center around 0.86.
This method seems better at estimating the positions, but it can be lacking in interpretability, and scales in order of $O(n^2 d)$ in relation to the number of data points $n$ and the number of features $d$. Let's try decision trees:

### Decision Trees

For decision trees, we will try different depth caps, including no depth cap:

```python
max_depth = 21
max_depth_list = list(range(1, max_depth, 3))
max_depth_list.append(None)

tree_distances = {}



for depth in max_depth_list:
    print("depth="+str(depth))
    model1 = DecisionTreeRegressor(max_depth=depth)
    
    if (depth is not None):
        tree_distances[depth] = measure_distances(model1, None, number_of_trials=max_trials)
    else:
        tree_distances[-1] = measure_distances(model1, None, number_of_trials=max_trials)
    
    print()
```
```python
model_type = []
observation = []
DT_values = []

for distance in lr_distances:
    model_type.append("LinearRegression")
    observation.append(distance)
    DT_values.append(0)
    
for depth in tree_distances:
    for distance in tree_distances[depth]:
        if (depth == -1):
            model_type.append(f'DT(None)')
        else:
            model_type.append(f'DT({depth})')
        observation.append(distance)
        DT_values.append(depth)
        
dt_df = pd.DataFrame({"Model_Type": model_type,
                      "Observation": observation,
                      "DT_value": DT_values})
```
```python
unique_models = dt_df["Model_Type"].unique()

fig = go.Figure()
for model in unique_models:
    mask = dt_df["Model_Type"] == model
    depth_values = dt_df["DT_value"][mask]
    depth = depth_values.iloc[0]
    if (depth == 0):
        color = f'rgb(0,255,255)'
    elif (depth == -1):
        r = 255
        g = 255
        b = 255
    else:
        r = b = 255-int(255*(depth-1)/max_depth)
        g = 255
        color = f'rgb({r},{g},{b})'
    fig.add_trace(go.Violin(x=dt_df['Model_Type'][mask],
                            y=dt_df['Observation'][mask],
                            name=model,
                            box_visible=False,
                            meanline_visible=True,
                            fillcolor=color,
                            line_color="black",
                            opacity=1.0,
                            hoverinfo='y'))
fig.update_yaxes(range=[0.0, 1.0])
fig.update_xaxes(tickangle=45)
plotly.offline.iplot(fig)
```
<iframe width="900" height="500" frameborder="0" src="/assets/plotly/DT_distances.html"></iframe>
