---
layout: post
title: Reasoning With Probability - Is My Model Good Enough?
mathjax: true
image: Pyro-tutorial-image.png
---

At its core, probabilistic programming is designed to answer questions about uncertainty. Some very popular examples online discuss the use of probabilistic programming to aid with neural networks and making them more capable of dealing with uncertain examples. However, in this tutorial I am going to introduce the most basic functionality probabilistic programming can help with.

The example here, to a very large extent, is taken from a <a href="https://www.youtube.com/watch?v=5f-9xCuyZh4" target="_blank">talk given by Zach Anglin</a> where he introduces probabilistic programming using PyMC3. I will be using Pyro since I see this library taking off in the future and I want to explore it a bit more.

## The Example Scenario

Let's consider the following simple workflow

* Obtain data (in this case a sample dataset)
* Perform a train/test split
* Train a model on the train data
* Test the model's performance and decide if the performance is good or not


```python
# imports
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# Obtain data
mnist = fetch_mldata('MNIST original')

MNIST_X = mnist.data
MNIST_Y = mnist.target
print(MNIST_X.shape)
print(MNIST_Y.shape)
```

```
(70000, 784)
(70000,)
```

```python
# Split the data
X_train, X_test, y_train, y_test = train_test_split(MNIST_X, MNIST_Y, test_size=0.2, random_state=1)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
```
```
(56000, 784) (14000, 784)
(56000,) (14000,)
``` 


```python
# Train our model
my_tree = DecisionTreeClassifier(min_samples_leaf=2)
my_tree.fit(X_train, y_train)
```

```python
# Check the model accuracy score
accuracy = my_tree.score(X_test, y_test)
print(f'Accuracy is {"{0:.2f}".format(accuracy*100)}%')
```
```
Accuracy is 87.06%
```    

### Analysis

We can try and improve the model's accuracy to reach higher than 87%, or even employ a stronger model such as a CNN, but there is a slightly less obvious issue. Consider how we got to know we have 87% accuracy.

Our test set is 14,000 labeled examples. A true luxury since for some data science problems we may not encounter such a high number of nicely labeled examples. What if instead of 14,000 examples, we had only 100 examples?


```python
_, X_100_test, _, y_100_test = train_test_split(X_test, y_test, test_size=(100/X_test.shape[0]), random_state=1)
print(X_100_test.shape, y_100_test.shape)
```
```
(100, 784) (100,)
```    


```python
# Check the model accuracy score again
accuracy = my_tree.score(X_100_test, y_100_test)
print(f'Accuracy is {"{0:.2f}".format(accuracy*100)}%')
```
```
Accuracy is 86.00%
```    

We got a slightly lower number, but still comparable.

That said, while the numbers are close, having faith the model based on 100 examples is harder than having faith based on 14,000 examples.

Intuitively, our confidence in the test set's score increases as the size of the test set increases. Pyro allows us to numerically estimate our confidence and how much room for error we have.

## The Pyro Model

The Pyro workflow is very unique and requires three components

1. A model function which simulates our underlying model (not the decision tree, but the process which gives us correct or incorrect labels). This function starts with a **prior distribution**
2. A kernel which measures the likelihood of the observed examples as they are produced by the model function
3. A sampler which builds the **posterior distribution**


```python
# Import required libraries
import pyro 
import torch 
import pyro.distributions as dist 
import pyro.poutine as poutine

from pyro.infer.mcmc import HMC, MCMC

assert pyro.__version__.startswith('1.1.0')
```

First we need to define our observations for Pyro. In our case, our observations are the 
cases where we see correct and incorrect classification. Let's consider both the small test 
set (100 observations) and the large test set (14,000 observations).


```python
# define our observations
y_100_pred = my_tree.predict(X_100_test)
correctness_values_100 = y_100_pred == y_100_test   # An array of observations that shows
                                                    # if our predictions match the test
                                                    # or not

y_pred = my_tree.predict(X_test)
correctness_values = y_pred == y_test

# We have to convert our observations into PyTorch tensors
correctness_values_100 = torch.from_numpy(correctness_values_100).float()
correctness_values = torch.from_numpy(correctness_values).float()
```

Now we have to define the model function. The function accepts our observations and tries to sample from a prior distribution and calculate the likelihood of the prior based on our observations. The kernel will then update the prior to a more likely posterior distribution.


```python
# define our model, y is our set of observations
def model(y):
    # Our observations are binary observations which come
    # from a Bernoulli distribution with some percent
    # "p" to be correct and (1-p) of being wrong
    
    # we want to estimate that p value. We start not
    # knowing anything about it, so our prior will be
    # a uniform distribution from 0.0 to 1.0
    underlying_p = pyro.sample("p", dist.Uniform(0.0, 1.0)) 
    
    # for each observation
    for i in range(len(y)):
        
        # our hidden distribution
        y_hidden_dist = dist.Bernoulli(underlying_p)
        
        # now sample from our distribution conditioned on our observation
        y_real = pyro.sample("obs_{}".format(i), y_hidden_dist, obs = y[i])
        
```

It doesn't look like much, but in this model two important events happen.

First, we have used the `pyro.sample` function to register a parameter named `"p"` as a learnable value for Pyro. The use of either `pyro.sample` or `pyro.param` registers the resulting value with Pyro's internal store (a special dictionary-like object) as learnable values. In our case we've said `"p"` is a learnable distribution.

We've also registered every observation as a learnable value which should comply with the observation we provide to it.

The kernel we will register this model with (Hamiltonian Monte Carlo kernel) will look at all learnable values defined in this model and will attempt to adjust the learnable distributions such that they increase the likelihood of provided observations. 

The sampler we will use (Markov Chain Monte Carlo) will run the HMC kernel to find a posterior distribution.


```python
# First clear the old values of all our stored parameters
pyro.clear_param_store()

# the kernel we will use
hmc_kernel = HMC(model,
                 step_size = 0.1)


# the sampler which will run the kernel
mcmc = MCMC(hmc_kernel, num_samples=100, warmup_steps=100)

# the .run method accepts as parameter the same parameters our model function uses
mcmc.run(correctness_values_100)
```
```
Sample: 100%|█████████████████████████████████████████| 200/200 [03:46,  1.13s/it, step size=1.51e+00, acc. prob=0.916]
```    

Now we've run our sampler, we can do several things with our resulting posterior probability. First, we may want to visualize the probability `"p"` we've defined. We can use do so by sampling from our optimizer.


```python
sample_dict = mcmc.get_samples(num_samples=5000)
```


```python
plt.figure(figsize=(10,7))
sns.distplot(sample_dict['p'].numpy(), color="orange");
plt.xlabel("Observed probability value")
plt.ylabel("Observed frequency")
plt.show();
```


![svg](/assets/images/small_distributions.svg)

```python
mcmc.summary(prob=0.95)
```
```
        mean       std    median      2.5%     97.5%     n_eff     r_hat
 p      0.86      0.04      0.86      0.79      0.93    185.83      1.00
    
 Number of divergences: 0
```
    

Not such a great result... 100 observations is not really enough to settle on a good outcome. 
The distribution does not seem to center around a particular value, and when we ask for the 
95% credibility interval our true value can lie anywhere between 79% and 93%. This may or may not 
be accurate enough for our purposes.

Let's see how confident we can be in our model when we use all 14,000 observations.

### Modifying Our Model

If we run all 14,000 observations through the same model, it would take a very long time to run. 
This is because we cycle through each observation in our code:

```python
for i in range(len(y)):
        
        # our hidden distribution
        y_hidden_dist = dist.Bernoulli(underlying_p)
        
        # now sample from our distribution conditioned on our observation
        y_real = pyro.sample("obs_{}".format(i), y_hidden_dist, obs = y[i])
```

Pyro contains a more convenient, vectorized, method of approaching our model.

First, we redefine our model function such that it accepts NO observations, but rather it 
returns its own observations


```python
# define our model, y is our set of observations
def model2():
    # Our observations are binary observations which come
    # from a Bernoulli distribution with some percent
    # "p" to be correct and (1-p) of being wrong
    
    # we want to estimate that p value. We start not
    # knowing anything about it, so our prior will be
    # a uniform distribution from 0.0 to 1.0
    underlying_p = pyro.sample("p2", dist.Uniform(0.0, 1.0)) 
    
    
    
        
    # our hidden distribution
    y_hidden_dist = dist.Bernoulli(underlying_p)

    # now sample from our distribution conditioned on our observation
    y_real = pyro.sample("obs", y_hidden_dist)
    
    return y_real
        
```

Now, we defined a second function that takes as input a model function, and observations, 
and utilizes `pyro.poutine` to run the model function in a conditioned environment. It's 
important our observations here have the same name ("obs") as they do in the model function.


```python
def conditioned_model(model, y):
    conditioned_model_function = poutine.condition(model, data={"obs": y}) # this returns a function
    return conditioned_model_function()
```

Finally, we re-run the MCMC sampler, but now with our conditioned model, and we send our model 
function, as well as our observations, as arguments


```python
# First clear the old values of all our stored parameters
pyro.clear_param_store()

# the kernel we will use
hmc_kernel2 = HMC(conditioned_model, 
                  step_size = 0.1)


# the estimator which will run the kernel
mcmc2 = MCMC(hmc_kernel2, num_samples=100, warmup_steps=100)

# the .run method accepts as parameter the same parameters our model function uses
mcmc2.run(model2, correctness_values)
```
```
Sample: 100%|█████████████████████████████████████████| 200/200 [01:47,  1.86it/s, step size=7.87e-01, acc. prob=0.984]
```    


```python
sample_dict2 = mcmc2.get_samples(num_samples=5000)
```


```python
plt.figure(figsize=(10,7))
sns.distplot(sample_dict2['p2'].numpy(), color="blue");
plt.xlabel("Observed probability value")
plt.ylabel("Observed frequency")
plt.show();
```


![svg](/assets/images/large_distributions.svg)


```python
mcmc2.summary(prob=0.95)
```

```    
         mean       std    median      2.5%     97.5%     n_eff     r_hat
 p2      0.87      0.00      0.87      0.87      0.87      9.88      1.07
    
 Number of divergences: 0
```
    

Now we can get a much tighter fit around the 87% mark. Just to compare the two distributions, 
we could plot them together.


```python
plt.figure(figsize=(10,7))
sns.distplot(sample_dict2['p2'].numpy(), label="14,000 examples", color="blue")
sns.distplot(sample_dict['p'].numpy(), label="100 examples", color="orange")
plt.legend()
plt.show()
```


![svg](/assets/images/two_distributions.svg)


Notice these are both posterior distributions plotted, which are supported entirely by the 
evidence collected. If in the first case (100 examples), the results would not be conclusive 
enough, this would be a very strong rational to label more examples so that an acceptable 
credibility interval is derived.