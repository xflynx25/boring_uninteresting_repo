from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

class Dataset:
    def __init__(self, data, target):
        self.train, self.test, self.target_train, self.target_test = train_test_split(data, target, test_size=0.2, random_state=42)

def get_mnist_dataset(size=1000):
    # Load MNIST dataset from sklearn (which contains a subset of original MNIST)
    mnist = datasets.fetch_openml('mnist_784', version=1)
    data, target = mnist.data[:size], mnist.target[:size]
    return Dataset(data, target)

def get_random_classification_dataset(input_dim=784, categories=10, size=1000):
    data = np.random.random((size, input_dim))
    target = np.random.randint(categories, size=size)
    return Dataset(data, target)

def get_breast_cancer_dataset(size=1000):
    bc = datasets.load_breast_cancer()
    data, target = bc.data[:size], bc.target[:size]
    return Dataset(data, target)


# Dataset
"""
input your data 
will have many statistics fields, but will not calculate by default, you have funcs you can call to calculate
- print summary statistics
- calculate prelims
- calculate univariate
- calculate bivariate
- calculate multivariate
- calculate nonlinear
with options to save and such. 
So this dataset will have much embedded, but we also need a way to take it and do stuff with. Well, we have the .x, .y
So maybe a seperate file with advanced stats 
File with training methods. 
This should just represent the data, itself 

How to do augmentation? Maybe you call a new object 

We have a file for doing more complicated stat

A file for models, and inheriting there as well 

A file for visualizations complicated 

Then the evaluator module which i have already started on
- this needs to take dataset, and models, and do stuff with it. 
"""

# Regression Dataset inherits from Dataset

# Classification Dataset inherits from Dataset


# functions that implement basic ones of these 