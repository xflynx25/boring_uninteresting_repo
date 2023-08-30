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
