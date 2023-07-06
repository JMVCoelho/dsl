#!/usr/bin/env python

import argparse
import random
import os
from itertools import count
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_data(path, bias=False):
    """
    path: location of FashioMNIST data: expects a folder with np arrays. See hw1-dataload.py
    feature_rep: a function or None. Use it to transform the binary pixel
        representation into something more interesting in Q 2.2a
    bias: whether to add a bias term as an extra feature dimension
    """
    
    train_X = np.load(os.path.join(path,'train_features.npy'))
    train_y = np.load(os.path.join(path,'train_labels.npy'))
    dev_X = np.load(os.path.join(path,'dev_features.npy'))
    dev_y = np.load(os.path.join(path,'dev_labels.npy'))
    test_X = np.load(os.path.join(path,'test_features.npy'))
    test_y = np.load(os.path.join(path,'test_labels.npy'))

    
    train_X = train_X.reshape(train_X.shape[0],-1)/255
    mean = np.mean(train_X.reshape(-1,1))
    std = np.std(train_X.reshape(-1,1))
    print("normalisation: ",mean,std)
    train_X = (train_X - mean) / std
    
    dev_X = dev_X.reshape(dev_X.shape[0],-1)/255
    dev_X = (dev_X - mean) / std
    test_X = test_X.reshape(test_X.shape[0],-1)/255
    test_X = (test_X - mean) / std
    if bias:
        bias_vector = np.ones((train_X.shape[0], 1), dtype=int)
        train_X = np.hstack((train_X, bias_vector))
        bias_vector = np.ones((dev_X.shape[0], 1), dtype=int)
        dev_X = np.hstack((dev_X, bias_vector))
        bias_vector = np.ones((test_X.shape[0], 1), dtype=int)
        test_X = np.hstack((test_X, bias_vector))

    return {"train": (train_X, train_y),
            "dev": (dev_X, dev_y),
            "test": (test_X, test_y)}



def softmax(z, axis=None):
    raw = np.exp(z)
    return raw / raw.sum(axis=axis)


def stable_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in tqdm(zip(X, y)):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        
        return n_correct / n_possible

## Question 2.1.a
class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        l2_penalty (float): BONUS
        """
        # Question 2.1 a
        output = np.argmax(np.dot(self.W, x_i.T))

        lr = 1
        l2_penalty = 0

        # update rule + l2 penalty + learning rate
        if output != y_i:
            self.W[output] = self.W[output] - lr*x_i - lr*l2_penalty*self.W[output] 
            self.W[y_i] = self.W[y_i] + lr*x_i - lr*l2_penalty*self.W[y_i]

def relu(z):
    return np.clip(z, 0, None)

def relu_prime(z):
    return z > 0

f_derivatives = {relu: relu_prime}

## Question 2.1.b
class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        output = stable_softmax(np.dot(self.W, x_i.T))
        
        # this computes the derivative of the categorical cross-entropy for a single input vector x_i;
        # label converted from class integer to one-hot vector, to match output shape.
        label = np.zeros_like(output)
        label[y_i] = 1
        error_w = np.outer(x_i, output-label)

        # sgd update
        self.W = self.W - learning_rate * error_w.T


## Question 2.2.c
class MLP(object):
    
    def __init__(self, n_classes, n_features, hidden_sizes, layers):
        # Initialize an MLP
        # The implementation supports multiple layers.
        assert len(hidden_sizes) == layers
        print(f'Hidden sizes: {hidden_sizes}')

        layer_dimensions = [n_features] + hidden_sizes + [n_classes]
        sizes = [[layer_dimensions[i], layer_dimensions[i+1]] for i in range(len(layer_dimensions)-1)]

        # init weights from uniform distribution, using info from slides
        # low variance, low positive expected value.
        self.weights = [np.random.uniform(size=(s_in, s_out), low = -1, high = 1.01) for s_in, s_out in sizes]

        # zeros for biases
        self.biases = [np.zeros(s_out) for _, s_out in sizes]

        self.activations = [relu for _ in range(layers)] + [stable_softmax]

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required

        def batch(X, batch_size=1):
            l = X.shape[0]
            for idx in range(0, l, batch_size):
                yield np.array(X[idx:min(idx + batch_size, l)])
        
        batched_predicted_classes = []
        for x_i in batch(X):
            x = x_i
            for i in range(len(self.weights)):
                w, b, f = self.weights[i], self.biases[i], self.activations[i]
                x = f(np.dot(x, w) + b)
            batched_predicted_classes.append(np.argmax(x, axis=1))

        predictions = np.concatenate(batched_predicted_classes)
        assert len(predictions) == X.shape[0]
        return predictions
        
    def evaluate(self, X, y):
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate):
        #Implement the forward and backward passes
        
        # algorithm:
        # forward: same as predict but caching W[i] values
        # backward:
        # W[l] <- W[l] - lr d/dW[l] E
        # b[l] <- b[l] - lr d/db[l] E
        # 
        # where E is the error functions (categorical cross entropy)

        batch_size = 1 # stochastic gradient descent.

        def batch(X, batch_size=1):
            l = X.shape[0]
            for idx in range(0, l, batch_size):
                yield np.array(X[idx:min(idx + batch_size, l)])
        
        for x_i, y_i in zip(batch(X,batch_size), batch(y,batch_size)):
            x_cache = [x_i]
            z_cache = []
            for i in range(len(self.weights)):
                w, b, f = self.weights[i], self.biases[i], self.activations[i]
                z = np.dot(x_cache[i], w) + b
                z_cache.append(z)
                x = f(z)
                x_cache.append(x)
            
            y_i_onehot = np.zeros_like(x)
            for idx, class_label in enumerate(y_i):
                y_i_onehot[idx][class_label] = 1

            # compute deltas
            # last layer delta initialized using the derivative of the cross entropy 
            # using relu_prime as derivative for hidden layer activations
            deltas = [None] * (len(self.weights))
            deltas[-1] = x - y_i_onehot


            for i in reversed(range(1, len(deltas))):
                w = self.weights[i]
                deltas[i-1] = np.multiply(np.dot(deltas[i], w.T), relu_prime(z_cache[i-1]))

            # update weights
            for i in range(len(self.weights)):                
                # batchsize 1 only:
                #self.weights[i] = self.weights[i] - learning_rate * np.outer(x_cache[i], deltas[i])
                #self.biases[i] = self.biases[i] - learning_rate * deltas[i].reshape(-1,)

                # for larger batch size (this works for both, but the above is slightly faster for batchs=1):
                # compute outer product "along the 0 axis", and take the average

                self.weights[i] = self.weights[i] - learning_rate * np.mean((x_cache[i][:,:,np.newaxis] * deltas[i][:,np.newaxis,:]), axis=0)
                self.biases[i] = self.biases[i] - learning_rate * np.mean(deltas[i], axis=0).reshape(-1,)



def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-data', default='letter.data',
                        help="Path to letter.data OCR corpus.")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-bias', action='store_true',
                        help="""Whether to add an extra bias feature to all
                        samples in the dataset. In an MLP, where there can be
                        biases for each neuron, adding a bias feature to the
                        input is not sufficient.""")
    parser.add_argument('-hidden_sizes', type=int, default=[100])
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    opt = parser.parse_args()

    configure_seed(seed=42)

    data = load_data(opt.data, bias=opt.bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 26
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        # Q3
        model = MLP(n_classes, n_feats, opt.hidden_sizes, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []

    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            model.train_epoch(train_X, train_y, learning_rate=opt.learning_rate)
        else:
            model.train_epoch(train_X, train_y)
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))
        print('dev: {:.4f} | test: {:.4f}'.format(
            valid_accs[-1], test_accs[-1],
        ))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()


# Reproducing results
# Perceptron: python hw1-q2-2023-skeleton-fashion.py perceptron -epochs 15 -bias 
# Logistic regression: python hw1-q2-2023-skeleton-fashion.py logistic_regression -epochs 15 -bias -learning_rate 0.001 
# MLP: python hw1-q2-2023-skeleton-fashion.py mlp -epochs 15 -learning_rate 0.001