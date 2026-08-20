import math
import time

import numpy as np
from tqdm import tqdm

import pandas as pd
import numpy as np



def test(w, x, y):
    l = x.shape[0]
    right = 0
    y_pred = x.dot(w)
    pred = (y_pred > 0.5).astype(int)
    label = y
    right += np.sum(pred == label) 
    print("test accuracy ", right / l)


def forward(weights, x):
    wx = x.dot(weights)
    # output = 1 / (1 + np.exp(-wx))
    return wx



def load_test_data():
    path = "../data/mnist/mnist_test.csv"
    test_data = pd.read_csv(path, header=None)
    header = ['label']
    for i in range(784):
        header.append(str(i))
    test_data.columns = header
    # change to binary classification
    test_data['label'] = np.where(test_data['label'] == 0, 0, 1)
    # change to array and normalized
    test_images = test_data.drop(['label'], axis=1).to_numpy().reshape(-1,  784) / 255.0
    test_labels = test_data['label'].to_numpy()
    test_labels = test_labels.reshape((-1, 1))
    # change to one_hot
    # test_labels = np.eye(10)[test_labels]
    return test_images, test_labels


def load_train_data():
    # 读入数据
    path = "../data/mnist/mnist_train.csv"
    train_data = pd.read_csv(path, header=None)
    header = ['label']
    for i in range(784):
        header.append(str(i))
    train_data.columns = header
    # change to binary classification
    train_data['label'] = np.where(train_data['label'] == 0, 0, 1)
    # change to array and normalized
    train_images = train_data.drop(['label'], axis=1).to_numpy().reshape(-1,  784) / 255.0
    train_labels = train_data['label'].to_numpy()
    train_labels = train_labels.reshape((-1, 1))

    # change to one_hot
    # train_labels = np.eye(10)[train_labels]
    return train_images, train_labels

def linear():
    train_x, train_y = load_train_data()
    test_x, test_y = load_test_data()
    weights = np.zeros((784,1))
    batch_size = 128
    learning_rate = 0.01
    l = train_x.shape[0]
    batch_number = math.floor(l / batch_size)
    start = time.time()
    iterations = 0
    for epoch in range(5):
        total_error = 0
        right = 0
        print("epoch: ", epoch)
        for i in tqdm(range(batch_number)):
            if(iterations % 100 == 0):
                print("第",iterations,"迭代")
                test(weights, test_x, test_y)
            iterations = iterations + 1
            x_batch = train_x[i * batch_size: min((i + 1) * batch_size, l)]
            y_batch = train_y[i * batch_size: min((i + 1) * batch_size, l)]
            y_pred = np.dot(x_batch, weights)
            pred = (y_pred > 0.5).astype(int)
            label = y_batch
            error = y_pred - y_batch
            delta = x_batch.T.dot(error) * learning_rate / error.shape[0]
            weights = weights - delta     
         # 计算误差
        total_error += np.sum(error * error)
        right += np.sum(pred == label)
        print("training error: ", total_error/l)
        test(weights,test_x, test_y)
        

linear()