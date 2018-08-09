import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataSet = pd.read_csv('database_SH600637_18-06-15.csv', encoding='gbk')
label = dataSet['5min'].values
price = dataSet['price'].values


def sequence_plot(price, label):
    label_len = len(label)
    price_len = len(price)

    x = np.arange(0, label_len, 1)
    y = np.arange(100, price_len + 100, 1)

    plt.figure(figsize=(50, 20))
    plt.plot(x, price, label='stock price', marker='.', color='green')
    plt.plot(y, label, label='price mean', color='blue')
    plt.legend(['stock price', 'price mean'], loc='upper right')
    plt.title('stack price vs price mean')
    plt.xlabel('time')
    plt.ylabel('price')

    plt.show()


def drawCumulativeHist(data, label):
    plt.hist(data, 20, normed=True, histtype='step', cumulative=True)
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.title(label)
    plt.show()


def drawHist(data, label):
    plt.hist(data, 200)
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.title(label)
    plt.show()