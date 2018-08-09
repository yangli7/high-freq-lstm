import pandas as pd
import matplotlib.figure as plt

# load data
dataSet = pd.read_csv('Tick_Level_Data/SH600637_2018-06-11.csv', encoding='gbk')
dataSet = dataSet[dataSet.close > 0]
dataSet = dataSet.drop(['stockid', 'StockID', 'StockName', 'open', 'close', 'high', 'syl1', 'syl2'], axis=1)


print(dataSet)