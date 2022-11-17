import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8,4)
neurons = pd.read_csv("data/neurons.csv")
neurons.boxplot(grid='True',column =['10_neurona', '20_neurona', '50_neurona'], color='green')
plt.show()

lag = pd.read_csv("data/lag.csv")
lag.boxplot(grid='True',column =['24h', '48h', '72h', '120h'], color='green')
plt.show()

features = pd.read_csv("data/features.csv")
features.boxplot(grid='True',column =['a', 'b', 'c', 'd'], color='green')
plt.show()
