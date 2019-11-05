import pandas as pd

df = pd.read_csv("TestData.1.csv")

print(df['TIC'].tolist())