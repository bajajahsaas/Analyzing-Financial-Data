import pandas as pd

df = pd.read_csv("YahooFinances.csv")

print(df['company'].tolist())