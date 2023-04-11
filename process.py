import pandas as pd

df = pd.read_csv("output", header = None)

df2 = pd.concat([df[0][i*10:i*10+10].reset_index(drop=True) for i in range(df.size // 10)], axis=1)
df2.to_csv("csv.csv")
