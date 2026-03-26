import pandas as pd
from sklearn.linear_model import LinearRegression

# 1
df = pd.read_csv("dataset.csv")
print(df.head())   
print(df.info())   
print(df.shape)  
#problémově vypadávají sloupce: sleep_hours, attendance, previous_score, final_score

#2
#datové typy
for col in ["voltage_v", "current_a", "lamp_distance_cm", "temperature_c"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")   
#hodnoty
df = df[df[""] >= 0]
df = df[df[""].between(0, 90)]
# Duplicity  
df = df.drop_duplicates()