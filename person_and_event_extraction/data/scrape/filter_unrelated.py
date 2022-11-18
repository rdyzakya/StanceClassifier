import os
import json
import pandas as pd

data_path = "./result_data.csv"
output_path = "./data_filtered.csv"

df = pd.read_csv(data_path, encoding="utf-8")

def person_occurence(row):
    person = row["person"].lower()
    title = row["title"].lower()

    return title.count(person) > 0

df = df[df.apply(person_occurence, axis=1)]
df = df[["person","content"]]

df.to_csv(output_path, index=False)