"""
generate_sample_data.py
Generates synthetic Sentinel-2 style band data with land cover labels.
Run this to create sample training data before running train.py.

Land cover classes:
  0 = Water        (high NDWI, low NDVI)
  1 = Forest       (high NDVI, high NIR)
  2 = Agriculture  (moderate NDVI, seasonal variation)
  3 = Urban        (low NDVI, high red/blue reflectance)
  4 = Bare soil    (low NDVI, high red, low NIR)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N_PER_CLASS = 500

def add_noise(arr, scale=0.02):
    return arr + np.random.normal(0, scale, size=arr.shape)

rows = []

# Class 0: Water
n = N_PER_CLASS
blue  = add_noise(np.full(n, 0.08), 0.01)
green = add_noise(np.full(n, 0.06), 0.01)
red   = add_noise(np.full(n, 0.04), 0.01)
nir   = add_noise(np.full(n, 0.02), 0.005)
for i in range(n):
    rows.append([max(0,blue[i]), max(0,green[i]), max(0,red[i]), max(0,nir[i]), 0])

# Class 1: Forest
blue  = add_noise(np.full(n, 0.04), 0.01)
green = add_noise(np.full(n, 0.08), 0.01)
red   = add_noise(np.full(n, 0.05), 0.01)
nir   = add_noise(np.full(n, 0.45), 0.05)
for i in range(n):
    rows.append([max(0,blue[i]), max(0,green[i]), max(0,red[i]), max(0,nir[i]), 1])

# Class 2: Agriculture
blue  = add_noise(np.full(n, 0.06), 0.015)
green = add_noise(np.full(n, 0.10), 0.02)
red   = add_noise(np.full(n, 0.08), 0.02)
nir   = add_noise(np.full(n, 0.30), 0.06)
for i in range(n):
    rows.append([max(0,blue[i]), max(0,green[i]), max(0,red[i]), max(0,nir[i]), 2])

# Class 3: Urban
blue  = add_noise(np.full(n, 0.12), 0.02)
green = add_noise(np.full(n, 0.11), 0.02)
red   = add_noise(np.full(n, 0.13), 0.02)
nir   = add_noise(np.full(n, 0.14), 0.02)
for i in range(n):
    rows.append([max(0,blue[i]), max(0,green[i]), max(0,red[i]), max(0,nir[i]), 3])

# Class 4: Bare soil
blue  = add_noise(np.full(n, 0.09), 0.015)
green = add_noise(np.full(n, 0.10), 0.015)
red   = add_noise(np.full(n, 0.18), 0.03)
nir   = add_noise(np.full(n, 0.22), 0.04)
for i in range(n):
    rows.append([max(0,blue[i]), max(0,green[i]), max(0,red[i]), max(0,nir[i]), 4])

df = pd.DataFrame(rows, columns=["blue", "green", "red", "nir", "class"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("sample_data", exist_ok=True)
df.to_csv("sample_data/training_data.csv", index=False)
print(f"Saved {len(df)} rows → sample_data/training_data.csv")
print(df["class"].value_counts().sort_index())
