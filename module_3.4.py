import numpy as np
import pandas as pd
import tensorflow as tf

raw_dataset = pd.read_csv('datasets/auto-mpg.csv', names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

columns_names = ['mpg','cylinders','displacement','horsepower','weight',
                 'acceleration','model year','origin']


df = raw_dataset.copy()
df.head()
df.shape

df.keys()
df.columns

df.isna().sum()

df1 = df.dropna()
df1.shape

mean_hp = df['horsepower'].mean()
df2 = df.fillna(mean_hp)

df.shape
df1.shape
df2.shape

df2.isna().sum()

df = df2.copy()
df.dtypes
df['origin'].tail()

df['origin'] = df['origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
df['origin'].tail()

df.get_dummies(df, prefix='', prefix_sep='')
df.head()

train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)


train_dataset = train_dataset.pop('mpg')
test_dataset = test_dataset.pop('mpg')

