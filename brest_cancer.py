import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

df = pd.read_csv(r"C:\Users\Harsh\Downloads\data.csv").drop(["Unnamed: 32"], axis=1)
df_target = pd.get_dummies(df["diagnosis"])*1
df_target = df_target[[df_target.columns[0]]]
df_target.columns = ["result"]
df["diagnosis"] = df_target["result"]
df = df.drop(["id"], axis=1)

x = df.drop(["diagnosis"], axis=1)
y = df["diagnosis"]

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7)

model = GradientBoostingClassifier(verbose=1)
model.fit(x_train,y_train)

print(model.score(x_test,y_test))

model = Sequential([
    Dense(10000, activation="relu"),
    Dense(300, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train, epochs = 50, verbose = 0)
print(model.evaluate(x_test,y_test))