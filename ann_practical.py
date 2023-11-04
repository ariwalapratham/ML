import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

df=pd.read_csv('/kaggle/input/heart-attack-analysis-prediction-dataset/heart.csv')
df.head()

df.shape

# df['type'] = df['type'].replace({'white': 0, 'red': 1})

df.isnull().sum()

# df.dropna(inplace=True)

X = df.drop(columns=['output'])
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  
    layers.Dense(64, activation='relu'),  
    layers.Dense(32, activation='relu'), 
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()



history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')


new_sample = np.array([56,1,1,120,236,0,1,178,0,0.8,2,0,2]) 
# Replace 'your_input_features' with actual data
new_sample = new_sample.reshape(1, -1)
scaled_sample = scaler.transform(new_sample)
predicted_probability = model.predict(scaled_sample)
predicted_class = 1 if predicted_probability >= 0.5 else 0
print(f'Predicted Class: {predicted_class}')