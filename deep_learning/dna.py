import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("hello world!")

# TODO Load penguins.csv
data = pd.read_csv('deep_learning/penguins.csv')

#TODO Handle NA Values
data = data.dropna(axis=0)

# TODO encode string data using LabelEncoder
encoder = LabelEncoder()
cate_cols = data.select_dtypes(exclude="number").columns.tolist()
for col in cate_cols:
    data[col] = encoder.fit_transform(data[col])

#TODO Select your features. Select body_mass_g as your "target" (y) and everything else as X
X = data.drop(columns=["body_mass_g"])
y = data["body_mass_g"]

# TODO : Split the data into testing and training data. Use a 20% split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42
)
# TODO create a neural network with tensorflow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])