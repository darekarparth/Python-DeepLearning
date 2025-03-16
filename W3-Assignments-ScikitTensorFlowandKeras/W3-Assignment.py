# Import libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Dummy data (replace with Ionosphere dataset)
X = np.random.rand(351, 34)  # 351 samples, 34 features
y = np.random.choice(["good", "bad"], 351)  # Binary labels

# Preprocess labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # "good" -> 0, "bad" -> 1

# Define the model
model = Sequential()
model.add(Dense(64, input_shape=(34,), activation="relu"))  # Hidden layer 1
model.add(Dense(32, activation="relu"))  # Hidden layer 2
model.add(Dense(1, activation="sigmoid"))  # Output layer (binary)

# Compile the model
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# Fit the model
model.fit(X, y_encoded, epochs=100, batch_size=32, verbose=0)

# Evaluate (assuming a holdout set)
X_test = X[:50]  # First 50 samples as test set
y_test = y_encoded[:50]
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predict
predictions = model.predict(X_test)
print("First 5 predictions:", (predictions[:5] > 0.5).astype(int))  # Threshold at 0.5