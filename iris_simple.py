# iris_ann_deep.py
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Convert to DataFrame for visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = iris.target

# 2. Visualize dataset (heatmap of correlations)
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 3. Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Build deeper ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 7. Compile and train
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

# 8. Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.2f}")

# 9. Visualize ANN architecture
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)