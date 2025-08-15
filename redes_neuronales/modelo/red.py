import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv("clasificacion/datos/datos.csv")
X = df['numeros'].to_numpy().astype(int).reshape(-1, 1)
y = df['par_impar'].to_numpy().astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data:", X_train.flatten())
print("Training labels:", y_train)
print("Testing data:", X_test.flatten())
print("Testing labels:", y_test)

# Build and compile model
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=[1])
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Comenzando entrenamiento...")

# Train model
historial = modelo.fit(
    X_train, 
    y_train,
    validation_data=(X_test, y_test),
    epochs=500,
    verbose=1
)

modelo.save('modelo/modelo.keras')
print("Modelo guardado")

# Evaluate model
test_loss, test_acc = modelo.evaluate(X_test, y_test)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Generate predictions
predictions = (modelo.predict(X_test) > 0.5).astype(int)
print("Predictions:", predictions.flatten())
print("Actual values:", y_test)

# Plot training history
plt.figure(figsize=(8, 5))
plt.plot(historial.history['loss'], label='Train Loss')
plt.plot(historial.history['val_loss'], label='Validation Loss')
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()
plt.savefig('perdida.png')
plt.close()

# Save model
modelo.save('modelo_par_impar.h5')
print("Modelo guardado correctamente")