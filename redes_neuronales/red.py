import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

numeros = np.array([8, 4, 5, 6, 3, 10, 15, 78, 35, 43], dtype=float)
par_impar = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

historial = modelo.fit(numeros, par_impar, epochs=5, verbose=True)
modelo.summary()

plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.plot(historial.history['loss'])
plt.savefig('perdida.png')