import tensorflow as tf
from src.model import build_simple_cnn

# Demo using MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None]/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)

model = build_simple_cnn((28,28,1), 10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.1)
