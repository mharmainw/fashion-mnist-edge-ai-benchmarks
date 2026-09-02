import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf # type: ignore

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss="sparse_categorical_crossentropy", 
    metrics = ['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size= 64,
    validation_data=(X_test, y_test)
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

prediction = model.predict(X_test)

prediction.argmax(axis = 1)

model.save('model.keras')

