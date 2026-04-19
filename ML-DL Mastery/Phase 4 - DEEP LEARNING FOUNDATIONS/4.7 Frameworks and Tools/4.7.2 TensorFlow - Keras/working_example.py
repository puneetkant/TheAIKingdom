"""
Working Example: TensorFlow / Keras
Covers Sequential API, Functional API, custom layers, callbacks,
model compilation, tf.data pipeline, and saving/loading.
Runs gracefully when TensorFlow is not installed.
"""

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # suppress TF C++ logs


# -- helper --------------------------------------------------------------------
def code_block(code: str):
    for line in code.strip().splitlines():
        print(f"  {line}")


# -- 1. Sequential API ---------------------------------------------------------
def sequential_api():
    print("=== Keras Sequential API ===")
    if not HAS_TF:
        code_block("""
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(20,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=64,
                    validation_split=0.1)
        """)
        return

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                               n_informative=10, random_state=0)
    X = StandardScaler().fit_transform(X).astype("float32")
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=0)

    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(20,)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(Xtr, ytr, epochs=20, batch_size=32,
                        validation_split=0.1, verbose=0)
    _, acc = model.evaluate(Xts, yts, verbose=0)
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Final val_loss: {history.history['val_loss'][-1]:.4f}")


# -- 2. Functional API ---------------------------------------------------------
def functional_api():
    print("\n=== Keras Functional API ===")
    print("  Use for: multi-input/output, residual connections, DAG-style networks")
    if not HAS_TF:
        code_block("""
# Multi-input example
inp_a = keras.Input(shape=(20,), name='numerical')
inp_b = keras.Input(shape=(100,), name='text_embed')

x_a = layers.Dense(64, activation='relu')(inp_a)
x_b = layers.Dense(64, activation='relu')(inp_b)
x   = layers.Concatenate()([x_a, x_b])
x   = layers.Dense(32, activation='relu')(x)
out = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=[inp_a, inp_b], outputs=out)
        """)
        return

    inp = keras.Input(shape=(20,))
    x   = layers.Dense(64, activation='relu')(inp)
    # Residual block
    shortcut = layers.Dense(32)(x)
    x        = layers.Dense(32, activation='relu')(x)
    x        = layers.Dense(32, activation='relu')(x)
    x        = layers.Add()([x, shortcut])
    x        = layers.LayerNormalization()(x)
    out      = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inp, outputs=out)
    print(f"  Functional model: {len(model.layers)} layers  "
          f"params={model.count_params():,}")


# -- 3. Custom layers ----------------------------------------------------------
def custom_layers():
    print("\n=== Custom Keras Layers ===")
    if not HAS_TF:
        code_block("""
class MyDense(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight("W", shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight("b", shape=(self.units,),
                                 initializer='zeros', trainable=True)

    def call(self, x):
        return tf.matmul(x, self.W) + self.b


# Custom model with train_step override
class MyModel(keras.Model):
    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            y_hat = self(X, training=True)
            loss  = self.compiled_loss(y, y_hat)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}
        """)
        return

    class GeLULayer(layers.Layer):
        def call(self, x):
            return x * 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))

    inp = keras.Input(shape=(20,))
    x   = layers.Dense(64)(inp)
    x   = custom_layers.GeLULayer()(x)
    out = layers.Dense(1)(x)
    m   = keras.Model(inp, out)
    print(f"  Custom GeLU model built: {m.count_params()} params")


# -- 4. Callbacks --------------------------------------------------------------
def callbacks_demo():
    print("\n=== Keras Callbacks ===")
    callback_table = [
        ("EarlyStopping",   "monitor='val_loss', patience=10, restore_best_weights=True"),
        ("ReduceLROnPlateau","monitor='val_loss', factor=0.5, patience=5"),
        ("ModelCheckpoint", "filepath='best.keras', save_best_only=True"),
        ("TensorBoard",     "log_dir='./logs', histogram_freq=1"),
        ("LambdaCallback",  "on_epoch_end=lambda e, log: print(log)"),
        ("LearningRateScheduler", "schedule=lambda ep: 1e-3 * 0.9**ep"),
    ]
    print(f"  {'Callback':<25} {'Common kwargs'}")
    print(f"  {'-'*25} {'-'*45}")
    for name, kwargs in callback_table:
        print(f"  {name:<25} {kwargs}")

    if not HAS_TF:
        return

    # Quick model with early stopping
    X, y = make_classification(n_samples=500, n_features=10, random_state=1)
    X    = StandardScaler().fit_transform(X).astype("float32")
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=1)

    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(10,)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    cb = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                        verbose=0)]
    hist = model.fit(Xtr, ytr, epochs=100, validation_split=0.15,
                     callbacks=cb, verbose=0)
    stopped_at = len(hist.history['loss'])
    _, acc = model.evaluate(Xts, yts, verbose=0)
    print(f"\n  EarlyStopping demo: stopped at epoch {stopped_at}/100, test acc={acc:.4f}")


# -- 5. tf.data pipeline -------------------------------------------------------
def tf_data_pipeline():
    print("\n=== tf.data Pipeline ===")
    print("  Performance: prefetch, cache, and parallel map prevent GPU stalls")
    code_block("""
# Typical production pipeline
dataset = (
    tf.data.Dataset.from_tensor_slices((X, y))
    .cache()                              # cache in RAM after first epoch
    .shuffle(buffer_size=1000, seed=42)
    .batch(batch_size=64)
    .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)           # prepare next batch while GPU trains
)

# From files
dataset = tf.data.Dataset.list_files("data/*.tfrecord")
dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_reads=4)
dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    """)

    if not HAS_TF:
        return

    X = np.random.randn(200, 4).astype("float32")
    y = (X[:, 0] + X[:, 1] > 0).astype("float32")

    ds = (tf.data.Dataset.from_tensor_slices((X, y))
          .shuffle(200).batch(32).prefetch(tf.data.AUTOTUNE))
    batches = list(ds)
    print(f"\n  tf.data demo: {len(batches)} batches from 200 samples (bs=32)")
    print(f"  First batch shape: X={batches[0][0].shape}, y={batches[0][1].shape}")


# -- 6. Save and load ----------------------------------------------------------
def save_load_patterns():
    print("\n=== Saving and Loading Models ===")
    code_block("""
# Save entire model (recommended)
model.save('my_model.keras')               # Keras v3 format
model = keras.models.load_model('my_model.keras')

# Weights only
model.save_weights('weights.h5')
model.load_weights('weights.h5')

# SavedModel format (TF serving compatible)
model.export('saved_model_dir/')

# TFLite for mobile / edge
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    """)


if __name__ == "__main__":
    sequential_api()
    functional_api()
    callbacks_demo()
    tf_data_pipeline()
    save_load_patterns()
