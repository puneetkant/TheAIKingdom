# 4.7.2 TensorFlow / Keras

Keras Sequential API, `model.fit()`, callbacks (EarlyStopping, ModelCheckpoint), TF data pipeline.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Keras functional API with custom metrics |
| `working_example2.py` | Sequential API end-to-end training with EarlyStopping + plots |
| `working_example.ipynb` | Interactive: build → compile → fit → evaluate |

## Quick Reference

```python
import tensorflow as tf
from tensorflow import keras

# Sequential model
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(n_features,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best.h5", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
]
history = model.fit(X_train, y_train, validation_split=0.1,
                    epochs=200, batch_size=32, callbacks=callbacks)
```

## Keras vs PyTorch

| Feature | Keras | PyTorch |
|---------|-------|---------|
| Training loop | `model.fit()` | Custom loop |
| Debugging | Less transparent | torch.autograd |
| Production | TFServing/SavedModel | TorchScript/ONNX |
| Research flex | Medium | High |

## Learning Resources
- [Keras docs](https://keras.io/api/)
- [TF tutorials](https://www.tensorflow.org/tutorials)

Build a simple model with Keras APIs.

## What to build

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
