"""Train the Stage 2 MNIST CNN and export a fully-INT8 TFLite model.

Run with the TensorFlow environment:
    .venv-tf/bin/python sw/model/train_mnist.py --out models/mnist_int8.tflite

Architecture (matches the NPU program in the design plan):
    28x28x1 -> conv3x3(8)+ReLU -> maxpool2 -> conv3x3(16)+ReLU -> maxpool2 -> dense(10)
Output is int8 logits (no softmax); the NPU takes the argmax.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

SEED = 0


def build_model() -> tf.keras.Model:
    inp = tf.keras.Input((28, 28, 1), name="image")
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu", name="conv1")(inp)
    x = tf.keras.layers.MaxPooling2D(2, name="pool1")(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    out = tf.keras.layers.Dense(10, name="fc")(x)
    return tf.keras.Model(inp, out, name="mnist_npu")


def augmenter() -> tf.keras.Sequential:
    """Geometric + photometric jitter so the model tolerates a hand-held camera view."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomTranslation(0.12, 0.12, fill_mode="constant", seed=SEED),
        tf.keras.layers.RandomRotation(0.04, fill_mode="constant", seed=SEED),   # about ±14°
        tf.keras.layers.RandomZoom((-0.2, 0.15), fill_mode="constant", seed=SEED),
    ])


def stroke_and_noise(x):
    # Randomly thicken strokes (camera blur/marker pens) and add sensor-like noise.
    thick = tf.nn.max_pool2d(x, 2, 1, "SAME")
    x = tf.where(tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.3, thick, x)
    x = x + tf.random.normal(tf.shape(x), stddev=0.04)
    return tf.clip_by_value(x, 0.0, 1.0)


def datasets(batch: int):
    (x, y), (xt, yt) = tf.keras.datasets.mnist.load_data()
    x = (x[..., None] / 255.0).astype("float32")
    xt = (xt[..., None] / 255.0).astype("float32")
    aug = augmenter()
    train = (tf.data.Dataset.from_tensor_slices((x, y))
             .shuffle(60000, seed=SEED).batch(batch)
             .map(lambda a, b: (stroke_and_noise(aug(a, training=True)), b),
                  num_parallel_calls=tf.data.AUTOTUNE)
             .prefetch(tf.data.AUTOTUNE))
    test = tf.data.Dataset.from_tensor_slices((xt, yt)).batch(512)
    return train, test, x, xt, yt


def to_int8_tflite(model, calib: np.ndarray) -> bytes:
    def rep():
        for i in range(0, 1000, 10):
            yield [calib[i:i + 10]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def int8_accuracy(tflite: bytes, xt: np.ndarray, yt: np.ndarray) -> float:
    itp = tf.lite.Interpreter(
        model_content=tflite,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    itp.allocate_tensors()
    i_det, o_det = itp.get_input_details()[0], itp.get_output_details()[0]
    scale, zp = i_det["quantization"]
    correct = 0
    for img, label in zip(xt, yt):
        q = np.clip(np.round(img / scale + zp), -128, 127).astype(np.int8)
        itp.set_tensor(i_det["index"], q[None])
        itp.invoke()
        correct += int(np.argmax(itp.get_tensor(o_det["index"])[0]) == label)
    return correct / len(yt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/mnist_int8.tflite")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    train, test, x, xt, yt = datasets(args.batch)
    model = build_model()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.fit(train, epochs=args.epochs, validation_data=test, verbose=2,
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)])

    _, float_acc = model.evaluate(test, verbose=0)
    tflite = to_int8_tflite(model, x)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite)

    q_acc = int8_accuracy(tflite, xt, yt)
    print(f"float accuracy: {float_acc:.4f}")
    print(f"int8  accuracy: {q_acc:.4f}  (TFLite reference kernels)")
    print(f"params: {model.count_params()}  tflite: {len(tflite)} bytes -> {out}")


if __name__ == "__main__":
    main()
