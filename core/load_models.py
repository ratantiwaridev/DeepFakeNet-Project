import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

GLOBAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "mobilenetv2_global_final_tuned.keras"
)

FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "efficientnet_face_best.keras"
)

_global_model = None
_face_model = None


def get_global_model():
    global _global_model
    if _global_model is None:
        _global_model = tf.keras.models.load_model(GLOBAL_MODEL_PATH)
    return _global_model


def get_face_model():
    global _face_model
    if _face_model is None and os.path.exists(FACE_MODEL_PATH):
        _face_model = tf.keras.models.load_model(FACE_MODEL_PATH)
    return _face_model