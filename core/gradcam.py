import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Model loaders
from core.load_models import get_global_model, get_face_model

# Preprocessing
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess


# ============================================================
# Image preprocessing (model-aware)
# ============================================================
def preprocess_image(img_path, model_type="global", target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if model_type == "global":
        img_array = mobilenet_preprocess(img_array)
    else:  # face model
        img_array = efficientnet_preprocess(img_array)

    return img_array, img


# ============================================================
# Grad-CAM Core (supports both models)
# ============================================================
def make_gradcam_heatmap(img_array, model, model_type="global"):

    if model_type == "global":
        # Sequential MobileNetV2
        base_model = model.layers[0]
        last_conv_layer_name = "out_relu"
        last_conv_layer = base_model.get_layer(last_conv_layer_name)

        conv_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=last_conv_layer.output
        )

        with tf.GradientTape() as tape:
            conv_outputs = conv_model(img_array)
            tape.watch(conv_outputs)

            x = conv_outputs
            for layer in model.layers[1:]:
                x = layer(x)

            predictions = x
            loss = predictions[:, 0]

    else:
        # EfficientNet face model (Functional)
        last_conv_layer_name = "top_conv"
        last_conv_layer = model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (224, 224),
        method="bilinear"
    )

    heatmap = tf.squeeze(heatmap).numpy()
    prob = predictions.numpy()[0][0]

    return heatmap, prob


# ============================================================
# Public API
# ============================================================
def generate_gradcam(img_path, model_type="global"):

    if model_type == "global":
        model = get_global_model()
    else:
        model = get_face_model()

    img_array, original_img = preprocess_image(img_path, model_type)
    heatmap, prob = make_gradcam_heatmap(img_array, model, model_type)

    label = "Real" if prob > 0.5 else "Fake"

    return {
        "heatmap": heatmap,
        "label": label,
        "probability": float(prob),
        "image": original_img
    }


# ============================================================
# Visualization
# ============================================================
def visualize_gradcam(img_path, model_type="global"):

    result = generate_gradcam(img_path, model_type)

    heatmap = result["heatmap"]
    label = result["label"]
    prob = result["probability"]
    original_img = result["image"]

    color = "green" if label == "Real" else "red"
    title_model = "Face Model" if model_type == "face" else "Global Model"

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.axis("off")
    plt.title(f"{label} ({prob:.2f})", color=color)

    plt.subplot(1, 2, 2)
    plt.imshow(original_img)
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.axis("off")
    plt.title(f"{title_model} Grad-CAM")

    plt.tight_layout()
    plt.show()