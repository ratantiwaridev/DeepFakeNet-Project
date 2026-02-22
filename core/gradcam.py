import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from core.load_models import get_global_model


# ============================================================
# Image preprocessing
# ============================================================
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img


# ============================================================
# Grad-CAM (Ultra-stable version)
# ============================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    """
    Stable for:
    Sequential([
        MobileNetV2,
        GAP,
        Dense,
        ...
    ])
    """

    # Get base MobileNet
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Model: input → conv output
    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # Forward pass through base model
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)

        # Now pass through classifier head
        x = conv_outputs
        for layer in model.layers[1:]:
            x = layer(x)

        predictions = x
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    # Resize to image size
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (224, 224),
        method="bilinear"
    )
    heatmap = tf.squeeze(heatmap).numpy()

    prob = predictions.numpy()[0][0]

    return heatmap, prob


# ============================================================
# Public API (for Django / pipeline)
# ============================================================
def generate_gradcam(img_path):
    model = get_global_model()

    img_array, original_img = preprocess_image(img_path)
    heatmap, prob = make_gradcam_heatmap(img_array, model)

    label = "Real" if prob > 0.5 else "Fake"

    return {
        "heatmap": heatmap,
        "label": label,
        "probability": float(prob),
        "image": original_img
    }


# ============================================================
# Visualization (Notebook / Demo)
# ============================================================
def visualize_gradcam(img_path):
    result = generate_gradcam(img_path)

    heatmap = result["heatmap"]
    label = result["label"]
    prob = result["probability"]
    original_img = result["image"]

    color = "green" if label == "Real" else "red"

    plt.figure(figsize=(8, 4))

    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.axis("off")
    plt.title(f"{label} ({prob:.2f})", color=color)

    # Grad-CAM
    plt.subplot(1, 2, 2)
    plt.imshow(original_img)
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.axis("off")
    plt.title("Grad-CAM")

    plt.tight_layout()
    plt.show()