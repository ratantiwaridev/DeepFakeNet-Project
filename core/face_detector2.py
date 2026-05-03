from PIL import Image
import numpy as np

# Simple heuristic (can upgrade later)
def detect_face(img_path):
    try:
        img = Image.open(img_path)
        img = np.array(img)

        # Placeholder logic (upgrade later)
        h, w, _ = img.shape

        # If image is portrait-like → assume face
        if h < w * 1.2:
            return True

        return False

    except:
        return False