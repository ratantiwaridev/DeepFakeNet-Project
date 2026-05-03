from core.gradcam import generate_gradcam
from core.face_detector import detect_face


def predict_image(img_path):

    face = detect_face(img_path)

    if face:
        result = generate_gradcam(img_path, model_type="face")
        result["model_used"] = "Face Model"
    else:
        result = generate_gradcam(img_path, model_type="global")
        result["model_used"] = "Global Model"

    return result