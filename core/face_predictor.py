from core.load_models import get_face_model


def predict_face(img_path):
    model = get_face_model()

    # Face model not trained yet
    if model is None:
        return None, None

    # Face detection + preprocessing will be added later
    return "Real", 0.5