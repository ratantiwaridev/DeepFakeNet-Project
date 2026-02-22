from core.global_predictor import predict_global
from core.face_predictor import predict_face


def hybrid_predict(img_path):
    global_label, global_prob = predict_global(img_path)

    face_label, face_prob = predict_face(img_path)

    # If face model available → combine
    if face_prob is not None:
        final_prob = (global_prob + face_prob) / 2
        final_label = "Real" if final_prob > 0.5 else "Fake"
        return final_label, final_prob

    # Otherwise fallback to global
    return global_label, global_prob