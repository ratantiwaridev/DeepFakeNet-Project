import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from core.hybrid_predictor import hybrid_predict

img_path = r"D:\Project v2\datasets\global_dataset\test\real\0001.jpg"

label, prob = hybrid_predict(img_path)

print(label, prob)