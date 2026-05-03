from django.shortcuts import render

# Create your views here.
import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from core.hybrid_predictor import predict_image


def home(request):
    return render(request, "index.html")


def predict(request):
    if request.method == "POST" and request.FILES.get("image"):
        
        image_file = request.FILES["image"]
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        filepath = fs.path(filename)

        # 🔥 Hybrid pipeline
        result = predict_image(filepath)

        context = {
            "image_url": fs.url(filename),
            "label": result["label"],
            "confidence": result["probability"],
        }

        return render(request, "result.html", context)

    return render(request, "index.html")