from django.shortcuts import render
from . import forms

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Create your views here.
def index(request):
    return render(request, 'imgUpload/index.html')

def imageprocess(request):
    form = forms.ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        handle_uploaded_file(request.FILES['image'])
        
        model = ResNet50(weights='imagenet')
        img_path = 'img.jpg'
        
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
       
        preds = model.predict(x)
        print("Predicted: ", decode_predictions(preds, top=3)[0])
        
        html = decode_predictions(preds, top=3)[0]
        res = []
        for e in html:
            res.append((e[1], np.round(e[2]*100, 2)))
        context = {'res' : res}
        return render(request, 'imgUpload/result.html', context=context)

    return render(request, 'imgUpload/result.html')


def handle_uploaded_file(f):
    with open('img.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)