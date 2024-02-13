from django.shortcuts import render
from django.views import View
from .userPred import new_prediction

# Create your views here.

class HomeView(View):
    def get(self, request, *args, **kwargs):
        return render(request, "home/index.html")
    

    def post(self, request, *args, **kwargs):
        audio = request.FILES['audio']
        prediction = new_prediction(audio)
        print(prediction)
        context = {}

        if prediction == 0:
            context["pred"] = "ANGRY"
        elif prediction == 1:
            context["pred"] = "HAPPY"
        elif prediction == 1:
            context["pred"] = "SAD"

        print(context)
        
        return render(request, "home/prediction.html", context)