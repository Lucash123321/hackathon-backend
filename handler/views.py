from django.shortcuts import render
from django.http import JsonResponse
from neuro.neuro import MessageResults
import numpy


def main(request):
    return render(request, 'index.html')


def handle(request):
    Classifier = MessageResults()
    Result = Classifier.PredictClass1([request.POST["text"]])
    data1 = numpy.argmax(Result, axis=1).tolist()
    Result = Classifier.PredictClass1([request.POST["text"]])
    data2 = numpy.argmax(Result, axis=1).tolist()
    Result = Classifier.PredictClass1([request.POST["text"]])
    data3 = numpy.argmax(Result, axis=1).tolist()
    return JsonResponse({'class1': data1, 'class2': data2, 'class3': data3, 'entity': 'data4'})


def paint(request):
    print(request.GET)
    class_id = request.GET["class_id"]
    Classifier = MessageResults()
    if class_id == 'class1':
        Result = Classifier.GetColorDataClass1([request.GET["text"]]).base_values
    elif class_id == 'class2':
        Result = Classifier.GetColorDataClass2([request.GET["text"]]).base_values
    elif class_id == 'class3':
        Result = Classifier.GetColorDataClass3([request.GET["text"]]).base_values
    else:
        Result = ''
    print(Result)
    return JsonResponse({"paint": Result})