from django.shortcuts import render
from django.http import JsonResponse



def main(request):
    return render(request, 'index.html')


def handle(request):
    print(request.POST)
    return JsonResponse({'class1': 'data1', 'class2': 'data2', 'class3': 'data3', 'entity': 'data4'})


def paint(request):
    return JsonResponse({"paint": "paint"})