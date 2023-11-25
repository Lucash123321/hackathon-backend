from django.urls import path
from . import views

app_name = "handler"

urlpatterns = [
    path('', views.main),
    path('handle', views.handle),
    path('paint', views.paint)
    ]