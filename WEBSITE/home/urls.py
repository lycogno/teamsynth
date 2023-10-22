from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='index'),
    path('home/', views.home, name='home'),
    path('teams/', views.teams, name='teams'),
    path('teams/<int:team_size>', views.teams2, name='teams2'),
    path('result/', views.result, name='result'),
]
