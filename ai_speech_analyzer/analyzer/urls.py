from django.urls import path
from . import views

urlpatterns = [
    path('', views.record_speech, name='record_speech'),
    path('analyze/<int:pk>/', views.analyze_speech, name='analyze_speech'),
    path('results/<int:pk>/', views.results_dashboard, name='results_dashboard'),
    path('history/', views.speech_history, name='speech_history'),
    path('dashboard/', views.improvement_dashboard, name='improvement_dashboard'),
]
