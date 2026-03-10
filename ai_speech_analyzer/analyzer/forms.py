from django import forms
from .models import SpeechAnalysis

class UploadSpeechForm(forms.ModelForm):
    class Meta:
        model = SpeechAnalysis
        fields = ['uploaded_file']
        widgets = {
            'uploaded_file': forms.FileInput(attrs={'class': 'form-control', 'accept': 'audio/*,video/*'})
        }
