from django.db import models

class SpeechAnalysis(models.Model):
    uploaded_file = models.FileField(upload_to='speeches/')
    transcript = models.TextField(blank=True, null=True)
    speech_speed = models.FloatField(default=0.0)
    filler_word_count = models.IntegerField(default=0)
    pause_count = models.IntegerField(default=0)
    voice_stability = models.FloatField(default=0.0)
    confidence_score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Speech Analysis {self.id} - Score: {self.confidence_score}"
