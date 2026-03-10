import os
from django.shortcuts import render, redirect, get_object_or_404
from .models import SpeechAnalysis
from .forms import UploadSpeechForm
from .speech_analyzer import SpeechAnalyzer
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required

@login_required(login_url='login')
def record_speech(request):
    if request.method == 'POST':
        form = UploadSpeechForm(request.POST, request.FILES)
        if form.is_valid():
            analysis = form.save()
            # If javascript MediaRecorder submits a file, it's captured here
            return redirect('analyze_speech', pk=analysis.pk)
    else:
        form = UploadSpeechForm()
    
    return render(request, 'record.html', {'form': form})

@login_required(login_url='login')
def analyze_speech(request, pk):
    analysis = get_object_or_404(SpeechAnalysis, pk=pk)
    
    # if already analyzed
    if analysis.transcript:
        return redirect('results_dashboard', pk=pk)

    try:
        #get uploaded file path
        file_path = analysis.uploaded_file.path
    
        #check if file exists
        if not os.path.exists(file_path):
            print("File not found:",file_path)
            return redirect('results_dashboard', pk=pk)
        analyzer = speechAnalyzer(file_path)
        
        results = analyzer.analyze()
        
        # Save results
        analysis.transcript = results.get('transcript',"")
        analysis.speech_speed = results.get('speech_speed',0)
        analysis.filler_word_count = results.get('filler_word_count',0)
        analysis.pause_count = results.get('pause_count',0)
        analysis.voice_stability = results.get('voice_stability',0)
        analysis.confidence_score = results.get('confidence_score',0)
        # We can optionally save the score breakdown in a JSONField or related model,
        # but for simplicity we'll just reconstruct or rely on the total score
        analysis.save()
        
        return redirect('results_dashboard', pk=pk)
    except Exception as e:
        print(f"Error analyzing speech: {e}")
        # In a real app we'd show an error message. For demo, just redirect to results (empty or partial).
        return redirect('results_dashboard', pk=pk)

@login_required(login_url='login')
def results_dashboard(request, pk):
    analysis = get_object_or_404(SpeechAnalysis, pk=pk)

    # Simplified suggestion logic
    suggestions = []
    if analysis.speech_speed > 160:
        suggestions.append("Slow down! You are speaking too fast.")
    elif analysis.speech_speed < 120:
        suggestions.append("Try to speak a little faster and more energetically.")
    
    if analysis.filler_word_count > 5:
        suggestions.append("You have too many filler words. Practice pausing instead of saying 'um' or 'uh'.")
    
    if analysis.pause_count < 2:
        suggestions.append("Use more pauses to emphasize points.")
        
    if analysis.voice_stability < 50:
        suggestions.append("Your voice volume isn't stable. Try to maintain a constant tone and volume.")

    # Re-calculate or hardcode the breakdown based on values:
    breakdown = {
        'speed': min(25, 25 - abs(140 - analysis.speech_speed) * 0.1),
        'filler': max(0, 25 - analysis.filler_word_count * 2.5),
        'pause': max(0, 25 - abs(10 - analysis.pause_count) * 1.5),
        'stability': min(25, (analysis.voice_stability / 100) * 25)
    }

    context = {
        'analysis': analysis,
        'suggestions': suggestions,
        'breakdown': breakdown,
    }
    return render(request, 'results.html', context)

@login_required(login_url='login')
def speech_history(request):
    history = SpeechAnalysis.objects.all().order_by('-created_at')
    return render(request, 'history.html', {'history': history})

@login_required(login_url='login')
def improvement_dashboard(request):
    # Fetch all records ordered by date
    history = SpeechAnalysis.objects.all().order_by('created_at')
    
    dates = [h.created_at.strftime("%Y-%m-%d %H:%M") for h in history]
    scores = [h.confidence_score for h in history]
    speeds = [h.speech_speed for h in history]
    
    context = {
        'dates': dates,
        'scores': scores,
        'speeds': speeds,
    }
    return render(request, 'dashboard.html', context)
