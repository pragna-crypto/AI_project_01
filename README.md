# AI Public Speaking Confidence Analyzer

## Goal
A Django web application that allows users to record or upload a speech and analyze their public speaking confidence using AI-based speech analysis. It provides metrics like Speech Speed (WPM), Filler Words count, Pause count, and Voice Stability, and gives an overall Confidence Score.

## Tech Stack
- Django (Backend & Web Framework)
- Python
- Bootstrap (Frontend styling)
- SpeechRecognition (Transcribing audio)
- librosa (Acoustic and stability analysis)
- pydub (Audio file conversion to wav)
- numpy & scipy (Signal processing and calculations)
- Chart.js (Visualizing the data)

## Requirements & Prerequisites
Ensure you have the following installed:
1. Python 3.8+
2. **FFmpeg** (Required by pydub/librosa for audio conversion). Download it and add it to your System PATH if running on Windows.

## Installation & Setup in VS Code

1. **Open the Project Folder** in VS Code.
   Navigate to your project directly (or open `d:\pragna_repose\AI_project_01` in VS Code).

2. **Create a Virtual Environment** 
   Open the internal terminal in VS Code (`Ctrl + \``) and run:
   ```bash
   python -m venv env
   ```

3. **Activate the Virtual Environment**
   - Windows: `env\Scripts\activate`
   - Linux/Mac: `source env/bin/activate`

4. **Install Python Dependencies**
   Run the following command in the terminal to install all required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Database Migration Commands

Before running the server, you need to apply the database migrations for the new `SpeechAnalysis` model we created:

1. Change directory to where `manage.py` is located:
   ```bash
   cd ai_speech_analyzer
   ```

2. Generate the migrations for the `analyzer` app:
   ```bash
   python manage.py makemigrations analyzer
   ```

3. Apply all migrations to your standard SQLite database:
   ```bash
   python manage.py migrate
   ```

## Running the Project

1. Start the Django Development Server:
   ```bash
   python manage.py runserver
   ```

2. Open your web browser and navigate to:
   http://127.0.0.1:8000/

## Features
1. **Record Speech Page**: Allows recording via microphone or uploading audio/video file directly.
2. **Analysis Engine**: Transcribes with `SpeechRecognition` and analyzes volume/pauses with `librosa`.
3. **Results Dashboard**: View your score out of 100 with insights and a Score Breakdown chart.
4. **Speech History**: Keeps track of all prior recordings.
5. **Improvement Dashboard**: Plots progress over time on various speech metrics.