import os, sys, traceback
with open('error.log', 'w') as f:
    sys.stderr = f
    sys.stdout = f
    if __name__ == '__main__':
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_speech_analyzer.settings')
        try:
            from django.core.management import execute_from_command_line
            execute_from_command_line(['manage.py', 'runserver', '--noreload'])
        except BaseException as e:
            traceback.print_exc()
