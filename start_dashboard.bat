@echo off
start cmd /k "python flask_app.py"
timeout /t 3 /nobreak > NUL
start http://127.0.0.1:5000
