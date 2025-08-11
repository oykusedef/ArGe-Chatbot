# Backend'i başlat
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd backend; python app.py'

# Frontend'i başlat
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd frontend; npm start'
