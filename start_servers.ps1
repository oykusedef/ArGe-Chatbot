# ArGe Chatbot - Backend ve Frontend Başlatma Scripti
Write-Host "ArGe Chatbot sunucuları başlatılıyor..." -ForegroundColor Green

# Backend sunucusunu başlat
Write-Host "Backend sunucusu başlatılıyor (http://localhost:8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; uvicorn app:app --reload --host 0.0.0.0 --port 8000"

# 3 saniye bekle
Start-Sleep -Seconds 3

# Frontend sunucusunu başlat
Write-Host "Frontend sunucusu başlatılıyor (http://localhost:3000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm start"

Write-Host "Sunucular başlatıldı!" -ForegroundColor Green
Write-Host "Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "Tarayıcıda http://localhost:3000 adresine git" -ForegroundColor White 