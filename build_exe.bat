@echo off
REM این اسکریپت پروژه را به یک فایل exe مستقل تبدیل می‌کند
REM قبل از اجرا: pip install pyinstaller

pyinstaller --noconfirm --onefile --windowed ^
  --name AgeEstimator ^
  --add-data "models;models" ^
  --add-data "fonts;fonts" ^
  age_estimator.py

echo.
echo ساخته شد. فایل exe در پوشه dist قرار دارد: dist\AgeEstimator.exe
pause
