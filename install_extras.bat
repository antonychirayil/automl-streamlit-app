@echo off
REM ---------------------------------------------------------------
REM  One-click installer for the AutoML app's Python 3.13 environment.
REM  Double-click this file. It creates/uses venv313, installs the core
REM  requirements, then installs the EDA extras (ydata-profiling +
REM  sweetviz), and finally reports what actually landed.
REM ---------------------------------------------------------------
cd /d "%~dp0"

echo ===============================================================
echo   AutoML app - installing into venv313 (Python 3.13)
echo ===============================================================
echo.

py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo   [X] Python 3.13 was not found by the "py" launcher.
    echo       ydata-profiling supports Python 3.10-3.13 only, so it
    echo       cannot be installed on Python 3.14. Install Python 3.13
    echo       first, then run this file again.
    echo.
    pause
    exit /b 1
)

py -3.13 --version
echo.

if exist "venv313\Scripts\activate.bat" (
    echo   [1/5] Reusing the existing venv313 environment.
) else (
    echo   [1/5] Creating a Python 3.13 environment in venv313 ...
    py -3.13 -m venv venv313
)
if not exist "venv313\Scripts\activate.bat" goto failed
echo.

echo   [2/5] Activating venv313 ...
call "venv313\Scripts\activate.bat"
echo.

echo   [3/5] Upgrading pip and installing setuptools ^(pinned below 82^) ...
echo         ydata-profiling imports pkg_resources, which lives inside
echo         setuptools - but setuptools 82 deleted it, and Python 3.12+
echo         no longer puts setuptools in new venvs at all. So we need
echo         setuptools, and specifically a version older than 82.
python -m pip install --upgrade pip
python -m pip install "setuptools<82"
if errorlevel 1 goto failed
echo.

echo   [4/5] Installing core requirements (streamlit, pandas, sklearn...) ...
pip install -r requirements.txt
if errorlevel 1 goto failed
echo.

echo   [5/5] Installing EDA extras (ydata-profiling, sweetviz) ...
echo         This one is big - a few minutes is normal.
pip install -r requirements-extras.txt
if errorlevel 1 goto failed
echo.

echo ===============================================================
echo   Finished. Checking what is importable now:
echo ===============================================================
python -c "import importlib.util as u; print('  streamlit       :', 'OK' if u.find_spec('streamlit') else 'MISSING')"
python -c "import importlib.util as u; print('  sweetviz        :', 'OK' if u.find_spec('sweetviz') else 'MISSING')"
python -c "import importlib.util as u; print('  ydata-profiling :', 'OK' if u.find_spec('ydata_profiling') else 'MISSING')"
echo.
echo   To start the app in this environment:
echo       venv313\Scripts\activate
echo       streamlit run app.py
echo.
pause
exit /b 0

:failed
echo.
echo   [X] A step above failed. Scroll up to read the actual error
echo       message - that text is what matters, not this line.
echo.
pause
exit /b 1
