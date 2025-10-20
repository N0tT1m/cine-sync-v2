@echo off
REM Simple batch script to start the movie recommendation server on Windows
REM This is an alternative to the PowerShell script for users who prefer batch files

setlocal EnableDelayedExpansion

echo ================================
echo   Lupe Movie Recommendation Server
echo ================================
echo.

REM Check if Rust is installed
where cargo >nul 2>nul
if !errorlevel! neq 0 (
    echo ERROR: Rust/Cargo not found in PATH
    echo Please install Rust from https://rustup.rs/
    pause
    exit /b 1
)

REM Set environment variables
set RUST_LOG=info
set TORCH_CUDA_VERSION=cu118

REM Default configuration
set MODELS_PATH=models
set PORT=3000
set HOST=127.0.0.1

REM Parse command line arguments
:parse_args
if "%1"=="" goto end_parse
if "%1"=="--models-path" (
    set MODELS_PATH=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--port" (
    set PORT=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--host" (
    set HOST=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--cpu-only" (
    set CPU_ONLY=true
    shift
    goto parse_args
)
if "%1"=="--help" (
    goto show_help
)
shift
goto parse_args
:end_parse

REM Check if models directory exists
if not exist "%MODELS_PATH%" (
    echo ERROR: Models directory not found: %MODELS_PATH%
    echo Please train your model first or specify correct path with --models-path
    pause
    exit /b 1
)

REM Check if server executable exists
if not exist "target\release\movie-recommendation-server.exe" (
    echo Server not built yet. Building now...
    echo.
    cargo build --release
    if !errorlevel! neq 0 (
        echo ERROR: Failed to build server
        pause
        exit /b 1
    )
    echo Build successful!
    echo.
)

REM Export model artifacts if Python is available
where python >nul 2>nul
if !errorlevel! equ 0 (
    if exist "export_metadata_for_rust.py" (
        echo Exporting model artifacts...
        python export_metadata_for_rust.py --models-path "%MODELS_PATH%"
        echo.
    )
) else (
    where py >nul 2>nul
    if !errorlevel! equ 0 (
        if exist "export_metadata_for_rust.py" (
            echo Exporting model artifacts...
            py export_metadata_for_rust.py --models-path "%MODELS_PATH%"
            echo.
        )
    ) else (
        echo WARNING: Python not found. Skipping model artifact export.
        echo.
    )
)

REM Check for NVIDIA GPU
nvidia-smi >nul 2>nul
if !errorlevel! equ 0 (
    echo GPU detected. Server will use GPU acceleration.
) else (
    echo No GPU detected. Server will run on CPU.
)
echo.

REM Build command
set COMMAND=target\release\movie-recommendation-server.exe --models-path "%MODELS_PATH%" --port %PORT% --host %HOST%
if defined CPU_ONLY (
    set COMMAND=!COMMAND! --cpu-only
)

echo Starting Lupe Movie Recommendation Server...
echo Models Path: %MODELS_PATH%
echo Server URL: http://%HOST%:%PORT%
echo Command: !COMMAND!
echo.
echo Press Ctrl+C to stop the server
echo ================================
echo.

REM Start the server
!COMMAND!

echo.
echo Server stopped.
pause
exit /b 0

:show_help
echo Usage: start_server.bat [OPTIONS]
echo.
echo Options:
echo   --models-path PATH    Path to models directory (default: models)
echo   --port PORT          Server port (default: 3000)
echo   --host HOST          Server host (default: 127.0.0.1)
echo   --cpu-only           Force CPU usage
echo   --help               Show this help
echo.
echo Examples:
echo   start_server.bat
echo   start_server.bat --port 8080
echo   start_server.bat --models-path "C:\my_models" --cpu-only
echo.
pause
exit /b 0