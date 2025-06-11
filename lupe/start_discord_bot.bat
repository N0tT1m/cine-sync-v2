@echo off
REM Windows batch script to start the Lupe Discord bot

setlocal EnableDelayedExpansion

echo ================================
echo     Lupe Discord Bot
echo ================================
echo.

REM Check if .env file exists
if not exist ".env" (
    echo ERROR: .env file not found
    echo Please copy .env.example to .env and configure your Discord token
    echo.
    echo Example .env content:
    echo DISCORD_TOKEN=your_bot_token_here
    echo API_BASE_URL=http://localhost:3000
    echo COMMAND_PREFIX=!
    echo.
    pause
    exit /b 1
)

REM Load environment variables from .env file
for /f "usebackq tokens=1,2 delims==" %%i in (".env") do (
    if not "%%i"=="" if not "%%j"=="" (
        set "%%i=%%j"
    )
)

REM Check if Discord token is set
if not defined DISCORD_TOKEN (
    echo ERROR: DISCORD_TOKEN not found in .env file
    echo Please add your Discord bot token to the .env file:
    echo DISCORD_TOKEN=your_bot_token_here
    echo.
    pause
    exit /b 1
)

REM Check if Rust is installed
where cargo >nul 2>nul
if !errorlevel! neq 0 (
    echo ERROR: Rust/Cargo not found in PATH
    echo Please install Rust from https://rustup.rs/
    pause
    exit /b 1
)

REM Set default values if not in .env
if not defined API_BASE_URL set API_BASE_URL=http://localhost:3000
if not defined COMMAND_PREFIX set COMMAND_PREFIX=!
if not defined MODEL_TYPE set MODEL_TYPE=hybrid
if not defined RUST_LOG set RUST_LOG=info

echo Configuration:
echo - Discord Token: ****%DISCORD_TOKEN:~-4%
echo - API Base URL: %API_BASE_URL%
echo - Command Prefix: %COMMAND_PREFIX%
echo - Model Type: %MODEL_TYPE%
echo.

REM Test API connection
echo Testing connection to movie recommendation API...
curl -s "%API_BASE_URL%/health" >nul 2>nul
if !errorlevel! equ 0 (
    echo ✅ API connection successful
) else (
    echo ⚠️  WARNING: Cannot connect to API at %API_BASE_URL%
    echo Make sure the movie recommendation server is running first
    echo.
    set /p continue="Continue anyway? (y/N): "
    if /i not "!continue!"=="y" (
        echo Exiting...
        pause
        exit /b 1
    )
)
echo.

REM Check if bot is built
if not exist "target\release\lupe-discord-bot.exe" (
    echo Bot not built yet. Building now...
    echo This may take a few minutes on first build...
    echo.
    cargo build --release
    if !errorlevel! neq 0 (
        echo ERROR: Failed to build Discord bot
        pause
        exit /b 1
    )
    echo Build successful!
    echo.
)

echo Starting Lupe Discord Bot...
echo Press Ctrl+C to stop the bot
echo ================================
echo.

REM Start the bot
target\release\lupe-discord-bot.exe

echo.
echo Bot stopped.
pause
exit /b 0