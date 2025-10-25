@echo off
echo Setting up PostgreSQL with Docker for CineSync Application...
echo.

REM Check if Docker is installed and running
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not installed or not in PATH.
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo Docker found. Checking if Docker is running...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo Docker is running. Proceeding with PostgreSQL setup...
echo.

REM Stop and remove existing containers (if any)
echo Cleaning up existing containers...
docker-compose down 2>nul
docker container rm cinesync-postgres 2>nul

echo.
echo Starting PostgreSQL container...
docker-compose up -d postgres

if %errorlevel% neq 0 (
    echo Failed to start PostgreSQL container.
    echo Please check docker-compose.yml and try again.
    pause
    exit /b 1
)

echo.
echo Waiting for PostgreSQL to be ready...
:wait_loop
docker-compose exec postgres pg_isready -U postgres -d cinesync >nul 2>&1
if %errorlevel% neq 0 (
    echo PostgreSQL is starting up... waiting 3 seconds
    timeout /t 3 /nobreak >nul
    goto wait_loop
)

echo PostgreSQL is ready!
echo.

REM Create .env file for the application
echo Creating .env file with Docker configuration...
echo NOTE: Please edit .env file and set a secure password!
(
echo DB_HOST=localhost
echo DB_NAME=cinesync
echo DB_USER=postgres
echo DB_PASSWORD=CHANGE_THIS_PASSWORD
echo DB_PORT=5432
echo.
echo # PostgreSQL Docker Configuration
echo POSTGRES_DB=cinesync
echo POSTGRES_USER=postgres
echo POSTGRES_PASSWORD=CHANGE_THIS_PASSWORD
echo.
echo DISCORD_TOKEN=
echo DEBUG=true
) > .env

echo.
echo Creating environment setup batch file...
(
echo @echo off
echo set DB_HOST=localhost
echo set DB_NAME=cinesync
echo set DB_USER=postgres
echo set DB_PASSWORD=CHANGE_THIS_PASSWORD
echo set POSTGRES_DB=cinesync
echo set POSTGRES_USER=postgres
echo set POSTGRES_PASSWORD=CHANGE_THIS_PASSWORD
echo set DB_PORT=5432
echo set DISCORD_TOKEN=
echo set DEBUG=true
echo echo Environment variables set for current session.
echo echo WARNING: Please set secure passwords before use!
echo echo Please set DISCORD_TOKEN manually: set DISCORD_TOKEN=your_token_here
) > set_env.bat

echo.
echo Testing database connection...
docker-compose exec postgres psql -U postgres -d cinesync -c "SELECT 'Database connection successful!' as status;"

if %errorlevel% equ 0 (
    echo Database connection test passed!
) else (
    echo Database connection test failed.
)

echo.
echo ========================================
echo Docker PostgreSQL Setup Complete!
echo ========================================
echo.
echo Container: cinesync-postgres
echo Database: cinesync
echo Host: localhost
echo Port: 5432
echo User: postgres
echo.
echo IMPORTANT: Edit .env file and set secure passwords!
echo.
echo Next steps:
echo 1. Set your Discord bot token: set DISCORD_TOKEN=your_token_here
echo 2. Load your movie data into the database
echo 3. Run your Python application
echo.
echo Docker commands:
echo - View logs: docker-compose logs postgres
echo - Stop database: docker-compose down
echo - Start database: docker-compose up -d postgres
echo - Connect to database: docker-compose exec postgres psql -U postgres -d cinesync
echo.
echo Files created:
echo - docker-compose.yml (Docker configuration)
echo - init-db.sql (Database schema)
echo - .env (environment configuration)
echo - set_env.bat (manual environment setup)
echo.
pause