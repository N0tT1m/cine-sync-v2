# Movie Recommendation Server Setup Script for Windows
# PowerShell script to set up and run the Rust-based recommendation server

param(
    [string]$Command = "run",
    [string]$ModelsPath = "models",
    [int]$Port = 3000,
    [string]$Host = "127.0.0.1",
    [switch]$CpuOnly,
    [switch]$Help
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Cyan"

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

function Show-Usage {
    Write-Host @"
Movie Recommendation Server Setup Script for Windows

Usage: .\setup.ps1 [COMMAND] [OPTIONS]

Commands:
  check      Check prerequisites
  build      Build the server
  export     Export model artifacts from pickle to JSON
  run        Run the server (default)
  docker     Build and run with Docker
  test       Test the running server
  all        Run complete setup (check, build, export, run)

Options:
  -ModelsPath PATH    Path to models directory (default: models)
  -Port PORT          Server port (default: 3000)
  -Host HOST          Server host (default: 127.0.0.1)
  -CpuOnly           Force CPU usage
  -Help              Show this help

Examples:
  .\setup.ps1 all                           # Complete setup and run
  .\setup.ps1 run -Port 8080               # Run on port 8080
  .\setup.ps1 docker -ModelsPath .\models  # Run with Docker
"@
}

function Test-Command {
    param([string]$CommandName)
    return (Get-Command $CommandName -ErrorAction SilentlyContinue) -ne $null
}

function Check-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    # Check Rust/Cargo
    if (Test-Command "cargo") {
        Write-Success "Rust is installed"
        $rustVersion = cargo --version
        Write-Host "  $rustVersion" -ForegroundColor Gray
    } else {
        Write-Error "Rust is not installed. Please install Rust from https://rustup.rs/"
        Write-Host "To install Rust on Windows:" -ForegroundColor Yellow
        Write-Host "1. Download and run rustup-init.exe from https://rustup.rs/" -ForegroundColor Yellow
        Write-Host "2. Restart PowerShell after installation" -ForegroundColor Yellow
        return $false
    }
    
    # Check Python (optional, for model export)
    if (Test-Command "python") {
        Write-Success "Python is available"
        $pythonVersion = python --version
        Write-Host "  $pythonVersion" -ForegroundColor Gray
    } elseif (Test-Command "py") {
        Write-Success "Python is available (via py launcher)"
        $pythonVersion = py --version
        Write-Host "  $pythonVersion" -ForegroundColor Gray
    } else {
        Write-Warning "Python not found. You'll need Python to export model artifacts."
        Write-Host "Install from: https://www.python.org/downloads/" -ForegroundColor Yellow
    }
    
    # Check NVIDIA GPU (optional)
    try {
        $nvidiaOutput = nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>$null
        if ($nvidiaOutput) {
            Write-Success "NVIDIA GPU detected: $($nvidiaOutput[0])"
        }
    } catch {
        Write-Warning "No NVIDIA GPU detected or nvidia-smi not available. Server will run on CPU."
    }
    
    # Check Visual Studio Build Tools (required for some Rust crates on Windows)
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $vsInstalls = & $vsWhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64
        if ($vsInstalls) {
            Write-Success "Visual Studio Build Tools found"
        } else {
            Write-Warning "Visual Studio Build Tools not found. May be needed for compilation."
            Write-Host "Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Yellow
        }
    }
    
    # Check Docker (optional)
    if (Test-Command "docker") {
        Write-Success "Docker is available"
        try {
            $dockerVersion = docker --version
            Write-Host "  $dockerVersion" -ForegroundColor Gray
        } catch {
            Write-Warning "Docker command found but may not be running"
        }
    } else {
        Write-Warning "Docker not found. Docker deployment will not be available."
        Write-Host "Install from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    }
    
    return $true
}

function Build-Server {
    Write-Status "Building the recommendation server..."
    
    # Set environment variables for PyTorch on Windows
    $env:TORCH_CUDA_VERSION = "cu118"
    $env:LIBTORCH_USE_PYTORCH = "1"
    
    # Windows-specific: ensure we're using the right linker
    if ($env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER -eq $null) {
        Write-Status "Setting up Windows build environment..."
    }
    
    try {
        $buildResult = cargo build --release
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Server built successfully"
            return $true
        } else {
            Write-Error "Failed to build server"
            Write-Host "Try running: cargo clean && cargo build --release" -ForegroundColor Yellow
            return $false
        }
    } catch {
        Write-Error "Build failed: $($_.Exception.Message)"
        return $false
    }
}

function Export-ModelArtifacts {
    param([string]$ModelsPath)
    
    if (-not (Test-Path $ModelsPath)) {
        Write-Error "Models directory not found: $ModelsPath"
        Write-Status "Please train your model first using: python run_training_pytorch.py"
        return $false
    }
    
    Write-Status "Exporting model artifacts for Rust compatibility..."
    
    if (Test-Path "export_metadata_for_rust.py") {
        try {
            if (Test-Command "python") {
                python export_metadata_for_rust.py --models-path $ModelsPath
            } elseif (Test-Command "py") {
                py export_metadata_for_rust.py --models-path $ModelsPath
            } else {
                Write-Error "Python not found"
                return $false
            }
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Model artifacts exported"
                return $true
            } else {
                Write-Error "Export failed"
                return $false
            }
        } catch {
            Write-Error "Export failed: $($_.Exception.Message)"
            return $false
        }
    } else {
        Write-Warning "Export script not found. Make sure export_metadata_for_rust.py is in the current directory."
        return $false
    }
}

function Start-Server {
    param(
        [string]$ModelsPath,
        [int]$Port,
        [string]$Host,
        [bool]$CpuOnly
    )
    
    Write-Status "Starting recommendation server..."
    Write-Status "Models path: $ModelsPath"
    Write-Status "Server URL: http://${Host}:${Port}"
    
    # Check if executable exists
    $exePath = ".\target\release\movie-recommendation-server.exe"
    if (-not (Test-Path $exePath)) {
        Write-Error "Server executable not found at $exePath"
        Write-Status "Please build the server first with: .\setup.ps1 build"
        return $false
    }
    
    # Build command arguments
    $args = @(
        "--models-path", $ModelsPath,
        "--port", $Port,
        "--host", $Host
    )
    
    if ($CpuOnly) {
        $args += "--cpu-only"
        Write-Status "Running in CPU-only mode"
    }
    
    # Set logging level
    if ($env:RUST_LOG -eq $null) {
        $env:RUST_LOG = "info"
    }
    
    Write-Status "Running: $exePath $($args -join ' ')"
    
    try {
        & $exePath @args
    } catch {
        Write-Error "Failed to start server: $($_.Exception.Message)"
        return $false
    }
}

function Start-Docker {
    param([string]$ModelsPath)
    
    if (-not (Test-Command "docker")) {
        Write-Error "Docker is not installed"
        return $false
    }
    
    Write-Status "Building Docker image..."
    docker build -t movie-recommendation-server .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed"
        return $false
    }
    
    Write-Status "Running Docker container..."
    
    # Convert Windows path to Docker-compatible path
    $dockerModelsPath = (Resolve-Path $ModelsPath).Path.Replace('\', '/').Replace('C:', '/c')
    
    docker run -d `
        --name movie-recommendation-server `
        --gpus all `
        -p 3000:3000 `
        -v "${dockerModelsPath}:/app/models:ro" `
        -e RUST_LOG=info `
        movie-recommendation-server
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker container started"
        Write-Status "Check logs with: docker logs movie-recommendation-server"
        Write-Status "Server available at: http://localhost:3000"
        return $true
    } else {
        Write-Error "Failed to start Docker container"
        return $false
    }
}

function Test-Server {
    param([int]$Port, [string]$Host)
    
    Write-Status "Testing server at http://${Host}:${Port}..."
    
    # Wait for server to start
    Start-Sleep -Seconds 3
    
    try {
        $response = Invoke-RestMethod -Uri "http://${Host}:${Port}/health" -Method Get -TimeoutSec 10
        Write-Success "Health check passed"
        
        Write-Status "Server information:"
        Write-Host ($response | ConvertTo-Json -Depth 3) -ForegroundColor Gray
        
        return $true
    } catch {
        Write-Error "Health check failed: $($_.Exception.Message)"
        Write-Status "Make sure the server is running and accessible"
        return $false
    }
}

# Main execution
if ($Help) {
    Show-Usage
    exit 0
}

Write-Status "Movie Recommendation Server Setup (Windows)"
Write-Status "Command: $Command"

switch ($Command.ToLower()) {
    "check" {
        Check-Prerequisites
    }
    "build" {
        if (Check-Prerequisites) {
            Build-Server
        }
    }
    "export" {
        Export-ModelArtifacts -ModelsPath $ModelsPath
    }
    "run" {
        if (Check-Prerequisites) {
            if (-not (Test-Path ".\target\release\movie-recommendation-server.exe")) {
                Write-Status "Server not built yet. Building..."
                if (-not (Build-Server)) {
                    exit 1
                }
            }
            
            Export-ModelArtifacts -ModelsPath $ModelsPath | Out-Null
            Start-Server -ModelsPath $ModelsPath -Port $Port -Host $Host -CpuOnly $CpuOnly
        }
    }
    "docker" {
        if (Check-Prerequisites) {
            Export-ModelArtifacts -ModelsPath $ModelsPath | Out-Null
            Start-Docker -ModelsPath $ModelsPath
        }
    }
    "test" {
        Test-Server -Port $Port -Host $Host
    }
    "all" {
        if (Check-Prerequisites) {
            if (Build-Server) {
                if (Export-ModelArtifacts -ModelsPath $ModelsPath) {
                    Write-Status "Setup complete! Starting server..."
                    Start-Server -ModelsPath $ModelsPath -Port $Port -Host $Host -CpuOnly $CpuOnly
                }
            }
        }
    }
    default {
        Write-Error "Unknown command: $Command"
        Show-Usage
        exit 1
    }
}