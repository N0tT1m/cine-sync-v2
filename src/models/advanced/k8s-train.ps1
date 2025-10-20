#!/usr/bin/env pwsh
# CineSync v2 Advanced Models Kubernetes Training Script
# PowerShell Version

param(
    [Parameter(Position=0)]
    [string]$Command = "",
    
    [Parameter(Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$Arguments = @()
)

# Colors for output
$Green = "`e[92m"
$Red = "`e[91m"
$Yellow = "`e[93m"
$NC = "`e[0m"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = $NC)
    Write-Host "${Color}${Message}${NC}"
}

function Show-Header {
    Write-ColorOutput "CineSync v2 Kubernetes Training Manager" $Green
    Write-Host "=============================================="
}

function Show-Usage {
    Write-Host "Usage: k8s-train.ps1 [COMMAND] [OPTIONS]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  setup       Setup Kubernetes resources"
    Write-Host "  build       Build Docker image for training"
    Write-Host "  deploy      Deploy all Kubernetes resources"
    Write-Host "  train       Run training job with specified model"
    Write-Host "  jupyter     Start Jupyter notebook service"
    Write-Host "  tensorboard Start TensorBoard monitoring"
    Write-Host "  status      Check status of all resources"
    Write-Host "  logs        Show training logs"
    Write-Host "  stop        Stop training job"
    Write-Host "  clean       Clean up all resources"
    Write-Host "  shell       Open shell in training pod"
    Write-Host ""
    Write-Host "Training Options (use with train):"
    Write-Host "  -Model MODEL_NAME     Model to train (default: enhanced_two_tower)"
    Write-Host "  -Epochs N            Number of epochs (default: 10)"
    Write-Host "  -BatchSize N         Batch size (default: auto)"
    Write-Host "  -LearningRate FLOAT  Learning rate (default: 0.003)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  k8s-train.ps1 setup"
    Write-Host "  k8s-train.ps1 build"
    Write-Host "  k8s-train.ps1 deploy"
    Write-Host "  k8s-train.ps1 train -Model sentence_bert -Epochs 20"
    Write-Host "  k8s-train.ps1 tensorboard"
    Write-Host "  k8s-train.ps1 status"
    Write-Host "  k8s-train.ps1 logs training-job-12345"
    Write-Host "  k8s-train.ps1 clean"
}

function Test-Prerequisites {
    # Check kubectl
    try {
        kubectl version --client | Out-Null
        Write-ColorOutput "kubectl is available" $Green
    } catch {
        Write-ColorOutput "kubectl not found. Please install kubectl first." $Red
        Write-Host "Download from: https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/"
        exit 1
    }
    
    # Check Docker
    try {
        docker version | Out-Null
        Write-ColorOutput "Docker is available" $Green
    } catch {
        Write-ColorOutput "Docker not running. You will need Docker to build images." $Yellow
    }
    
    # Check if running in correct directory
    if (-not (Test-Path "k8s")) {
        Write-ColorOutput "k8s directory not found. Please run from advanced_models directory." $Red
        exit 1
    }
}

function Setup-K8s {
    Write-ColorOutput "Setting up Kubernetes resources..." $Yellow
    
    # Create namespace
    Write-Host "Creating namespace..."
    kubectl create namespace cinesync --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply storage resources
    Write-Host "Setting up storage..."
    kubectl apply -f k8s/storage.yaml
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Failed to setup storage" $Red
        return
    }
    
    # Apply ConfigMaps
    Write-Host "Setting up configuration..."
    kubectl apply -f k8s/configmap.yaml
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Failed to setup configuration" $Red
        return
    }
    
    Write-ColorOutput "Kubernetes setup complete" $Green
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. Build Docker image: k8s-train.ps1 build"
    Write-Host "2. Deploy resources: k8s-train.ps1 deploy"
    Write-Host "3. Run training: k8s-train.ps1 train -Model your_model"
}

function Build-Image {
    Write-ColorOutput "Building Docker image..." $Yellow
    
    docker build -t cinesync-advanced-models:latest .
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "Docker image built successfully" $Green
    } else {
        Write-ColorOutput "Docker build failed" $Red
    }
}

function Deploy-Resources {
    Write-ColorOutput "Deploying Kubernetes resources..." $Yellow
    
    kubectl apply -f k8s/storage.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/services.yaml
    
    Write-ColorOutput "Resources deployed" $Green
}

function Start-Training {
    param(
        [string]$Model = "enhanced_two_tower",
        [int]$Epochs = 10,
        [string]$BatchSize = "auto",
        [double]$LearningRate = 0.003
    )
    
    # Parse PowerShell-style arguments
    for ($i = 0; $i -lt $Arguments.Count; $i++) {
        switch ($Arguments[$i]) {
            "-Model" { $Model = $Arguments[$i+1]; $i++ }
            "-Epochs" { $Epochs = [int]$Arguments[$i+1]; $i++ }
            "-BatchSize" { $BatchSize = $Arguments[$i+1]; $i++ }
            "-LearningRate" { $LearningRate = [double]$Arguments[$i+1]; $i++ }
        }
    }
    
    Write-ColorOutput "Starting training job..." $Yellow
    Write-Host "Training Configuration:"
    Write-Host "  Model: $Model"
    Write-Host "  Epochs: $Epochs"
    Write-Host "  Batch Size: $BatchSize"
    Write-Host "  Learning Rate: $LearningRate"
    Write-Host ""
    
    # Create unique job name with timestamp
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $jobName = "training-job-$timestamp"
    
    # Create training job YAML
    $jobYaml = @"
apiVersion: batch/v1
kind: Job
metadata:
  name: $jobName
  namespace: cinesync
spec:
  template:
    spec:
      containers:
      - name: training
        image: cinesync-advanced-models:latest
        imagePullPolicy: Never
        command: ["python", "train_advanced_models.py"]
        args: ["--model-type", "$Model", "--epochs", "$Epochs", "--learning-rate", "$LearningRate"]
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "12"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: PYTHONPATH
          value: "/app:/app/.."
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: checkpoints
          mountPath: /app/checkpoints
        - name: logs
          mountPath: /app/logs
        - name: outputs
          mountPath: /app/outputs
        - name: hf-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: app-code
        hostPath:
          path: E:\workspace\ai-apps\cine-sync-v2\advanced_models
      - name: checkpoints
        persistentVolumeClaim:
          claimName: cinesync-checkpoints-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: cinesync-logs-pvc
      - name: outputs
        persistentVolumeClaim:
          claimName: cinesync-outputs-pvc
      - name: hf-cache
        persistentVolumeClaim:
          claimName: cinesync-hf-cache-pvc
      nodeSelector:
        accelerator: nvidia-rtx-4090
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      restartPolicy: Never
"@
    
    $jobYaml | kubectl apply -f -
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "Training job started: $jobName" $Green
        Write-Host ""
        Write-Host "Monitor with: k8s-train.ps1 logs $jobName"
        Write-Host "Check status: k8s-train.ps1 status"
    } else {
        Write-ColorOutput "Failed to start training job" $Red
    }
}

function Start-Jupyter {
    Write-ColorOutput "Starting Jupyter notebook..." $Yellow
    kubectl apply -f k8s/services.yaml
    $service = kubectl get service jupyter-service -n cinesync 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "Jupyter started" $Green
        Write-Host "Access at: http://localhost:30888"
    } else {
        Write-ColorOutput "Failed to start Jupyter" $Red
    }
}

function Start-TensorBoard {
    Write-ColorOutput "Starting TensorBoard..." $Yellow
    kubectl apply -f k8s/services.yaml
    $service = kubectl get service tensorboard-service -n cinesync 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "TensorBoard started" $Green
        Write-Host "Access at: http://localhost:30006"
    } else {
        Write-ColorOutput "Failed to start TensorBoard" $Red
    }
}

function Show-Status {
    Write-ColorOutput "Checking status..." $Yellow
    Write-Host ""
    
    Write-Host "=== Namespace ==="
    kubectl get namespace cinesync
    
    Write-Host ""
    Write-Host "=== Pods ==="
    kubectl get pods -n cinesync
    
    Write-Host ""
    Write-Host "=== Jobs ==="
    kubectl get jobs -n cinesync
    
    Write-Host ""
    Write-Host "=== Services ==="
    kubectl get services -n cinesync
    
    Write-Host ""
    Write-Host "=== PVCs ==="
    kubectl get pvc -n cinesync
}

function Show-Logs {
    param([string]$ResourceName)
    
    if (-not $ResourceName) {
        Write-Host "Usage: k8s-train.ps1 logs [JOB_NAME|POD_NAME]"
        Write-Host ""
        Write-Host "Available jobs:"
        kubectl get jobs -n cinesync
        return
    }
    
    Write-ColorOutput "Showing logs for $ResourceName..." $Yellow
    
    # Try as job first
    kubectl logs -n cinesync -f "job/$ResourceName" 2>$null
    if ($LASTEXITCODE -ne 0) {
        # Try as pod name
        kubectl logs -n cinesync -f $ResourceName
    }
}

function Stop-Training {
    Write-ColorOutput "Stopping training jobs..." $Yellow
    kubectl delete jobs -n cinesync --all
    Write-ColorOutput "Training jobs stopped" $Green
}

function Remove-All {
    Write-ColorOutput "Cleaning up resources..." $Yellow
    $confirm = Read-Host "This will delete ALL CineSync training resources. Are you sure? (y/N)"
    
    if ($confirm -eq 'y' -or $confirm -eq 'Y') {
        kubectl delete namespace cinesync
        Write-ColorOutput "Cleanup complete" $Green
    } else {
        Write-Host "Cleanup cancelled."
    }
}

function Open-Shell {
    Write-ColorOutput "Opening shell in training pod..." $Yellow
    kubectl run -it --rm debug --image=cinesync-advanced-models:latest --restart=Never -n cinesync -- bash
}

# Main script logic
Show-Header
Test-Prerequisites

switch ($Command.ToLower()) {
    "setup" { Setup-K8s }
    "build" { Build-Image }
    "deploy" { Deploy-Resources }
    "train" { Start-Training }
    "jupyter" { Start-Jupyter }
    "tensorboard" { Start-TensorBoard }
    "status" { Show-Status }
    "logs" { Show-Logs $Arguments[0] }
    "stop" { Stop-Training }
    "clean" { Remove-All }
    "shell" { Open-Shell }
    "help" { Show-Usage }
    "--help" { Show-Usage }
    "-h" { Show-Usage }
    "" { Show-Usage }
    default {
        Write-ColorOutput "Unknown command: $Command" $Red
        Write-Host "Use k8s-train.ps1 help for usage information."
    }
}