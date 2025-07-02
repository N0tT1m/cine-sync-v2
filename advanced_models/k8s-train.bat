@echo off
REM CineSync v2 Advanced Models Kubernetes Training Script
REM Windows Batch Version
setlocal enabledelayedexpansion

set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "NC=[0m"

echo %GREEN%🎬 CineSync v2 Kubernetes Training Manager%NC%
echo ==================================================

REM Check if kubectl is available
kubectl version --client >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ kubectl not found. Please install kubectl first.%NC%
    echo Download from: https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/
    pause
    exit /b 1
)

REM Check if Docker is running (for building images)
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Docker not running. You'll need Docker to build images.%NC%
)

if "%1"=="" goto :show_usage
if "%1"=="help" goto :show_usage
if "%1"=="--help" goto :show_usage
if "%1"=="-h" goto :show_usage

if "%1"=="setup" goto :setup_k8s
if "%1"=="build" goto :build_image
if "%1"=="deploy" goto :deploy_resources
if "%1"=="train" goto :run_training
if "%1"=="jupyter" goto :start_jupyter
if "%1"=="tensorboard" goto :start_tensorboard
if "%1"=="status" goto :check_status
if "%1"=="logs" goto :show_logs
if "%1"=="stop" goto :stop_training
if "%1"=="clean" goto :cleanup
if "%1"=="shell" goto :open_shell

goto :unknown_command

:show_usage
echo Usage: %0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   setup       Setup Kubernetes resources (namespace, storage, etc.)
echo   build       Build Docker image for training
echo   deploy      Deploy all Kubernetes resources
echo   train       Run training job with specified model
echo   jupyter     Start Jupyter notebook service
echo   tensorboard Start TensorBoard monitoring
echo   status      Check status of all resources
echo   logs        Show training logs
echo   stop        Stop training job
echo   clean       Clean up all resources
echo   shell       Open shell in training pod
echo.
echo Training Options (use with 'train'):
echo   --model MODEL_NAME    Model to train (default: enhanced_two_tower)
echo   --epochs N           Number of epochs (default: 10)
echo   --batch-size N       Batch size (default: auto)
echo   --lr FLOAT           Learning rate (default: 0.003)
echo.
echo Examples:
echo   %0 setup
echo   %0 build
echo   %0 deploy
echo   %0 train --model sentence_bert --epochs 20
echo   %0 tensorboard
echo   %0 status
echo   %0 logs training-job
echo   %0 clean
goto :end

:setup_k8s
echo %YELLOW%🔧 Setting up Kubernetes resources...%NC%

REM Create namespace
echo Creating namespace...
kubectl create namespace cinesync --dry-run=client -o yaml | kubectl apply -f -

REM Apply storage resources
echo Setting up storage...
kubectl apply -f k8s/storage.yaml
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to setup storage%NC%
    goto :end
)

REM Apply ConfigMaps
echo Setting up configuration...
kubectl apply -f k8s/configmap.yaml
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to setup configuration%NC%
    goto :end
)

echo %GREEN%✅ Kubernetes setup complete%NC%
echo.
echo Next steps:
echo 1. Build Docker image: %0 build
echo 2. Deploy resources: %0 deploy
echo 3. Run training: %0 train --model your_model
goto :end

:build_image
echo %YELLOW%🔨 Building Docker image...%NC%
cd /d "%~dp0"

docker build -t cinesync-advanced-models:latest .
if %errorlevel% neq 0 (
    echo %RED%❌ Docker build failed%NC%
    goto :end
)

echo %GREEN%✅ Docker image built successfully%NC%
goto :end

:deploy_resources
echo %YELLOW%🚀 Deploying Kubernetes resources...%NC%

REM Apply all resources
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/services.yaml

echo %GREEN%✅ Resources deployed%NC%
goto :end

:run_training
echo %YELLOW%🎯 Starting training job...%NC%

REM Parse training arguments
set "MODEL_TYPE=enhanced_two_tower"
set "EPOCHS=10"
set "BATCH_SIZE=auto"
set "LEARNING_RATE=0.003"

:parse_train_args
if "%2"=="" goto :run_training_job
if "%2"=="--model" (
    set "MODEL_TYPE=%3"
    shift
    shift
    goto :parse_train_args
)
if "%2"=="--epochs" (
    set "EPOCHS=%3"
    shift
    shift
    goto :parse_train_args
)
if "%2"=="--batch-size" (
    set "BATCH_SIZE=%3"
    shift
    shift
    goto :parse_train_args
)
if "%2"=="--lr" (
    set "LEARNING_RATE=%3"
    shift
    shift
    goto :parse_train_args
)
shift
goto :parse_train_args

:run_training_job
echo Training Configuration:
echo   Model: %MODEL_TYPE%
echo   Epochs: %EPOCHS%
echo   Batch Size: %BATCH_SIZE%
echo   Learning Rate: %LEARNING_RATE%
echo.

REM Create a unique job name with timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "TIMESTAMP=%dt:~0,8%-%dt:~8,6%"
set "JOB_NAME=training-job-%TIMESTAMP%"

REM Create training job YAML
echo apiVersion: batch/v1 > temp-job.yaml
echo kind: Job >> temp-job.yaml
echo metadata: >> temp-job.yaml
echo   name: %JOB_NAME% >> temp-job.yaml
echo   namespace: cinesync >> temp-job.yaml
echo spec: >> temp-job.yaml
echo   template: >> temp-job.yaml
echo     spec: >> temp-job.yaml
echo       containers: >> temp-job.yaml
echo       - name: training >> temp-job.yaml
echo         image: cinesync-advanced-models:latest >> temp-job.yaml
echo         imagePullPolicy: Never >> temp-job.yaml
echo         command: ["python", "train_advanced_models.py"] >> temp-job.yaml
echo         args: ["--model-type", "%MODEL_TYPE%", "--epochs", "%EPOCHS%", "--learning-rate", "%LEARNING_RATE%"] >> temp-job.yaml
echo         resources: >> temp-job.yaml
echo           requests: >> temp-job.yaml
echo             nvidia.com/gpu: 1 >> temp-job.yaml
echo           limits: >> temp-job.yaml
echo             nvidia.com/gpu: 1 >> temp-job.yaml
echo         env: >> temp-job.yaml
echo         - name: CUDA_VISIBLE_DEVICES >> temp-job.yaml
echo           value: "0" >> temp-job.yaml
echo         volumeMounts: >> temp-job.yaml
echo         - name: app-code >> temp-job.yaml
echo           mountPath: /app >> temp-job.yaml
echo         - name: checkpoints >> temp-job.yaml
echo           mountPath: /app/checkpoints >> temp-job.yaml
echo         - name: logs >> temp-job.yaml
echo           mountPath: /app/logs >> temp-job.yaml
echo       volumes: >> temp-job.yaml
echo       - name: app-code >> temp-job.yaml
echo         hostPath: >> temp-job.yaml
echo           path: E:\workspace\ai-apps\cine-sync-v2\advanced_models >> temp-job.yaml
echo       - name: checkpoints >> temp-job.yaml
echo         persistentVolumeClaim: >> temp-job.yaml
echo           claimName: cinesync-checkpoints-pvc >> temp-job.yaml
echo       - name: logs >> temp-job.yaml
echo         persistentVolumeClaim: >> temp-job.yaml
echo           claimName: cinesync-logs-pvc >> temp-job.yaml
echo       restartPolicy: Never >> temp-job.yaml

kubectl apply -f temp-job.yaml
del temp-job.yaml

if %errorlevel% equ 0 (
    echo %GREEN%✅ Training job started: %JOB_NAME%%NC%
    echo.
    echo Monitor with: %0 logs %JOB_NAME%
    echo Check status: %0 status
) else (
    echo %RED%❌ Failed to start training job%NC%
)
goto :end

:start_jupyter
echo %YELLOW%📓 Starting Jupyter notebook...%NC%
kubectl apply -f k8s/services.yaml
kubectl get service jupyter-service -n cinesync
if %errorlevel% equ 0 (
    echo %GREEN%✅ Jupyter started%NC%
    echo Access at: http://localhost:30888
) else (
    echo %RED%❌ Failed to start Jupyter%NC%
)
goto :end

:start_tensorboard
echo %YELLOW%📊 Starting TensorBoard...%NC%
kubectl apply -f k8s/services.yaml
kubectl get service tensorboard-service -n cinesync
if %errorlevel% equ 0 (
    echo %GREEN%✅ TensorBoard started%NC%
    echo Access at: http://localhost:30006
) else (
    echo %RED%❌ Failed to start TensorBoard%NC%
)
goto :end

:check_status
echo %YELLOW%📋 Checking status...%NC%
echo.
echo === Namespace ===
kubectl get namespace cinesync

echo.
echo === Pods ===
kubectl get pods -n cinesync

echo.
echo === Jobs ===
kubectl get jobs -n cinesync

echo.
echo === Services ===
kubectl get services -n cinesync

echo.
echo === PVCs ===
kubectl get pvc -n cinesync
goto :end

:show_logs
if "%2"=="" (
    echo Usage: %0 logs [JOB_NAME^|POD_NAME]
    echo.
    echo Available jobs:
    kubectl get jobs -n cinesync
    goto :end
)

echo %YELLOW%📝 Showing logs for %2...%NC%
kubectl logs -n cinesync -f job/%2 2>nul
if %errorlevel% neq 0 (
    REM Try as pod name
    kubectl logs -n cinesync -f %2
)
goto :end

:stop_training
echo %YELLOW%🛑 Stopping training jobs...%NC%
kubectl delete jobs -n cinesync --all
echo %GREEN%✅ Training jobs stopped%NC%
goto :end

:cleanup
echo %YELLOW%🧹 Cleaning up resources...%NC%
echo This will delete ALL CineSync training resources.
set /p "confirm=Are you sure? (y/N): "
if /i not "%confirm%"=="y" goto :end

kubectl delete namespace cinesync
echo %GREEN%✅ Cleanup complete%NC%
goto :end

:open_shell
echo %YELLOW%🐚 Opening shell in training pod...%NC%
kubectl run -it --rm debug --image=cinesync-advanced-models:latest --restart=Never -n cinesync -- bash
goto :end

:unknown_command
echo %RED%❌ Unknown command: %1%NC%
echo Use '%0 help' for usage information.
goto :end

:end
echo.
pause