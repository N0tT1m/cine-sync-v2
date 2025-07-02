#!/usr/bin/env pwsh
# Train all advanced models sequentially

$models = @(
    @{Name="enhanced_two_tower"; Epochs=20},
    @{Name="sasrec"; Epochs=15},
    @{Name="enhanced_sasrec"; Epochs=15},
    @{Name="sentence_bert"; Epochs=20},
    @{Name="bert4rec"; Epochs=15},
    @{Name="enhanced_bert4rec"; Epochs=15},
    @{Name="lightgcn"; Epochs=15},
    @{Name="graphsage"; Epochs=15},
    @{Name="graph_transformer"; Epochs=10},
    @{Name="multvae"; Epochs=20},
    @{Name="enhanced_multvae"; Epochs=20},
    @{Name="t5_hybrid"; Epochs=10}
)

Write-Host "Starting training for all $($models.Count) models..."

foreach ($model in $models) {
    Write-Host ""
    Write-Host "=" * 50
    Write-Host "Training: $($model.Name) for $($model.Epochs) epochs"
    Write-Host "=" * 50
    
    # Start training
    .\k8s-train.ps1 train -Model $model.Name -Epochs $model.Epochs
    
    # Wait for job to complete
    Write-Host "Waiting for $($model.Name) to complete..."
    
    do {
        Start-Sleep 30
        $jobs = kubectl get jobs -n cinesync -o json | ConvertFrom-Json
        $runningJobs = $jobs.items | Where-Object { $_.status.conditions -eq $null -or $_.status.conditions[0].type -ne "Complete" }
        Write-Host "Jobs still running: $($runningJobs.Count)"
    } while ($runningJobs.Count -gt 0)
    
    Write-Host "$($model.Name) completed!"
}

Write-Host ""
Write-Host "All models training completed!"
Write-Host "Check results with: .\k8s-train.ps1 status"