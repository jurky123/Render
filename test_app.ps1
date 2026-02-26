# Test script to run PathTracer and capture logs

cd "e:\SomeDocs\Render"

# Remove old log file
if (Test-Path PathTracer_optix.log) {
    Remove-Item PathTracer_optix.log
}

# Start PathTracer in background
$proc = Start-Process -FilePath ".\build\Debug\PathTracer.exe" -NoNewWindow -PassThru

# Wait for app to initialize
Start-Sleep -Seconds 5

# Check if log file was created
if (Test-Path PathTracer_optix.log) {
    Write-Host "=== Log file created, first 150 lines ==="
    Get-Content PathTracer_optix.log | Select-Object -First 150
} else {
    Write-Host "Log file not created yet"
}

# Kill the process
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
