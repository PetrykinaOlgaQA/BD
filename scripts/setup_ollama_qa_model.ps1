# Создаёт локальную модель qa-kotofakty для полировки баг-репорта (нужен запущенный Ollama).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$base = (ollama list 2>$null | Select-String "llava").ToString()
if (-not $base) {
    Write-Host "Сначала: ollama pull llava:latest"
    exit 1
}

ollama create qa-kotofakty -f scripts/Modelfile.qa-kotofakty
Write-Host "Готово. В config.json укажите: `"gemma_model`": `"qa-kotofakty`", ollama.model: `"qa-kotofakty`""
