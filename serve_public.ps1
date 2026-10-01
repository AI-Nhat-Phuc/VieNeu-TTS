# Starts the VieNeu-TTS OpenAI-compatible API (apps/openai_speech.py) on this machine,
# listening on all interfaces. Settings come from .env (gitignored).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
$env:PYTHONUTF8 = "1"

Get-Content .env | Where-Object { $_ -match '^\s*([A-Z_][A-Z0-9_]*)=(.*)$' } | ForEach-Object {
    Set-Item -Path "env:$($Matches[1])" -Value $Matches[2].Trim()
}

& .\.venv\Scripts\python.exe -m apps.openai_speech
