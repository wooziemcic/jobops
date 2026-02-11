param(
  [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

if (!(Test-Path ".\.venv\Scripts\Activate.ps1")) {
  throw "Virtualenv not found. Create it with: python -m venv .venv"
}

.\.venv\Scripts\Activate.ps1 | Out-Null

streamlit run .\app\Home.py --server.port $Port
