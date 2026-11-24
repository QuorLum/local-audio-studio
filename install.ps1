<#
.SYNOPSIS
  local-audio-studio installer for Windows (PowerShell 5.1 or 7+).

.DESCRIPTION
  Creates a virtual environment, installs the right PyTorch build (CUDA if an NVIDIA GPU is found,
  otherwise CPU), installs the requirements and offers to download the models.
  Needs Python 3.10-3.12 and git (the official Stable Audio 3 code is installed from GitHub).

.EXAMPLE
  .\install.ps1                       # interactive: installs everything, then asks about the models
  .\install.ps1 -Models all           # download every model (about 6 GB) without asking
  .\install.ps1 -Models tts           # speech only (about 0.4 GB); also sfx, music, "sfx,music", none
  .\install.ps1 -Cpu                  # force the CPU-only PyTorch build
  .\install.ps1 -Cuda cu118           # choose the CUDA wheel set (cu118, cu126 default, cu128)
  .\install.ps1 -DryRun               # show what would run, change nothing

  If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\install.ps1
#>
[CmdletBinding()]
param(
    [switch]$Cpu,
    [ValidateSet("cu118", "cu126", "cu128")]
    [string]$Cuda = "cu126",
    [string]$Models = "ask",
    [string]$Venv = ".venv",
    [switch]$NoVenv,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$TorchVersion = "2.7.1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

function Say([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Run([string]$Exe, [string[]]$Arguments) {
    Write-Host ("+ " + $Exe + " " + ($Arguments -join " "))
    if (-not $DryRun) {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Exe" }
    }
}

# --- 1. Python 3.10 - 3.12 --------------------------------------------------------------------
function Find-Python {
    $candidates = @(
        @("py", "-3.12"), @("py", "-3.11"), @("py", "-3.10"), @("python3"), @("python")
    )
    foreach ($candidate in $candidates) {
        $exe = $candidate[0]
        $extra = @()
        if ($candidate.Count -gt 1) { $extra = @($candidate[1]) }
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        try {
            $version = & $exe @extra -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -match '^3\.(10|11|12)$') {
                return , (@($exe) + $extra)
            }
        } catch { }
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    Write-Host "error: Python 3.10, 3.11 or 3.12 is required (3.13+ is not supported by Kokoro's text front-end yet)." -ForegroundColor Red
    Write-Host "       Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') or: winget install Python.Python.3.12"
    exit 1
}
$pyExe = $python[0]
$pyArgs = @()
if ($python.Count -gt 1) { $pyArgs = $python[1..($python.Count - 1)] }
Say ("Using " + (& $pyExe @pyArgs --version))

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "error: git is required (the official Stable Audio 3 package is installed from GitHub)." -ForegroundColor Red
    Write-Host "       winget install Git.Git"
    exit 1
}

# --- 2. virtual environment -----------------------------------------------------------------------
if ($NoVenv) {
    $venvPython = $pyExe
    $venvArgs = $pyArgs
} else {
    Say "Creating virtual environment in $Venv"
    Run $pyExe ($pyArgs + @("-m", "venv", $Venv))
    $venvPython = Join-Path $Venv "Scripts\python.exe"
    $venvArgs = @()
}
function Py([string[]]$Arguments) { Run $venvPython ($venvArgs + $Arguments) }

Py @("-m", "pip", "install", "--upgrade", "pip")

# --- 3. PyTorch: pick the build for this machine ----------------------------------------------------
$torchIndex = "https://download.pytorch.org/whl/cpu"
$hasNvidia = $false
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    & nvidia-smi -L *> $null
    if ($LASTEXITCODE -eq 0) { $hasNvidia = $true }
}
if ($Cpu) {
    Say "PyTorch: CPU build (forced)"
} elseif ($hasNvidia) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | Select-Object -First 1)
    Say "NVIDIA GPU detected: $gpu"
    Say "PyTorch: CUDA build ($Cuda)"
    $torchIndex = "https://download.pytorch.org/whl/$Cuda"
} else {
    Say "No NVIDIA GPU found: installing the CPU-only PyTorch build"
}
Py @("-m", "pip", "install", "torch==$TorchVersion", "torchaudio==$TorchVersion", "--index-url", $torchIndex)

# --- 4. everything else ---------------------------------------------------------------------------
Say "Installing requirements"
Py @("-m", "pip", "install", "-r", "requirements.txt")
Py @("-m", "pip", "install", "-e", ".", "--no-deps")   # adds the local-audio and local-audio-setup commands

# --- 5. models --------------------------------------------------------------------------------------
switch ($Models.ToLower()) {
    "none" { Say "Skipping model download (run: python scripts\setup_models.py)" }
    "all"  { Say "Downloading all models (about 6 GB)"; Py @("scripts\setup_models.py", "--only", "all", "--yes") }
    "ask"  {
        if ([Environment]::UserInteractive -and -not $DryRun) {
            $answer = Read-Host "Download models now? [Y/n]"
            if ($answer -match '^[nN]') {
                Say "Skipped. Run later: python scripts\setup_models.py"
            } else {
                Py @("scripts\setup_models.py")
            }
        } else {
            Say "Skipping model download (run: python scripts\setup_models.py)"
        }
    }
    default { Say "Downloading: $Models"; Py @("scripts\setup_models.py", "--only", $Models, "--yes") }
}

# --- 6. health check -----------------------------------------------------------------------------------
Say "Checking the installation"
try { Py @("scripts\generate_audio.py", "--doctor") } catch { Write-Host "doctor reported problems (see above)" -ForegroundColor Yellow }

Write-Host ""
Write-Host "Done."
if (-not $NoVenv) { Write-Host "Activate the environment with:  $Venv\Scripts\Activate.ps1" }
Write-Host ""
Write-Host '  python scripts\generate_audio.py --mode tts   --text "Hello from my laptop." -o hello.wav'
Write-Host '  python scripts\generate_audio.py --mode sfx   --prompt "rain on a tin roof" --duration 8 -o rain.wav'
Write-Host '  python scripts\generate_audio.py --mode music --prompt "calm acoustic guitar" --duration 30 -o bed.wav'
