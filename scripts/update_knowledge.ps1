# Pull the latest vendored knowledge index from the knowledge repo (Windows).
# One-way sync: the knowledge repo is the source of truth; this script only copies.
# Usage: scripts\update_knowledge.ps1
$ErrorActionPreference = "Stop"

# --- Point this at the knowledge repo checkout ---
# Set the KNOWLEDGE_REPO env var, or edit the path below.
$KnowledgeRepo = $env:KNOWLEDGE_REPO
if (-not $KnowledgeRepo) { $KnowledgeRepo = "C:\longitudinal_ecg" }

$SourceDir = Join-Path $KnowledgeRepo "knowledge"
$DestDir = Join-Path $PSScriptRoot "..\knowledge"

if (-not (Test-Path $SourceDir)) {
    Write-Error "Knowledge source not found: $SourceDir`nSet KNOWLEDGE_REPO or edit this script."
    exit 1
}

$Files = @("index.sqlite", "chunks.jsonl", "manifest.json")
foreach ($f in $Files) {
    $src = Join-Path $SourceDir $f
    if (-not (Test-Path $src)) {
        Write-Error "Missing '$f' in the knowledge repo ($src)."
        exit 1
    }
    Copy-Item $src (Join-Path $DestDir $f) -Force
    Write-Host "Copied $f"
}

Write-Host "Knowledge index refreshed from $KnowledgeRepo"
