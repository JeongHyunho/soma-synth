# Build BOTH the Korean and English viewer manuals to docx, in sync.
#   pwsh docs\manual\build_manual.ps1
# Source of truth = the .qmd files. Edit those (not the docx), then re-run this to rebuild both.
# Needs Quarto ($env:QUARTO, else `quarto` on PATH, else the per-user install) and a `python` with
# python-docx for the post-render patch. Close the manuals in Word first: an open docx cannot be
# overwritten, and this script does not close Word for you.
$root = $PSScriptRoot
Set-Location $root
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { chcp 65001 > $null } catch {}

if ($env:QUARTO) { $q = $env:QUARTO }
elseif (Get-Command quarto -ErrorAction SilentlyContinue) { $q = "quarto" }
else { $q = "$env:LOCALAPPDATA\Programs\Quarto\bin\quarto.cmd" }

function Render($qmd) {
  for ($i = 1; $i -le 5; $i++) {
    # a docx open in Word locks the output: back off and ask for it to be closed
    & $q render $qmd --to docx
    if ($LASTEXITCODE -eq 0) { return }
    $wait = 5 * $i
    Write-Warning "render failed ($qmd), attempt $i/5; retrying in ${wait}s (close the docx in Word)..."
    Start-Sleep -Seconds $wait
  }
  throw "render failed after 5 attempts: $qmd (close any manual docx open in Word, then retry)"
}

$pairs = @(
  @{ qmd = "viewer_manual.qmd";    docx = "viewer_manual.docx" },      # Korean
  @{ qmd = "viewer_manual_en.qmd"; docx = "viewer_manual_en.docx" }    # English
)
foreach ($p in $pairs) {
  Render $p.qmd
  python "$root\patch_manual_docx.py" (Join-Path $root $p.docx)
}

Write-Output "Built manuals:"
Get-ChildItem "$root\*.docx" | Select-Object Name, @{n = "KB"; e = { [int]($_.Length / 1KB) } } | Format-Table -AutoSize
