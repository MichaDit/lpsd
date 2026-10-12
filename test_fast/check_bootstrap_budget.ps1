# Run the actual watchdog function without launching the bootstrap or network.
# Supported by Windows PowerShell 5.1 and PowerShell 7.
$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'scripts/run_benchmark.ps1'
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($scriptPath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'The bootstrap PowerShell script did not parse.' }
$definition = $ast.Find({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-Budget'
}, $true)
if (-not $definition) { throw 'Missing bootstrap watchdog.' }
. ([ScriptBlock]::Create($definition.Extent.Text))

function Assert-Rejected([string]$Message) {
    $observed = $false
    try { Test-Budget } catch {
        $observed = $true
        if ($_.Exception.Message -notlike "*$Message*") { throw }
    }
    if (-not $observed) { throw ('Expected rejection: ' + $Message) }
}

$root = Join-Path ([IO.Path]::GetTempPath()) ('lpsd-watchdog-test-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root | Out-Null
try {
    [IO.File]::WriteAllBytes((Join-Path $root 'present.bin'), (New-Object byte[] 32))
    $deadline = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + 60
    $setupGiB = 64.0 / 1GB
    Test-Budget
    $setupGiB = 16.0 / 1GB
    Assert-Rejected 'disk budget'
    $setupGiB = 64.0 / 1GB

    # Return real files, then reproduce a nonterminating enumeration error
    # when an installer has just removed or renamed an enumerated child.
    function Get-ChildItem {
        [CmdletBinding()]
        param([string]$LiteralPath, [switch]$Recurse, [switch]$File, [switch]$Force)
        Microsoft.PowerShell.Management\Get-ChildItem -LiteralPath $LiteralPath -Recurse -File -Force
        $record = New-Object System.Management.Automation.ErrorRecord (
            $script:injectedError, 'ConcurrentEnumeration',
            [System.Management.Automation.ErrorCategory]::ReadError, $LiteralPath)
        $PSCmdlet.WriteError($record)
    }
    foreach ($kind in @([IO.DirectoryNotFoundException], [IO.FileNotFoundException],
                       [System.Management.Automation.ItemNotFoundException])) {
        $script:injectedError = [Activator]::CreateInstance($kind, @('Vanished installer child'))
        Test-Budget
        # A disappearance must not discard the files already measured.
        $setupGiB = 16.0 / 1GB
        Assert-Rejected 'disk budget'
        $setupGiB = 64.0 / 1GB
    }
    $script:injectedError = New-Object UnauthorizedAccessException 'Access denied test'
    Assert-Rejected 'Access denied test'
    $script:injectedError = New-Object IO.IOException 'Storage error test'
    Assert-Rejected 'Storage error test'
    Remove-Item Function:Get-ChildItem

    $deadline = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - 1
    Assert-Rejected 'deadline'
    $deadline += 61
    Remove-Item -LiteralPath $root -Recurse -Force
    Assert-Rejected 'directory is unavailable'
    Write-Host 'Bootstrap watchdog: disappearance, access, storage, disk, deadline and missing-root checks passed.'
} finally {
    Remove-Item Function:Get-ChildItem -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
}
