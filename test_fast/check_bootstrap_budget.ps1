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

    # Return real files before the installer removes or renames a child. The
    # Windows reparse-point probe raises a terminating Win32/provider exception,
    # while other disappearance paths produce nonterminating errors.
    function Get-ChildItem {
        [CmdletBinding()]
        param([string]$LiteralPath, [switch]$Recurse, [switch]$File, [switch]$Force)
        Microsoft.PowerShell.Management\Get-ChildItem -LiteralPath $LiteralPath -Recurse -File -Force
        $record = New-Object System.Management.Automation.ErrorRecord (
            $script:injectedError, 'ConcurrentEnumeration',
            [System.Management.Automation.ErrorCategory]::ReadError, $LiteralPath)
        if ($script:terminateEnumeration) { $PSCmdlet.ThrowTerminatingError($record) }
        else { $PSCmdlet.WriteError($record) }
    }
    $missingErrors = @(
        (New-Object IO.DirectoryNotFoundException 'Vanished installer child'),
        (New-Object IO.FileNotFoundException 'Vanished installer child'),
        (New-Object System.Management.Automation.ItemNotFoundException 'Vanished installer child'),
        (New-Object ComponentModel.Win32Exception 2),
        (New-Object ComponentModel.Win32Exception 3),
        (New-Object System.Management.Automation.ProviderInvocationException (
            'Provider failed', (New-Object ComponentModel.Win32Exception 3)))
    )
    foreach ($script:terminateEnumeration in @($false, $true)) {
        foreach ($script:injectedError in $missingErrors) {
            Test-Budget
            # A disappearance must not discard the files already measured.
            $setupGiB = 16.0 / 1GB
            Assert-Rejected 'disk budget'
            $setupGiB = 64.0 / 1GB
        }
        foreach ($script:injectedError in @(
            (New-Object UnauthorizedAccessException 'Access denied test'),
            (New-Object IO.IOException 'Storage error test'),
            (New-Object ComponentModel.Win32Exception (5, 'Native access denied test')),
            (New-Object ComponentModel.Win32Exception (23, 'Native storage error test')),
            (New-Object System.Management.Automation.ProviderInvocationException (
                'Wrapped access denied test', (New-Object ComponentModel.Win32Exception 5))),
            (New-Object IO.IOException 'The system cannot find the path specified.')
        )) {
            # The same text without a missing-path exception/code is not enough.
            Assert-Rejected $script:injectedError.Message
        }
    }
    Remove-Item Function:Get-ChildItem

    $deadline = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - 1
    Assert-Rejected 'deadline'
    $deadline += 61
    Remove-Item -LiteralPath $root -Recurse -Force
    Assert-Rejected 'directory is unavailable'
    Write-Host 'Bootstrap watchdog: terminating/nonterminating disappearance, access, storage, disk, deadline and missing-root checks passed.'
} finally {
    Remove-Item Function:Get-ChildItem -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
}
