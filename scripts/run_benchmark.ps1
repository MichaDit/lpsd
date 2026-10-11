param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{40}$')][string]$Revision,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$BenchmarkArgs
)
# Run from Windows PowerShell 5.1+ or PowerShell 7. No profile/OS installation.
$ErrorActionPreference = 'Stop'
$savedEnvironment = @{}
foreach ($key in @('LPSD_BENCHMARK_STARTED_EPOCH', 'LPSD_BENCHMARK_DEADLINE_EPOCH',
                  'UV_PYTHON_INSTALL_DIR', 'UV_CACHE_DIR', 'UV_TOOL_DIR', 'UV_NO_CONFIG',
                  'UV_NO_PROGRESS', 'UV_PYTHON_PREFERENCE', 'UV_PYTHON_ARCH')) {
    $savedEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
$savedTlsPolicy = [Net.ServicePointManager]::SecurityProtocol
try {
$Revision = $Revision.ToLowerInvariant()
$hours = 24.0
$setupGiB = 2.0
$ramGiB = 'auto'
for ($index = 0; $index -lt $BenchmarkArgs.Count; $index++) {
    if ($BenchmarkArgs[$index] -eq '--max-hours') {
        $index++; $hours = [double]::Parse($BenchmarkArgs[$index], [Globalization.CultureInfo]::InvariantCulture)
    } elseif ($BenchmarkArgs[$index] -match '^--max-hours=(.+)$') {
        $hours = [double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture)
    } elseif ($BenchmarkArgs[$index] -eq '--bootstrap-disk-gib') {
        $index++; $setupGiB = [double]::Parse($BenchmarkArgs[$index], [Globalization.CultureInfo]::InvariantCulture)
    } elseif ($BenchmarkArgs[$index] -match '^--bootstrap-disk-gib=(.+)$') {
        $setupGiB = [double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture)
    } elseif ($BenchmarkArgs[$index] -eq '--ram-gib') {
        $index++; $ramGiB = $BenchmarkArgs[$index]
    } elseif ($BenchmarkArgs[$index] -match '^--ram-gib=(.+)$') {
        $ramGiB = $Matches[1]
    }
}
if ([double]::IsNaN($hours) -or [double]::IsInfinity($hours) -or $hours -le 0 -or $hours -gt 24) {
    throw 'Use 0 < --max-hours <= 24.'
}
if ([double]::IsNaN($setupGiB) -or [double]::IsInfinity($setupGiB) -or $setupGiB -le 0) {
    throw 'Use a positive --bootstrap-disk-gib.'
}
$started = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$deadline = $started + [Math]::Floor($hours * 3600)
$env:LPSD_BENCHMARK_STARTED_EPOCH = [string]$started
$env:LPSD_BENCHMARK_DEADLINE_EPOCH = [string]$deadline
$parent = if ($env:LPSD_BENCHMARK_BASE_DIR) { $env:LPSD_BENCHMARK_BASE_DIR } else { (Get-Location).Path }
$root = Join-Path $parent ('lpsd-benchmark-' + [Guid]::NewGuid().ToString('N').Substring(0, 12))
New-Item -ItemType Directory -Path $root | Out-Null
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$memory = Get-CimInstance -ClassName Win32_OperatingSystem -Property TotalVisibleMemorySize, FreePhysicalMemory
$ramBytes = [Math]::Min([double]$memory.TotalVisibleMemorySize * 1024 * .9,
                       [double]$memory.FreePhysicalMemory * 1024 * .8)
if ($ramGiB -ne 'auto') {
    $requested = [double]::Parse($ramGiB, [Globalization.CultureInfo]::InvariantCulture)
    if ([double]::IsNaN($requested) -or [double]::IsInfinity($requested) -or $requested -le 0) {
        throw 'Use --ram-gib auto or a positive number.'
    }
    $ramBytes = [Math]::Min($ramBytes, $requested * 1GB)
}
if ($ramBytes -lt 128MB) { throw 'Private setup needs at least 128 MiB of admitted RAM.' }

function Test-Budget {
    if ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() -ge $deadline) { throw 'Setup/run deadline reached.' }
    $used = (Get-ChildItem -LiteralPath $root -Recurse -File | Measure-Object -Property Length -Sum).Sum
    if ($used -gt $setupGiB * 1GB) { throw 'Private setup disk budget exceeded.' }
    $drive = New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($root))
    if ($drive.AvailableFreeSpace -lt 32MB) { throw 'Private setup free-space reserve reached.' }
}
function Get-OfficialFile([string]$Url, [string]$Destination) {
    Test-Budget
    $response = $null
    $inputStream = $null
    $outputStream = $null
    try {
        $current = [Uri]$Url
        for ($redirects = 0; $redirects -le 5; $redirects++) {
            if ($current.Scheme -ne 'https' -or $current.Port -ne 443 -or $current.UserInfo -or
                $current.Host -notin @('github.com', 'raw.githubusercontent.com',
                                     'release-assets.githubusercontent.com', 'objects.githubusercontent.com')) {
                throw 'Download URL is outside the official HTTPS allowlist.'
            }
            $left = [Math]::Min(300, [Math]::Max(1, $deadline - [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()))
            $request = [Net.HttpWebRequest]::Create($current)
            $request.AllowAutoRedirect = $false
            $request.Timeout = [int]($left * 1000)
            $request.ReadWriteTimeout = [int]($left * 1000)
            $response = $request.GetResponse()
            if ([int]$response.StatusCode -notin @(301, 302, 303, 307, 308)) { break }
            $next = New-Object Uri ($current, $response.Headers['Location'])
            $response.Close(); $response = $null
            $current = $next
        }
        if (-not $response -or [int]$response.StatusCode -ne 200) { throw 'Official download did not complete.' }
        if ($response.ContentLength -gt 160MB) { throw 'Download exceeds its bounded file size.' }
        $inputStream = $response.GetResponseStream()
        $outputStream = New-Object IO.FileStream ($Destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
        $buffer = New-Object byte[] 65536
        $received = 0L
        while (($count = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $received += $count
            if ($received -gt 160MB) { throw 'Download exceeds its bounded file size.' }
            Test-Budget
            $outputStream.Write($buffer, 0, $count)
        }
    } catch {
        throw 'Official HTTPS download failed or exhausted its budget; no response or credentials logged.'
    } finally {
        if ($outputStream) { $outputStream.Dispose() }
        if ($inputStream) { $inputStream.Dispose() }
        if ($response) { $response.Close() }
    }
    Test-Budget
}
function Convert-NativeArgument([string]$Value) {
    # ProcessStartInfo.Arguments on PowerShell 5.1 needs Windows CRT quoting.
    if ($Value -notmatch '[\s"]' -and $Value.Length -gt 0) { return $Value }
    return '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}
function Start-Bounded([string]$Executable, [string[]]$NativeArgs) {
    Test-Budget
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Executable
    $info.Arguments = (($NativeArgs | ForEach-Object { Convert-NativeArgument $_ }) -join ' ')
    $info.UseShellExecute = $false
    $process = [Diagnostics.Process]::Start($info)
    try {
        while (-not $process.WaitForExit(250)) {
            Test-Budget
            $process.Refresh()
            if ($process.WorkingSet64 -gt $ramBytes) { throw 'Private Python installation RAM budget exceeded.' }
        }
        if ($process.ExitCode -ne 0) { throw ('Private setup stage failed (exit ' + $process.ExitCode + ').') }
    } finally {
        if (-not $process.HasExited) {
            & taskkill /PID $process.Id /T /F 2>$null | Out-Null
        }
        $process.Dispose()
    }
}

# The pinned pandas/pyFFTW set has no native Windows ARM64 wheels. On ARM64
# Windows use the OS x64 emulation path deliberately, and report that distinction.
$hostArch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($hostArch -notin @('AMD64', 'ARM64')) { throw 'A 64-bit x86/ARM Windows host is required.' }
if ($hostArch -eq 'ARM64') { Write-Host 'Windows ARM64: this pinned benchmark runs through x64 emulation.' }
$target = 'x86_64-pc-windows-msvc'
$archive = Join-Path $root 'uv.zip'
Get-OfficialFile "https://github.com/astral-sh/uv/releases/download/0.13.0/uv-$target.zip" $archive
$digest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($digest -ne '088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c') {
    throw 'Pinned uv SHA256 mismatch.'
}
Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $root 'uv')
Remove-Item -LiteralPath $archive
$uvFiles = @(Get-ChildItem -LiteralPath (Join-Path $root 'uv') -Filter uv.exe -Recurse -File)
if ($uvFiles.Count -ne 1) { throw 'Expected exactly one private uv executable.' }
$uv = $uvFiles[0].FullName
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'python'
$env:UV_CACHE_DIR = Join-Path $root 'cache'
$env:UV_TOOL_DIR = Join-Path $root 'tools'
$env:UV_NO_CONFIG = '1'
$env:UV_NO_PROGRESS = '1'
# PowerShell 7.5+ converts $null to an empty string in this .NET API;
# uv treats that as a supplied preference without a value. Remove it explicitly.
Remove-Item -LiteralPath Env:UV_PYTHON_PREFERENCE -ErrorAction SilentlyContinue
$env:UV_PYTHON_ARCH = 'x86_64'
Start-Bounded $uv @('--no-config', 'python', 'install', '3.12.14', '--no-bin', '--no-registry')
$python = (& $uv --no-config python find --managed-python 3.12.14).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Private Python lookup failed.' }
$helper = Join-Path $root 'benchmark_bootstrap.py'
$localHelper = if ($PSCommandPath -and ([IO.Path]::GetFileName($PSCommandPath) -eq 'run_benchmark.ps1')) {
    Join-Path $PSScriptRoot 'benchmark_bootstrap.py'
} else { '' }
if ($localHelper -and (Test-Path -LiteralPath $localHelper)) {
    Copy-Item -LiteralPath $localHelper -Destination $helper
} else {
    Get-OfficialFile "https://raw.githubusercontent.com/MichaDit/lpsd/$Revision/scripts/benchmark_bootstrap.py" $helper
}
Test-Budget
# The Python watchdog includes the complete run and handles its subprocess tree.
& $python $helper --revision $Revision --bootstrap-root $root --uv-path $uv @BenchmarkArgs
if ($LASTEXITCODE -ne 0) { throw ('Benchmark stopped (exit ' + $LASTEXITCODE + ').') }
} finally {
    foreach ($key in $savedEnvironment.Keys) {
        if ($null -eq $savedEnvironment[$key]) {
            Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($key, $savedEnvironment[$key], 'Process')
        }
    }
    [Net.ServicePointManager]::SecurityProtocol = $savedTlsPolicy
}
