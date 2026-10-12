#!/bin/sh
# Pinned private bootstrap for 64-bit Linux/macOS, and prepared Termux.
set -eu

usage() {
    cat <<'EOF'
Usage: sh run_benchmark.sh --revision FULL_GIT_COMMIT [benchmark options]

Downloads private uv/Python, pinned scientific wheels and the requested source;
uses an existing C compiler or a private pinned Zig compiler. No administrator
rights, global package installation or shell profile changes are required.

Options are forwarded to the bounded benchmark, including:
  --max-hours 24        Total setup + benchmark budget (0 < hours <= 24)
  --ram-gib auto        Conservative RAM allocation budget
  --disk-gib auto       Temporary dataset/workspace disk budget
  --point-factors 1 2   Requested actual output-point ratios (default)
  --bootstrap-disk-gib 2  Separate private installation cap (default)
  --compiler auto|system|zig
  --github-login       Optional explicit browser/device login for public upload

Results stay local by default. --upload-github --accept-public-upload explicitly
publishes the sanitized report under your GitHub account through existing gh.
Set LPSD_BENCHMARK_BASE_DIR to select the parent of the fresh private directory.
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then usage; exit 0; fi
if [ "${1:-}" != "--revision" ] || [ -z "${2:-}" ]; then usage >&2; exit 2; fi
lpsd_revision=$2
shift 2
case "$lpsd_revision" in *[!0-9a-fA-F]*|'') echo 'A full Git commit is required.' >&2; exit 2;; esac
if [ "${#lpsd_revision}" -ne 40 ]; then echo 'A full Git commit is required.' >&2; exit 2; fi
lpsd_revision=$(printf '%s' "$lpsd_revision" | tr '[:upper:]' '[:lower:]')
lpsd_hours=24
lpsd_setup_gib=2
lpsd_ram_gib=auto
lpsd_pending=''
for lpsd_argument in "$@"; do
    case "$lpsd_pending" in
        hours) lpsd_hours=$lpsd_argument; lpsd_pending=''; continue;;
        disk) lpsd_setup_gib=$lpsd_argument; lpsd_pending=''; continue;;
        ram) lpsd_ram_gib=$lpsd_argument; lpsd_pending=''; continue;;
    esac
    case "$lpsd_argument" in
        --max-hours) lpsd_pending=hours;;
        --max-hours=*) lpsd_hours=${lpsd_argument#*=};;
        --bootstrap-disk-gib) lpsd_pending=disk;;
        --bootstrap-disk-gib=*) lpsd_setup_gib=${lpsd_argument#*=};;
        --ram-gib) lpsd_pending=ram;;
        --ram-gib=*) lpsd_ram_gib=${lpsd_argument#*=};;
    esac
done
if [ -n "$lpsd_pending" ]; then echo 'Missing numeric option value.' >&2; exit 2; fi
lpsd_seconds=$(awk -v h="$lpsd_hours" 'BEGIN {
    if (h !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ || h <= 0 || h > 24) exit 1;
    printf "%.0f", int(h * 3600)
}') || { echo 'Use 0 < --max-hours <= 24.' >&2; exit 2; }
lpsd_setup_kib=$(awk -v g="$lpsd_setup_gib" 'BEGIN {
    if (g !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ || g <= 0) exit 1;
    printf "%.0f", int(g * 1024 * 1024)
}') || { echo 'Use a positive --bootstrap-disk-gib.' >&2; exit 2; }
if [ "$lpsd_seconds" -lt 1 ]; then echo 'The total budget is less than one second.' >&2; exit 2; fi
lpsd_ram_requested_kib=$(awk -v g="$lpsd_ram_gib" 'BEGIN {
    if (g == "auto") { print 0; exit }
    if (g !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ || g <= 0) exit 1;
    printf "%.0f", int(g * 1024 * 1024)
}') || { echo 'Use --ram-gib auto or a positive number.' >&2; exit 2; }
lpsd_started=$(date +%s)
LPSD_BENCHMARK_STARTED_EPOCH=$lpsd_started
LPSD_BENCHMARK_DEADLINE_EPOCH=$((lpsd_started + lpsd_seconds))
export LPSD_BENCHMARK_STARTED_EPOCH LPSD_BENCHMARK_DEADLINE_EPOCH
lpsd_base=${LPSD_BENCHMARK_BASE_DIR:-$PWD}
if [ ! -d "$lpsd_base" ]; then echo 'The selected parent directory does not exist.' >&2; exit 2; fi
lpsd_root=$(mktemp -d "$lpsd_base/lpsd-benchmark.XXXXXXXX")
chmod 700 "$lpsd_root"
lpsd_child=''
lpsd_guard=''
cleanup() {
    if [ -n "$lpsd_child" ]; then kill "$lpsd_child" 2>/dev/null || :; fi
    if [ -n "$lpsd_guard" ]; then kill "$lpsd_guard" 2>/dev/null || :; fi
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM HUP

check_budget() {
    lpsd_now=$(date +%s)
    if [ "$lpsd_now" -ge "$LPSD_BENCHMARK_DEADLINE_EPOCH" ]; then
        echo 'Setup/run deadline reached.' >&2; exit 124
    fi
    lpsd_used=$(du -sk "$lpsd_root" | awk '{print $1}')
    lpsd_free=$(df -Pk "$lpsd_root" | awk 'END {print $4}')
    if [ "$lpsd_used" -gt "$lpsd_setup_kib" ] || [ "$lpsd_free" -lt 32768 ]; then
        echo 'Private setup disk budget/free-space reserve reached.' >&2; exit 2
    fi
}

download() {
    check_budget
    lpsd_remaining=$((LPSD_BENCHMARK_DEADLINE_EPOCH - $(date +%s)))
    if [ "$lpsd_remaining" -gt 300 ]; then lpsd_remaining=300; fi
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 --fail --location --silent --show-error \
        --connect-timeout 20 --max-time "$lpsd_remaining" --max-filesize 167772160 \
        "$1" --output "$2"
    check_budget
}

verify_sha() {
    if command -v sha256sum >/dev/null 2>&1; then
        lpsd_actual=$(sha256sum "$1" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        lpsd_actual=$(shasum -a 256 "$1" | awk '{print $1}')
    else echo 'A SHA256 utility (sha256sum or shasum) is required.' >&2; exit 2; fi
    if [ "$lpsd_actual" != "$2" ]; then echo 'Pinned uv SHA256 mismatch.' >&2; exit 2; fi
}

# uv's Python installation has no subprocess compiler. This pre-Python guard
# terminates that process at the deadline; the Python stage watches whole trees.
linux_child_memory() {
    lpsd_memory_parent=$1; lpsd_memory_child=$2
    shift 2
    awk -v parent="$lpsd_memory_parent" -v child="$lpsd_memory_child" '
        function record() {
            if (p == parent && (n == child || (!has_ns && pid == child))) {
                found++; found_pid=pid;
                if (pid !~ /^[0-9]+$/ || pid <= 0 || (has_ns && ns_first != pid)) bad=1;
                if (state == "Z" || state == "X" || state == "x") {
                    if (threads != 1) bad=1;
                } else if (!has_rss || state == "" || r !~ /^[0-9]+$/ || bad_rss) bad=1;
                else total += r;
            }
        }
        BEGIN {
            # Read paths individually: an exited foreign process must not
            # abort discovery of our uniquely identified, readable child.
            for (file=1; file<ARGC; file++) {
                p=0; n=0; pid=0; r=0; has_ns=0; has_rss=0;
                state=""; threads=0; ns_first=0; bad_rss=0;
                while ((reading=(getline < ARGV[file])) > 0) {
                    if ($1 == "Pid:") pid=$2;
                    if ($1 == "PPid:") p=$2;
                    if ($1 == "NSpid:") { has_ns=1; ns_first=$2; n=$NF }
                    if ($1 == "State:") state=$2;
                    if ($1 == "Threads:") threads=$2;
                    if ($1 == "VmRSS:") { r=$2; has_rss=1; bad_rss=(NF != 3 || $3 != "kB") }
                }
                close(ARGV[file]);
                if (reading == 0) record();
            }
            if (found != 1 || bad) exit 1;
            printf "%.0f %.0f\n", found_pid, total;
            exit;
        }
    ' "$@"
}

bounded_python_install() {
    "$@" &
    lpsd_child=$!
    (
        lpsd_child_proc=''
        while kill -0 "$lpsd_child" 2>/dev/null; do
            if [ "$(date +%s)" -ge "$LPSD_BENCHMARK_DEADLINE_EPOCH" ]; then
                kill -KILL "$lpsd_child" 2>/dev/null || :; exit
            fi
            lpsd_used=$(du -sk "$lpsd_root" | awk '{print $1}')
            lpsd_free=$(df -Pk "$lpsd_root" | awk 'END {print $4}')
            if [ "$lpsd_used" -gt "$lpsd_setup_kib" ] || [ "$lpsd_free" -lt 32768 ]; then
                kill -KILL "$lpsd_child" 2>/dev/null || :; exit
            fi
            if [ "$lpsd_system" = Linux ]; then
                if [ -n "$lpsd_child_proc" ]; then
                    lpsd_child_sample=$(linux_child_memory "$lpsd_proc_pid" "$lpsd_child" \
                        "/proc/$lpsd_child_proc/status" 2>/dev/null) || lpsd_child_sample=''
                else
                    lpsd_child_sample=$(linux_child_memory "$lpsd_proc_pid" "$lpsd_child" \
                        /proc/[0-9]*/status 2>/dev/null) || lpsd_child_sample=''
                fi
                # Only our verified target is cached. Its PID, parent and
                # namespace identity are checked again on each direct read.
                lpsd_child_proc=${lpsd_child_sample%% *}
                lpsd_child_rss=${lpsd_child_sample#* }
                case "$lpsd_child_proc" in ''|*[!0-9]*) lpsd_child_rss='';; esac
                if [ "$lpsd_child_sample" = "$lpsd_child_proc" ]; then lpsd_child_rss=''; fi
            else
                lpsd_child_rss=$(ps -o rss= -p "$lpsd_child" 2>/dev/null) || lpsd_child_rss=''
                lpsd_child_rss=$(printf '%s\n' "$lpsd_child_rss" | awk '
                    NF != 1 || $1 !~ /^[0-9]+$/ { bad=1 }
                    { count++; total += $1 }
                    END { if (bad || count != 1) exit 1; printf "%.0f\n", total }
                ') || lpsd_child_rss=''
            fi
            # No readable counter is not a zero-byte process. If it exited
            # during the probe, the best-effort kill is harmless and wait
            # below still obtains the installer exit status.
            case "$lpsd_child_rss" in ''|*[!0-9]*)
                kill -KILL "$lpsd_child" 2>/dev/null || :; exit;;
            esac
            if [ "$lpsd_child_rss" -gt "$lpsd_ram_kib" ]; then
                kill -KILL "$lpsd_child" 2>/dev/null || :; exit
            fi
            sleep 1
        done
    ) &
    lpsd_guard=$!
    lpsd_status=0
    wait "$lpsd_child" || lpsd_status=$?
    kill "$lpsd_guard" 2>/dev/null || :
    wait "$lpsd_guard" 2>/dev/null || :
    lpsd_child=''; lpsd_guard=''
    if [ "$lpsd_status" -ne 0 ]; then echo 'Private Python installation failed or exhausted its budget.' >&2; exit "$lpsd_status"; fi
}

if ! command -v curl >/dev/null 2>&1; then echo 'curl is required.' >&2; exit 2; fi
lpsd_helper="$lpsd_root/benchmark_bootstrap.py"
case "$0" in
    */run_benchmark.sh)
        lpsd_local_helper="$(dirname "$0")/benchmark_bootstrap.py"
        if [ -f "$lpsd_local_helper" ]; then cp "$lpsd_local_helper" "$lpsd_helper"; fi;;
esac
if [ ! -f "$lpsd_helper" ]; then
    download "https://raw.githubusercontent.com/MichaDit/lpsd/$lpsd_revision/scripts/benchmark_bootstrap.py" "$lpsd_helper"
fi

if [ -n "${TERMUX_VERSION:-}" ]; then
    if ! command -v python >/dev/null 2>&1; then
        echo 'Termux experimental path needs its existing python/scientific/clang/fftw packages; see docs/portable-benchmark.md.' >&2; exit 2
    fi
    exec python "$lpsd_helper" --revision "$lpsd_revision" --bootstrap-root "$lpsd_root" "$@"
fi

lpsd_system=$(uname -s)
lpsd_arch=$(uname -m)
if [ "$lpsd_system" = Linux ]; then
    lpsd_proc_pid=''
    while read -r lpsd_status_key lpsd_status_value lpsd_status_rest; do
        if [ "$lpsd_status_key" = 'Pid:' ]; then lpsd_proc_pid=$lpsd_status_value; break; fi
    done < /proc/self/status
    lpsd_ram_kib=$(awk '/^MemAvailable:/ {printf "%.0f", int($2 * .8)}' /proc/meminfo)
else
    # psutil is not installed yet; half the physical RAM is a conservative
    # macOS pre-Python ceiling. The Python stage detects resources again.
    lpsd_ram_kib=$(sysctl -n hw.memsize | awk '{printf "%.0f", int($1 / 2048)}')
fi
if [ -z "$lpsd_ram_kib" ] || [ "$lpsd_ram_kib" -le 0 ]; then
    echo 'Cannot determine a pre-Python RAM budget.' >&2; exit 2
fi
if [ "$lpsd_ram_requested_kib" -gt 0 ] && [ "$lpsd_ram_requested_kib" -lt "$lpsd_ram_kib" ]; then
    lpsd_ram_kib=$lpsd_ram_requested_kib
fi
if [ "$lpsd_ram_kib" -lt 131072 ]; then echo 'Private setup needs at least 128 MiB of admitted RAM.' >&2; exit 2; fi
if ! command -v ps >/dev/null 2>&1; then echo 'ps is required to monitor private Python installation.' >&2; exit 2; fi
if [ "$lpsd_system" = Linux ]; then
    lpsd_libc=$(getconf GNU_LIBC_VERSION 2>/dev/null || :)
    if ! printf '%s\n' "$lpsd_libc" | awk '{split($2, v, "."); exit !($1 == "glibc" && (v[1] > 2 || (v[1] == 2 && v[2] >= 28)))}'; then
        echo 'The pinned automatic wheel path requires Linux glibc 2.28+; see the manual/Termux route.' >&2; exit 2
    fi
fi
case "$lpsd_system:$lpsd_arch" in
    Linux:x86_64) lpsd_target=x86_64-unknown-linux-gnu; lpsd_sha=1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154;;
    Linux:aarch64|Linux:arm64) lpsd_target=aarch64-unknown-linux-gnu; lpsd_sha=3ccfb6af6e242433eb552f7d9676abd5412c6595c497e990d9c8cb7b5bd4d2c3;;
    Darwin:x86_64) lpsd_target=x86_64-apple-darwin; lpsd_sha=5f44dcbde809b632f47c36fadb241cb4d6f9af71d0c8f172b5d2026d3dde742c;;
    Darwin:arm64) lpsd_target=aarch64-apple-darwin; lpsd_sha=a9c1b29002cf3c83f07fa9cd8a887a3be0107d90e23189721221e7257db8e3d6;;
    *) echo 'This bootstrap supports 64-bit x86/ARM Linux or macOS; use the documented experimental/manual route elsewhere.' >&2; exit 2;;
esac
download "https://github.com/astral-sh/uv/releases/download/0.13.0/uv-$lpsd_target.tar.gz" "$lpsd_root/uv.tar.gz"
verify_sha "$lpsd_root/uv.tar.gz" "$lpsd_sha"
tar --no-same-owner -xzf "$lpsd_root/uv.tar.gz" -C "$lpsd_root"
rm "$lpsd_root/uv.tar.gz"
lpsd_uv="$lpsd_root/uv-$lpsd_target/uv"
UV_PYTHON_INSTALL_DIR="$lpsd_root/python"
UV_CACHE_DIR="$lpsd_root/cache"
UV_TOOL_DIR="$lpsd_root/tools"
UV_NO_CONFIG=1
UV_NO_PROGRESS=1
UV_CONCURRENT_DOWNLOADS=2
UV_CONCURRENT_INSTALLS=2
unset UV_PYTHON_PREFERENCE
export UV_PYTHON_INSTALL_DIR UV_CACHE_DIR UV_TOOL_DIR UV_NO_CONFIG UV_NO_PROGRESS UV_CONCURRENT_DOWNLOADS UV_CONCURRENT_INSTALLS
bounded_python_install "$lpsd_uv" --no-config python install 3.12.14 --no-bin
check_budget
lpsd_python=$("$lpsd_uv" --no-config python find --managed-python 3.12.14)
exec "$lpsd_python" "$lpsd_helper" --revision "$lpsd_revision" --bootstrap-root "$lpsd_root" --uv-path "$lpsd_uv" "$@"
