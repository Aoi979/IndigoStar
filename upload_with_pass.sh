#!/usr/bin/env bash
set -euo pipefail

default_host="user@example.com"
default_port="22"
default_remote_dir="/path/to/remote/dir"

usage() {
  cat <<USAGE
Usage:
  ./upload_to_server.sh [options] [remote_dir]

Defaults:
  host       ${default_host}
  port       ${default_port}
  remote_dir ${default_remote_dir}

Options:
      --host HOST        SSH host, default: ${default_host}
  -p, --port PORT        SSH/scp port, default: ${default_port}
  -n, --dry-run          Show commands without uploading.
  -h, --help             Show this help.

Examples:
  ./upload_to_server.sh
  ./upload_to_server.sh /path/to/remote/dir
  ./upload_to_server.sh --dry-run
  ./upload_to_server.sh --host user@example.com --port 22 /path/to/remote/dir

Equivalent login command:
  sshpass -p 'YOUR_PASSWORD' ssh -p ${default_port} ${default_host}
USAGE
}

host="$default_host"
port="$default_port"
remote_dir="$default_remote_dir"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      if [[ $# -lt 2 ]]; then
        echo "--host requires a value." >&2
        exit 2
      fi
      host="$2"
      shift 2
      ;;
    -p|--port)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a value." >&2
        exit 2
      fi
      port="$2"
      shift 2
      ;;
    -n|--dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      remote_dir="$1"
      shift
      if [[ $# -gt 0 ]]; then
        echo "Only one remote_dir is allowed." >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

for tool in tar scp ssh; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool is required but was not found." >&2
    exit 1
  fi
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archive_name="learncuda-$(date +%Y%m%d-%H%M%S).tar.gz"
archive_path="$(mktemp -t learncuda-upload.XXXXXX.tar.gz)"
remote_archive="/tmp/${archive_name}"

tar_excludes=(
  --exclude='./.git'
  --exclude='./build'
  --exclude='./.cache'
  --exclude='./cmake-build*'
  --exclude='./CMakeFiles'
  --exclude='./CMakeCache.txt'
  --exclude='./compile_commands.json'
  --exclude='*.ncu-rep'
  --exclude='*.nsys-rep'
  --exclude='*.qdrep'
  --exclude='*.o'
  --exclude='*.a'
  --exclude='*.so'
  --exclude='__pycache__'
)

remote_dir_q="$(printf '%q' "$remote_dir")"
remote_archive_q="$(printf '%q' "$remote_archive")"

cleanup() {
  rm -f "$archive_path"
}
trap cleanup EXIT

if [[ "$dry_run" -eq 1 ]]; then
  echo "Would package files from: $repo_root"
  tar -C "$repo_root" "${tar_excludes[@]}" -cvf /dev/null . >/dev/null
  echo
  echo "Would run:"
  echo "  sshpass -p 'YOUR_PASSWORD' ssh -p $port $host mkdir -p $remote_dir_q"
  echo "  sshpass -p 'YOUR_PASSWORD' scp -P $port $archive_name $host:$remote_archive"
  echo "  sshpass -p 'YOUR_PASSWORD' ssh -p $port $host tar -xzf $remote_archive_q -C $remote_dir_q '&&' rm -f $remote_archive_q"
  exit 0
fi

echo "Packaging $repo_root -> $archive_path"
tar -C "$repo_root" "${tar_excludes[@]}" -czf "$archive_path" .

echo "Creating remote directory: ${host}:${remote_dir}"
sshpass -p 'YOUR_PASSWORD' ssh -p "$port" "$host" "mkdir -p $remote_dir_q"

echo "Uploading archive with sshpass -p 'YOUR_PASSWORD' scp -P $port"
sshpass -p 'YOUR_PASSWORD' scp -P "$port" "$archive_path" "${host}:${remote_archive}"

echo "Extracting archive on remote"
sshpass -p 'YOUR_PASSWORD' ssh -p "$port" "$host" "tar -xzf $remote_archive_q -C $remote_dir_q && rm -f $remote_archive_q"

echo "Done. Login with:"
echo "  sshpass -p 'YOUR_PASSWORD' ssh -p $port $host"
echo "Remote project:"
echo "  cd $remote_dir"
