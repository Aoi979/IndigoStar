#!/usr/bin/env bash
set -euo pipefail

# Read connection defaults from environment so that host/port/remote_dir are not
# hard-coded in the repository. Set these variables in your shell or in a
# separate, untracked .env file before running the script.
default_host="${INDIGO_STAR_UPLOAD_HOST:-}"
default_port="${INDIGO_STAR_UPLOAD_PORT:-}"
default_remote_dir="${INDIGO_STAR_UPLOAD_REMOTE_DIR:-}"

usage() {
  cat <<USAGE
Usage:
  ./upload_to_server.sh [options] [remote_dir]

Environment defaults (all optional if provided via options):
  INDIGO_STAR_UPLOAD_HOST        SSH host, e.g. user@example.com
  INDIGO_STAR_UPLOAD_PORT        SSH/scp port, e.g. 22
  INDIGO_STAR_UPLOAD_REMOTE_DIR  Remote directory, e.g. /path/to/remote/dir

Options:
      --host HOST        SSH host, overrides INDIGO_STAR_UPLOAD_HOST
  -p, --port PORT        SSH/scp port, overrides INDIGO_STAR_UPLOAD_PORT
  -n, --dry-run          Show commands without uploading.
  -h, --help             Show this help.

Examples:
  INDIGO_STAR_UPLOAD_HOST=user@example.com \
  INDIGO_STAR_UPLOAD_PORT=22 \
  INDIGO_STAR_UPLOAD_REMOTE_DIR=/path/to/remote/dir \
    ./upload_to_server.sh

  ./upload_to_server.sh --host user@example.com --port 22 /path/to/remote/dir
  ./upload_to_server.sh --dry-run

Equivalent login command:
  ssh -p <port> <host>
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

if [[ -z "$host" ]]; then
  echo "Error: host is not set. Use --host or set INDIGO_STAR_UPLOAD_HOST." >&2
  usage >&2
  exit 2
fi

if [[ -z "$port" ]]; then
  echo "Error: port is not set. Use -p/--port or set INDIGO_STAR_UPLOAD_PORT." >&2
  usage >&2
  exit 2
fi

if [[ -z "$remote_dir" ]]; then
  echo "Error: remote_dir is not set. Pass it as an argument or set INDIGO_STAR_UPLOAD_REMOTE_DIR." >&2
  usage >&2
  exit 2
fi

for tool in tar scp ssh; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool is required but was not found." >&2
    exit 1
  fi
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archive_name="indigostar-$(date +%Y%m%d-%H%M%S).tar.gz"
archive_path="$(mktemp -t indigostar-upload.XXXXXX.tar.gz)"
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
  tar -C "$repo_root" "${tar_excludes[@]}" -hcvf /dev/null . >/dev/null
  echo
  echo "Would run:"
  echo "  ssh -p $port $host \"if [ -e $remote_dir_q ]; then rm -rf $remote_dir_q; fi; mkdir -p $remote_dir_q\""
  echo "  scp -P $port $archive_name $host:$remote_archive"
  echo "  ssh -p $port $host tar -xzf $remote_archive_q -C $remote_dir_q '&&' rm -f $remote_archive_q"
  exit 0
fi

echo "Packaging $repo_root -> $archive_path"
tar -C "$repo_root" "${tar_excludes[@]}" -hczf "$archive_path" .

echo "Preparing remote directory: ${host}:${remote_dir}"
ssh -p "$port" "$host" "if [ -e $remote_dir_q ]; then rm -rf $remote_dir_q; fi; mkdir -p $remote_dir_q"

echo "Uploading archive with scp -P $port"
scp -P "$port" "$archive_path" "${host}:${remote_archive}"

echo "Extracting archive on remote"
ssh -p "$port" "$host" "tar -xzf $remote_archive_q -C $remote_dir_q && rm -f $remote_archive_q"

echo "Done. Login with:"
echo "  ssh -p $port $host"
echo "Remote project:"
echo "  cd $remote_dir"
