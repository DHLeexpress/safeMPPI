#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/projects}"
SAFEGPC_DIR="${SAFEGPC_DIR:-$ROOT/safeGPC}"
CFM_DIR="${CFM_DIR:-$ROOT/cfm_mppi}"

pwd
find "$HOME" -maxdepth 4 -type d \( -name "safeGPC" -o -name "cfm_mppi" \) 2>/dev/null

echo "CFM_DIR=$CFM_DIR"
git -C "$CFM_DIR" status --short
git -C "$CFM_DIR" branch --show-current
git -C "$CFM_DIR" rev-parse HEAD
find "$CFM_DIR" -maxdepth 4 -type f | sort > /tmp/cfm_mppi_filemap.txt

echo "SAFEGPC_DIR=$SAFEGPC_DIR"
git -C "$SAFEGPC_DIR" status --short
git -C "$SAFEGPC_DIR" branch --all
git -C "$SAFEGPC_DIR" rev-parse HEAD
find "$SAFEGPC_DIR" -maxdepth 5 -type f | sort > /tmp/safegpc_filemap.txt
