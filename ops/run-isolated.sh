#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

task_root=/home/ratulsarna/assistants/forge/work/glm-compact-dedup-20260923
arm=${1:?Choose fix}
[[ $arm == fix ]] || exit 2
checkout=$task_root/$arm
set -a
source "$checkout/.env"
set +a
# The production build serves routed experts on the fast thin-decode kernels;
# the stock kernels cost about 9 ms per decode step on this model.
if [[ ${GLM53_EXL3_MOE_FAST-} != 1 ]]; then
    echo 'GLM53_EXL3_MOE_FAST=1 is required to match production decode' >&2
    exit 2
fi
mode=${2:-isolated}
[[ $mode == isolated || $mode == managed ]] || exit 2
pid_file=$task_root/$arm-supervisor.pid
lease_helper=/home/ratulsarna/Work/spark-serve/scripts/cluster-gpu-lease.sh
token_file=/run/spark-models/cluster/glm5.3-f-exl3.owner-token
owner_token=
owner_hash=
lease_pending=0
if [[ $mode == managed ]]; then
    export SERVED_MODEL_NAME=glm5.3-f-exl3
    export API_HOST=100.67.246.44 PORT=32305
    export LOGDIR=/run/spark-models/glm5.3-f-exl3
    export HEAD_SCRIPT=$LOGDIR/head.inner.sh WORKER_SCRIPT=$LOGDIR/worker.inner.sh
    pid_file=$LOGDIR/supervisor.pid
fi
worker=ratulsarna@192.168.100.11
launcher_pid=
cleanup_done=0
owns_trial=0

log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"; }
fail() { log "ERROR: $*"; return 1; }
ssh_worker() { ssh -o BatchMode=yes -o ConnectTimeout=5 "$worker" "$@"; }
min_mem_available_kb=2097152
warn_mem_available_kb=3145728
low_streak_limit=1
require_zero_swap_growth=1
source /home/ratulsarna/Work/spark-serve/scripts/lib/two-node-memory.sh

lease() { printf '%s\n' "$owner_token" | ssh_worker bash "$lease_helper" "$@"; }

stop_rank() {
    local name=$1 model=$2 rank=$3 image=$4 hash=$5 id identity
    id=$(docker ps -aq --filter "name=^/${name}$") || return 1
    [[ -n $id ]] || return 0
    identity=$(docker inspect -f '{{.Config.Image}} {{index .Config.Labels "forge.glm-dd.model"}} {{index .Config.Labels "forge.glm-dd.rank"}} {{index .Config.Labels "forge.glm-dd.lifecycle-owner"}} {{index .Config.Labels "forge.glm-dd.lease-token-sha256"}}' "$id") || return 1
    [[ $identity == "$image $model $rank forge-isolated $hash" ]] || return 1
    # llama-swap allows only five seconds to cancel a loading process.
    docker stop -t 0 "$id" >/dev/null || return 1
    [[ $(docker inspect -f '{{.State.Running}}' "$id") == false ]]
}

cleanup() {
    ((cleanup_done == 0)) || return 0
    cleanup_done=1
    # Finish stopping the ranks and releasing the lease even if llama-swap
    # sends another SIGTERM while this runs, for example an unload during crash cleanup.
    trap - EXIT
    trap '' INT TERM
    local stopped=1 stop_head_pid
    if ((owns_trial == 1)); then
        log 'Stopping the owned GLM ranks'
        if [[ -n $launcher_pid ]]; then
            kill -- "-$launcher_pid" 2>/dev/null || true
            wait "$launcher_pid" 2>/dev/null || true
        fi
        stop_rank "$CONTAINER_HEAD" "$SERVED_MODEL_NAME" head "$IMAGE" "$owner_hash" &
        stop_head_pid=$!
        { declare -f stop_rank; printf '\nstop_rank "$@"\n'; } | \
            ssh_worker "$(printf 'bash -s -- %q %q worker %q %q' "$CONTAINER_WORKER" "$SERVED_MODEL_NAME" "$IMAGE" "$owner_hash")" || stopped=0
        wait "$stop_head_pid" || stopped=0
        rm -f "$pid_file"
    fi
    if ((lease_pending == 1)); then
        local release_rc=1
        if ((stopped == 1)); then
            lease release && release_rc=0 || release_rc=$?
        fi
        # 73 means another owner holds the lease, for example when our claim lost,
        # so this token can never release it and would only block the next start.
        if ((release_rc == 0 || release_rc == 73)); then
            rm -f "$token_file"
        else
            log 'Cleanup incomplete; retaining the GPU lease and owner token'
            return 1
        fi
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Hold one lock across all arms, including startup and cleanup.
# A read-only descriptor also permits flock and works in llama-swap's sandbox.
exec {trial_lock_fd}<"$task_root/supervisor.lock"
flock -n "$trial_lock_fd" || fail 'Another isolated supervisor owns this GPU pair'

mkdir -p "$LOGDIR"
empty_running='import json,sys; d=json.load(sys.stdin); assert not (d if isinstance(d,list) else d["running"]), "Production is busy"'
if [[ $mode == managed ]]; then
    # /run is cleared on reboot; the lease helper creates this directory the same way.
    install -d -m 0700 /run/spark-models/cluster
    exec {managed_lock_fd}>/run/spark-models/cluster/glm5.3-f-exl3.wrapper.lock
    flock -n "$managed_lock_fd" || fail 'Another managed GLM supervisor is running'
    [[ ! -e $token_file ]] || fail 'A previous GPU lease needs cleanup'
    owner_token=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
    umask 077
    printf '%s\n' "$owner_token" > "$token_file"
    owner_hash=$(printf '%s' "$owner_token" | sha256sum | cut -d ' ' -f 1)
    export SPARK_SERVE_LEASE_TOKEN_SHA256=$owner_hash
    lease_pending=1
    lease claim "spark-a:$SERVED_MODEL_NAME"
    ssh_worker curl -fsS --max-time 30 -X POST http://127.0.0.1:30000/api/models/unload >/dev/null
    deadline=$((SECONDS + 300))
    until ssh_worker python3 /home/ratulsarna/Work/spark-serve/scripts/reconcile-model-containers.py --idle-probe /run/spark-models/cluster; do
        ((SECONDS < deadline)) || fail 'Worker did not become idle'
        sleep 2
    done
else
    curl -fsS --max-time 5 http://127.0.0.1:30000/running | python3 -c "$empty_running"
fi
ssh_worker 'curl -fsS --max-time 5 http://127.0.0.1:30000/running' | python3 -c "$empty_running"
[[ -z $(docker ps --filter label=forge.glm-dd.lifecycle-owner --format '{{.Names}}') ]] || fail 'Another trial is running on head'
[[ -z $(ssh_worker "docker ps --filter label=forge.glm-dd.lifecycle-owner --format '{{.Names}}'") ]] || fail 'Another trial is running on worker'
capture_safety_baseline
safety_green
gid_resolver=/home/ratulsarna/Work/spark-serve/scripts/resolve-roce-gid.py
export HEAD_GID
HEAD_GID=$(python3 "$gid_resolver" "$HEAD_CX7_IB" "$HEAD_IP" "$HEAD_CX7_IF")
export WORKER_GID
WORKER_GID=$(ssh_worker python3 - "$WORKER_CX7_IB" "$WORKER_IP" "$WORKER_CX7_IF" < "$gid_resolver")
printf '%s\n' "$$" > "$pid_file"
record_memory() {
    local trial_head_memory trial_worker_memory
    trial_head_memory=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
    trial_worker_memory=$(ssh_worker "awk '/MemAvailable:/{print \$2}' /proc/meminfo")
    printf '%s %s %s\n' "$(date --iso-8601=seconds)" "$trial_head_memory" "$trial_worker_memory" >> "$LOGDIR/memory.log"
}
owns_trial=1
setsid bash "$checkout/start.sh" start > "$LOGDIR/start.log" 2>&1 &
launcher_pid=$!
while kill -0 "$launcher_pid" 2>/dev/null; do
    safety_green
    record_memory
    sleep 2
done
wait "$launcher_pid"
launcher_pid=
if [[ $arm != baseline ]]; then
    python3 - "$LOGDIR/start.log" "$arm" <<'PY'
import re
import sys
from pathlib import Path

matches = re.findall(r"usable block ids: ([0-9]+)", Path(sys.argv[1]).read_text())
if not matches:
    raise SystemExit("Cannot verify the candidate's cache capacity; stopping trial")
usable = int(matches[-1])
minimum = 449 if sys.argv[2] == 'recipe-headroom' else 378
if usable < minimum:
    raise SystemExit(f"This candidate requires at least {minimum} usable cache blocks; found {usable}")
print(f"Cache capacity verified: {usable} usable blocks, minimum {minimum}", flush=True)
PY
fi
log "READY arm=$arm port=$PORT"
while true; do
    [[ $mode != managed ]] || lease owned >/dev/null
    safety_green
    record_memory
    curl -fsS --max-time 5 "http://${API_HOST:-127.0.0.1}:$PORT/health" >/dev/null
    sleep 2
done
