#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: NODE_IP=10.90.67.82 bash launch.sh {02|03} {0|1|2} /path/to/MiniMax-H3" >&2
  exit 2
fi

topology=$1
stage_id=$2
model=$3
node_ip=${NODE_IP:?Set NODE_IP to this machine 10.90.67.x address}
master_ip=${MASTER_IP:-10.90.67.82}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "$topology:$stage_id:$node_ip" in
  02:0:10.90.67.82|02:1:10.90.67.83|03:0:10.90.67.82|03:1:10.90.67.83|03:2:10.90.67.82) ;;
  *) echo "Invalid topology/stage/node placement: $topology/$stage_id/$node_ip" >&2; exit 2 ;;
esac

case "$topology" in
  02)
    template="$root/02-encoder8-dit8.yaml.in"
    base_config="$root/../../../vllm_omni/deploy/minimax_h3_disaggregated.yaml"
    ;;
  03)
    template="$root/03-encoder8-dit8-decoder8.yaml.in"
    base_config="$root/../../../vllm_omni/deploy/minimax_h3_disaggregated_decode.yaml"
    ;;
esac

run_dir=${RUN_DIR:-"$PWD/h3-a100-run-$topology"}
mkdir -p "$run_dir"
deploy_config="$run_dir/deploy-stage${stage_id}.yaml"
sed -e "s|__BASE_CONFIG__|$base_config|g" \
  -e "s/__NODE_IP__/$node_ip/g" -e "s/__MASTER_IP__/$master_ip/g" \
  "$template" > "$deploy_config"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400
export no_proxy="${no_proxy:+$no_proxy,}10.90.67.82,10.90.67.83"
export NO_PROXY="$no_proxy"

args=(serve "$model" --omni --trust-remote-code
  --deploy-config "$deploy_config" --stage-id "$stage_id"
  --omni-master-address "$master_ip" --omni-master-port 36000
  --stage-init-timeout 1800 --init-timeout 1800)
if [[ "$stage_id" == 0 ]]; then
  args+=(--host 0.0.0.0 --port 18091)
else
  args+=(--headless)
fi

exec vllm-omni "${args[@]}"
