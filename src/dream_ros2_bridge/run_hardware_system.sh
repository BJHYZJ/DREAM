#!/usr/bin/env bash
set -euo pipefail

ROS_SETUP=${ROS_SETUP:-/opt/ros/humble/setup.bash}
WORKSPACE_SETUP=${WORKSPACE_SETUP:-"$HOME/DREAM_ws/DREAM_ws/install/setup.bash"}
NODE_START_WAIT=${NODE_START_WAIT:-5}
RTABMAP_WAIT=${RTABMAP_WAIT:-10}

source_ros_env="source '$ROS_SETUP' && source '$WORKSPACE_SETUP'"

require_file() {
  local file=$1
  if [[ ! -f "$file" ]]; then
    echo "Missing setup file: $file" >&2
    exit 1
  fi
}

bring_up_can0() {
  local output

  if ! ip link show can0 >/dev/null 2>&1; then
    echo "can0 does not exist. Check the CAN adapter and driver." >&2
    exit 1
  fi

  if ip link show can0 | grep -q "<[^>]*UP"; then
    echo "[DREAM] can0 is already up; skipping CAN setup."
    return
  fi

  echo "[DREAM] Bringing up can0 at 500000 bitrate..."
  if output=$(sudo ip link set can0 up type can bitrate 500000 2>&1); then
    return
  fi

  echo "$output" >&2
  if grep -q "Device or resource busy" <<<"$output"; then
    echo "[DREAM] can0 is busy; assuming it is already in use and continuing."
    return
  fi

  if ip link show can0 | grep -q "<[^>]*UP"; then
    echo "[DREAM] can0 is up after setup attempt; continuing."
    return
  fi

  echo "Failed to bring up can0." >&2
  echo "If the error was 'Device or resource busy' and can0 is already usable, run:" >&2
  echo "  SKIP_CAN_SETUP=1 bash src/dream_ros2_bridge/run_hardware_system.sh" >&2
  exit 1
}

pane_command() {
  local title=$1
  local wait_seconds=$2
  local command=$3

  printf "bash -lc \"%s && printf '\\\\033]2;%s\\\\033\\\\\\\\' && echo '[DREAM] %s' && " "$source_ros_env" "$title" "$title"
  if (( wait_seconds > 0 )); then
    printf "echo '[DREAM] Waiting %ss before start...' && sleep %s && " "$wait_seconds" "$wait_seconds"
  fi
  printf "%s; echo; echo '[DREAM] %s exited. Press Ctrl+D to close this pane.'; exec bash\"" "$command" "$title"
}

write_terminator_config() {
  local config_file=$1
  local node_command=$2
  local rtabmap_command=$3
  local server_command=$4

  cat >"$config_file" <<EOF
[global_config]
[keybindings]
[profiles]
  [[default]]
[layouts]
  [[dream_hardware]]
    [[[window0]]]
      type = Window
      parent = ""
      title = DREAM hardware system
    [[[main_split]]]
      type = HPaned
      parent = window0
      position = 700
    [[[node_start]]]
      type = Terminal
      parent = main_split
      command = $node_command
      title = DREAM node start
    [[[right_split]]]
      type = VPaned
      parent = main_split
      position = 400
    [[[rtabmap_slam]]]
      type = Terminal
      parent = right_split
      command = $rtabmap_command
      title = DREAM RTAB-Map SLAM
    [[[dream_server]]]
      type = Terminal
      parent = right_split
      command = $server_command
      title = DREAM ROS2 bridge server
[plugins]
EOF
}

require_file "$ROS_SETUP"
require_file "$WORKSPACE_SETUP"

if ! command -v terminator >/dev/null 2>&1; then
  echo "terminator is required for split-pane terminal layout." >&2
  echo "Install it with: sudo apt install -y terminator" >&2
  exit 1
fi

if [[ "${SKIP_CAN_SETUP:-0}" == "1" ]]; then
  echo "[DREAM] SKIP_CAN_SETUP=1; skipping CAN setup."
else
  bring_up_can0
fi

node_command=$(pane_command \
  "DREAM node start" \
  0 \
  "ros2 launch dream_ros2_bridge dream_node_start.launch.py use_rviz:=false")

rtabmap_command=$(pane_command \
  "DREAM RTAB-Map SLAM" \
  "$NODE_START_WAIT" \
  "ros2 launch dream_ros2_bridge dream_rtabmap_slam.launch.py")

server_wait=$((NODE_START_WAIT + RTABMAP_WAIT))
server_command=$(pane_command \
  "DREAM ROS2 bridge server" \
  "$server_wait" \
  "ros2 launch dream_ros2_bridge dream_server.launch.py")

terminator_config=$(mktemp "${TMPDIR:-/tmp}/dream-terminator.XXXXXX")
write_terminator_config "$terminator_config" "$node_command" "$rtabmap_command" "$server_command"

echo "[DREAM] Opening one Terminator window with three split panes..."
terminator --config="$terminator_config" --layout=dream_hardware
