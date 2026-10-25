#!/usr/bin/env bash
# Start/stop/status for the brokers MV3DT needs:
#   - MQTT (native mosquitto systemd service, :1883): inter-camera tracker communication
#   - Kafka 4.2.0 (external/kafka_2.13-4.2.0, :9092, topic mv3dt): tracking metadata -> BEV visualizer
# Kafka keeps its KRaft storage in /tmp/kraft-combined-logs (as NVIDIA's setup script does),
# so after a reboot it is re-formatted on start.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAFKA_DIR="$ROOT/external/kafka_2.13-4.2.0"
LOGS=/tmp/kraft-combined-logs

kafka_up() { ss -ltn | grep -q ':9092 '; }

start() {
  systemctl is-active --quiet mosquitto || { echo "mosquitto not running: sudo systemctl start mosquitto"; exit 1; }
  if kafka_up; then
    echo "Kafka already listening on :9092"
  else
    cd "$KAFKA_DIR"
    if [[ ! -f "$LOGS/meta.properties" ]]; then
      bin/kafka-storage.sh format --standalone -t "$(bin/kafka-storage.sh random-uuid)" -c config/server.properties >/dev/null
    fi
    nohup bin/kafka-server-start.sh config/server.properties >"$ROOT/experiments/kafka.log" 2>&1 &
    for _ in {1..30}; do kafka_up && break; sleep 2; done
    kafka_up || { echo "Kafka failed to start, see experiments/kafka.log"; exit 1; }
  fi
  "$KAFKA_DIR/bin/kafka-topics.sh" --bootstrap-server localhost:9092 --create --if-not-exists \
    --topic mv3dt --partitions 1 --replication-factor 1 --config retention.ms=30000 >/dev/null
  status
}

stop() {
  kafka_up && "$KAFKA_DIR/bin/kafka-server-stop.sh" || true
  echo "Kafka stopped (mosquitto is a system service; left running)"
}

status() {
  printf "mosquitto :1883  %s\n" "$(systemctl is-active mosquitto)"
  if kafka_up; then
    printf "kafka     :9092  active, topics: %s\n" \
      "$("$KAFKA_DIR/bin/kafka-topics.sh" --bootstrap-server localhost:9092 --list 2>/dev/null | tr '\n' ' ')"
  else
    echo "kafka     :9092  inactive"
  fi
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status"; exit 1 ;;
esac
