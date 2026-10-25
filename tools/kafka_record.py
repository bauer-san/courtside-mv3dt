"""Record MV3DT Kafka messages (protobuf Frame) to gzipped JSON lines.

Runs with NVIDIA's mv3dt_venv python (kafka-python + protobuf) and imports schema_pb2 from the
app's utils/. Starts at the topic end (fresh run), stops after --idle-timeout seconds without
messages once at least one has arrived, or after --first-message-timeout with none.

  <app>/mv3dt_venv/bin/python tools/kafka_record.py --app <app> --out run/kafka.jsonl.gz
"""

import argparse
import os
import gzip
import json
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", type=Path, required=True, help="deepstream-tracker-3d-multi-view directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--broker", default="localhost:9092")
    ap.add_argument("--topic", default="mv3dt")
    ap.add_argument("--json", action="store_true", help="messages are JSON text (court tracker topic), not protobuf")
    ap.add_argument("--first-message-timeout", type=float, default=900)
    ap.add_argument("--idle-timeout", type=float, default=60)
    args = ap.parse_args()

    sys.path.insert(0, str(args.app / "utils"))
    from google.protobuf.json_format import MessageToDict
    from kafka import KafkaConsumer
    from schema_pb2 import Frame

    consumer = KafkaConsumer(bootstrap_servers=args.broker, group_id=f"courtside_record_{args.topic}_{os.getpid()}_{int(time.time())}",
                             auto_offset_reset="latest", enable_auto_commit=False, value_deserializer=lambda x: x)
    consumer.subscribe([args.topic])
    deadline = time.time() + 10
    while not consumer.assignment() and time.time() < deadline:
        consumer.poll(timeout_ms=100)
    consumer.seek_to_end()
    print(f"recording {args.topic} -> {args.out}", flush=True)

    start = last = time.time()
    count = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        while True:
            now = time.time()
            if count == 0 and now - start > args.first_message_timeout:
                break
            if count and now - last > args.idle_timeout:
                break
            for msgs in consumer.poll(timeout_ms=1000).values():
                for m in msgs:
                    if args.json:
                        f.write(m.value.decode() + "\n")
                    else:
                        frame = Frame()
                        frame.ParseFromString(m.value)
                        f.write(json.dumps(MessageToDict(frame), separators=(",", ":")) + "\n")
                    count += 1
                    last = time.time()
    consumer.close()
    print(f"recorded {count} messages", flush=True)


if __name__ == "__main__":
    main()
