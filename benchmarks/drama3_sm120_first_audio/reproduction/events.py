"""Build Perfetto request timelines and first-audio phase measurements."""

from collections import Counter, defaultdict
from pathlib import Path
import gzip
import json
import sys


def summarize(folder):
    clients = json.loads((folder / "http-events.json").read_text())
    records = {c["request_id"]: c for c in clients}
    by_request = defaultdict(list)
    schema = Counter()
    for f in (folder / "events").rglob("*.jsonl"):
        for line in f.open():
            obj = json.loads(line)
            if obj.get("request_id") not in records or "event_name" not in obj:
                continue
            schema[(obj["stage"], obj["event_name"])] += 1
            by_request[obj["request_id"]].append(obj)
    traces = []
    details = []
    for i, r in enumerate(clients):
        rid = r["request_id"]
        ev = sorted(by_request[rid], key=lambda e: e["timestamp_ns"])

        def t(e):
            return e["timestamp_ns"]

        begin = r["start_epoch_ns"]
        first = begin + r["first_ns"] - r["start_ns"]
        ends = begin + r["end_ns"] - r["start_ns"]
        emitted = [
            e
            for e in ev
            if e["event_name"] == "stage_stream_chunk_sent"
            and e["stage"] == "tts_engine"
        ]
        voc = [
            e for e in ev if e["event_name"] == "drama3_vocoder_window_decode_started"
        ]
        finished = [
            e for e in ev if e["event_name"] == "drama3_vocoder_window_decode_finished"
        ]
        trace = dict(request_id=rid, ttfa_ms=(first - begin) / 1e6)
        milestones = {
            "client start": begin,
            "client first PCM": first,
            "client EOF": ends,
        }
        if voc and finished:
            start = voc[0]
            meta = start["metadata"]
            chunk = meta.get("chunk_id")
            window = meta.get("window_index")
            send = next(
                (e for e in emitted if e["metadata"].get("chunk_id") == chunk), None
            )
            end = next(
                (
                    e
                    for e in finished
                    if e["metadata"].get("window_index") == window and t(e) >= t(start)
                ),
                None,
            )
            if send and end:
                span = [begin, t(send), t(start), t(end), first]
                trace["first_window_chunk_id"] = chunk
                trace["phases_ms"] = [(b - a) / 1e6 for a, b in zip(span, span[1:])]
                milestones.update(
                    dict(
                        zip(
                            [
                                "engine sent window-completing codes",
                                "vocoder starts first window",
                                "vocoder finishes first window",
                            ],
                            span[1:-1],
                        )
                    )
                )
            later = [
                e["metadata"]
                for e in voc[1:]
                if e["metadata"].get("emitted_frames", 0) > 0
            ]
            trace["later_windows"] = len(later)
            trace["later_windows_redecoded_history"] = sum(
                m["decoded_frames"] > m["emitted_frames"] for m in later
            )
        pipeline = [
            ("admission", "coordinator", "request_admission"),
            ("preprocessing done", "preprocessing", "stage_complete"),
            ("engine queue", "tts_engine", "scheduler_queue_enter"),
            ("prefill starts", "tts_engine", "scheduler_prefill_start"),
            ("prefill ends", "tts_engine", "scheduler_prefill_end"),
            ("first engine chunk", "tts_engine", "stage_first_stream_chunk_sent"),
        ]
        chain = {"client start": begin}
        for display, stage, event_name in pipeline:
            v = next(
                (
                    x
                    for x in ev
                    if x["stage"] == stage and x["event_name"] == event_name
                ),
                None,
            )
            if v:
                chain[display] = t(v)
        trace["pre_code_ms"] = {k: (v - begin) / 1e6 for k, v in chain.items()}
        milestones.update(chain)
        for name, ns in sorted(milestones.items(), key=lambda x: x[1]):
            traces.append(
                dict(
                    ph="i",
                    s="t",
                    pid=1,
                    tid=i + 1,
                    name=name,
                    ts=ns / 1000,
                    args={"request_id": rid},
                )
            )
        traces.append(
            dict(
                ph="X",
                pid=1,
                tid=i + 1,
                name="time to first PCM",
                ts=begin / 1000,
                dur=(first - begin) / 1000,
                args={"request_id": rid},
            )
        )
        details.append(trace)
    with gzip.open(folder / "request-timeline.json.gz", "wt") as out:
        json.dump({"traceEvents": traces}, out)
    output = dict(
        rows=sorted(details, key=lambda x: -x["ttfa_ms"]),
        matched=len([r for r in details if "phases_ms" in r]),
        event_counts={"/".join(k): v for k, v in schema.items()},
    )
    (folder / "phases.json").write_text(json.dumps(output, indent=2))
    print(folder, len(details), output["matched"], list(schema)[:20])


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(Path(p))
