#!/usr/bin/env python3
"""把 SerialTrace CSV 按时间整理成一份可读文档。

输入是 standard 在 record/auto_aim 下生成的 *.csv：
  mono_ns,wall_time,event,...
输出：
  *.timeline.csv  固定时间桶的 TX/RX/视频帧对齐表
  *.timeline.md   摘要 + 时间轴表格

用法：
  python3 tools/pack_trace.py record/auto_aim/2026-11-17_21-30-00.csv
  python3 tools/pack_trace.py trace.csv --bucket-ms 20 --md-limit 3000
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Event:
    mono_ns: int
    wall: str
    values: dict


def read_events(path: Path) -> tuple[dict[str, list[Event]], dict[str, str]]:
    meta: dict[str, str] = {}
    events: dict[str, list[Event]] = {"tx": [], "rx": [], "video": []}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = row.get("event", "")
            if event == "meta":
                for token in (row.get("note") or "").split(";"):
                    if "=" in token:
                        key, value = token.split("=", 1)
                        meta[key] = value
                continue
            if event not in events:
                continue
            try:
                mono = int(row["mono_ns"])
            except (KeyError, TypeError, ValueError):
                continue
            events[event].append(
                Event(mono_ns=mono, wall=row.get("wall_time") or "", values=row)
            )
    for event_list in events.values():
        event_list.sort(key=lambda item: item.mono_ns)
    return events, meta


def last_before(items: list[Event], mono_ns: int) -> Event | None:
    index = bisect.bisect_right(items, mono_ns, key=lambda item: item.mono_ns) - 1
    return items[index] if index >= 0 else None


def as_float(value: str | None, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(value: str | None, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def fmt(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def build_timeline(
    events: dict[str, list[Event]],
    bucket_ms: int,
) -> tuple[list[dict[str, str]], int, int]:
    tx = events["tx"]
    rx = events["rx"]
    video = events["video"]
    all_events = tx + rx + video
    if not all_events:
        return [], 0, 0

    start = min(event.mono_ns for event in all_events)
    end = max(event.mono_ns for event in all_events)
    period_ns = max(bucket_ms, 1) * 1_000_000
    buckets = (end - start + period_ns - 1) // period_ns if end >= start else 0
    rows: list[dict[str, str]] = []

    for bucket_index in range(buckets):
        bucket_end = start + (bucket_index + 1) * period_ns
        tx_event = last_before(tx, bucket_end)
        rx_event = last_before(rx, bucket_end)
        video_event = last_before(video, bucket_end)

        tx_values = tx_event.values if tx_event else {}
        rx_values = rx_event.values if rx_event else {}
        video_values = video_event.values if video_event else {}

        wall = (
            (rx_event or tx_event or video_event).wall
            if (rx_event or tx_event or video_event)
            else ""
        )
        capture_ns = as_int(video_values.get("capture_mono_ns"))
        rows.append(
            {
                "wall_time": wall,
                "bucket_mono_ns": str(bucket_end),
                "offset_s": f"{(bucket_end - start) / 1_000_000_000:.3f}",
                "tx_yaw_deg": fmt(as_float(tx_values.get("yaw_deg"))),
                "tx_pitch_deg": fmt(as_float(tx_values.get("pitch_deg"))),
                "tx_fire": "" if tx_values.get("fire") is None else str(as_int(tx_values.get("fire"), 0)),
                "rx_yaw_deg": fmt(as_float(rx_values.get("yaw_deg"))),
                "rx_pitch_deg": fmt(as_float(rx_values.get("pitch_deg"))),
                "rx_roll_deg": fmt(as_float(rx_values.get("roll_deg"))),
                "rx_mode": "" if rx_values.get("mode") is None else str(as_int(rx_values.get("mode"), -1)),
                "rx_robot_id": ""
                if rx_values.get("robot_id") is None
                else str(as_int(rx_values.get("robot_id"), -1)),
                "rx_bullet_speed": fmt(as_float(rx_values.get("bullet_speed"))),
                "video_frame": ""
                if video_event is None or not video_values.get("video_frame_index")
                else str(as_int(video_values.get("video_frame_index"), -1)),
                "video_capture_mono_ns": "" if capture_ns is None else str(capture_ns),
                "video_offset_s": ""
                if capture_ns is None
                else f"{(capture_ns - start) / 1_000_000_000:.3f}",
            }
        )
    return rows, start, end


def write_outputs(
    source: Path,
    events: dict[str, list[Event]],
    meta: dict[str, str],
    rows: list[dict[str, str]],
    start: int,
    end: int,
    bucket_ms: int,
    md_limit: int,
) -> tuple[Path, Path]:
    stem = source.with_suffix("")
    timeline_csv = Path(str(stem) + ".timeline.csv")
    timeline_md = Path(str(stem) + ".timeline.md")

    fields = list(rows[0].keys()) if rows else [
        "wall_time",
        "bucket_mono_ns",
        "offset_s",
        "tx_yaw_deg",
        "tx_pitch_deg",
        "tx_fire",
        "rx_yaw_deg",
        "rx_pitch_deg",
        "rx_roll_deg",
        "rx_mode",
        "rx_robot_id",
        "rx_bullet_speed",
        "video_frame",
        "video_capture_mono_ns",
        "video_offset_s",
    ]
    with timeline_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    duration = (end - start) / 1_000_000_000 if end >= start else 0.0
    table_rows = rows[:md_limit]
    lines = [
        "# 串口 / 录像时间轴",
        "",
        f"- trace: `{source}`",
        f"- video: `{meta.get('video', '-')}`",
        f"- 时间范围: `{table_rows[0]['wall_time'] if table_rows else '-'}` 起，约 `{duration:.3f} s`",
        f"- 分桶: `{bucket_ms} ms`",
        f"- TX 帧: `{len(events['tx'])}`，RX 帧: `{len(events['rx'])}`，录像帧: `{len(events['video'])}`",
        "",
        "## 时间轴",
        "",
        "| 时间 | 偏移 s | TX yaw | TX pitch | fire | RX yaw | RX pitch | RX roll | mode | id | 弹速 | 视频帧 | 视频采集偏移 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            "| {wall_time} | {offset_s} | {tx_yaw_deg} | {tx_pitch_deg} | {tx_fire} | "
            "{rx_yaw_deg} | {rx_pitch_deg} | {rx_roll_deg} | {rx_mode} | {rx_robot_id} | "
            "{rx_bullet_speed} | {video_frame} | {video_offset_s} |".format(**row)
        )
    if len(rows) > md_limit:
        lines.append("")
        lines.append(f"> Markdown 只显示前 {md_limit} 行，完整数据见 `{timeline_csv.name}`。")
    timeline_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return timeline_csv, timeline_md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, help="SerialTrace CSV")
    parser.add_argument("--bucket-ms", type=int, default=10)
    parser.add_argument("--md-limit", type=int, default=2000)
    args = parser.parse_args()

    events, meta = read_events(args.trace)
    rows, start, end = build_timeline(events, args.bucket_ms)
    timeline_csv, timeline_md = write_outputs(
        args.trace, events, meta, rows, start, end, args.bucket_ms, args.md_limit
    )
    print(timeline_csv)
    print(timeline_md)


if __name__ == "__main__":
    main()
