#!/usr/bin/env python3
"""Switch config/dong.yaml command_guard profile for one-at-a-time ablation.

Usage:
  python3 tools/guard_profile.py all_on
  python3 tools/guard_profile.py minus_jump
  ...
The baseline is config/dong.yaml.guard_all_on; do not edit that file.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "dong.yaml"
BASE = ROOT / "config" / "dong.yaml.guard_all_on"

PROFILES = {
    "all_on": {},
    "minus_jump": {
        "max_yaw_jump_deg": "360.0",
        "max_pitch_jump_deg": "180.0",
    },
    "minus_rate": {
        "max_yaw_rate_deg_s": "1000000000.0",
        "max_pitch_rate_deg_s": "1000000000.0",
    },
    "minus_lowpass": {
        "filter_time_constant_s": "0.0",
    },
    "minus_acquire": {
        "acquire_ms": "0",
    },
    "minus_smoothing": {
        "max_yaw_rate_deg_s": "1000000000.0",
        "max_pitch_rate_deg_s": "1000000000.0",
        "filter_time_constant_s": "0.0",
    },
    "minus_jump_smoothing": {
        "max_yaw_jump_deg": "360.0",
        "max_pitch_jump_deg": "180.0",
        "max_yaw_rate_deg_s": "1000000000.0",
        "max_pitch_rate_deg_s": "1000000000.0",
        "filter_time_constant_s": "0.0",
    },
    "minus_target_change": {
        "reset_on_target_change": "false",
    },
    "minus_fire_gate": {
        "fire_settle_ms": "0",
        "fire_alignment_tolerance_deg": "180.0",
    },
    "minus_distance_gate": {
        "min_distance_m": "0.01",
        "max_distance_m": "100.0",
    },
    "raw": {
        "max_yaw_jump_deg": "360.0",
        "max_pitch_jump_deg": "180.0",
        "max_yaw_rate_deg_s": "1000000000.0",
        "max_pitch_rate_deg_s": "1000000000.0",
        "filter_time_constant_s": "0.0",
        "acquire_ms": "0",
        "reset_on_target_change": "false",
        "fire_settle_ms": "0",
        "fire_alignment_tolerance_deg": "180.0",
        "min_distance_m": "0.01",
        "max_distance_m": "100.0",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=sorted(PROFILES))
    args = parser.parse_args()

    if not BASE.exists():
        shutil.copyfile(CFG, BASE)
        print(f"created baseline {BASE}")

    text = BASE.read_text(encoding="utf-8")
    changes = PROFILES[args.profile]

    text = re.sub(
        r"(?m)^    profile:.*$",
        f"    profile: {args.profile}",
        text,
        count=1,
    )
    for key, value in changes.items():
        pattern = rf"(?m)^    {re.escape(key)}:.*$"
        if not re.search(pattern, text):
            print(f"missing key in baseline: {key}", file=sys.stderr)
            return 2
        text = re.sub(pattern, f"    {key}: {value}", text, count=1)

    CFG.write_text(text, encoding="utf-8")
    print(f"profile={args.profile} written to {CFG}")
    if changes:
        for key, value in changes.items():
            print(f"  {key}: {value}")
    else:
        print("  all restrictions at baseline values")
    print("restart the standard program to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
