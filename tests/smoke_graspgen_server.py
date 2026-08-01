#!/usr/bin/env python3
"""Smoke-test GraspGen HTTP service on :8121."""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time

import numpy as np
import requests


def _npy_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8121")
    p.add_argument("--num-grasps", type=int, default=50)
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    health = f"{args.url}/health"
    for i in range(60):
        try:
            r = requests.get(health, timeout=3)
            if r.ok and r.json().get("status") == "ok":
                print("health:", r.json())
                break
        except requests.RequestException:
            pass
        time.sleep(2)
        print(f"waiting for {health} ({i+1}/60)...")
    else:
        print("FAIL: server not healthy", file=sys.stderr)
        return 1

    # Synthetic box-like cloud around origin
    rng = np.random.default_rng(0)
    pc = rng.uniform(-0.04, 0.04, size=(800, 3)).astype(np.float32)

    payload = {
        "pc_base64": _npy_b64(pc),
        "num_grasps": args.num_grasps,
        "topk_num_grasps": args.topk,
        "min_grasps": 1,
        "max_tries": 3,
        "remove_outliers": False,
    }
    t0 = time.time()
    r = requests.post(f"{args.url}/infer", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    dt = time.time() - t0
    print(
        f"infer ok: num_grasps={data['num_grasps']} "
        f"server_ms={data['infer_ms']:.1f} wall_s={dt:.2f}"
    )

    # plan_point_clouds alias
    payload2 = {
        "pc_full_base64": _npy_b64(pc),
        "pc_segment_base64": _npy_b64(pc),
        "num_grasps": args.num_grasps,
        "topk_num_grasps": args.topk,
        "min_grasps": 1,
        "max_tries": 2,
        "remove_outliers": False,
    }
    r2 = requests.post(f"{args.url}/plan_point_clouds", json=payload2, timeout=300)
    r2.raise_for_status()
    print("plan_point_clouds ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
