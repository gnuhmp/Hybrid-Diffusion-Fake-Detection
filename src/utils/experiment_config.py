"""Load YAML experiment specs, merge with CLI, and snapshot configs per run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from typing import Any
from uuid import uuid4

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment file must be a mapping: {path}")
    return data


def filter_for_parser(
    data: dict[str, Any], parser: argparse.ArgumentParser
) -> dict[str, Any]:
    known = set()
    for a in parser._actions:
        dest = getattr(a, "dest", None)
        if dest and dest != "help":
            known.add(dest)
    return {k: v for k, v in data.items() if k in known}


def parse_with_experiment(
    argv: list[str] | None,
    parser: argparse.ArgumentParser,
) -> tuple[argparse.Namespace, dict[str, Any], str | None, str | None]:
    """
    If argv contains --experiment PATH, load YAML defaults before parsing the rest.
    Returns (args, raw_for_defaults, experiment_abspath, experiment_type).
    """
    if argv is None:
        import sys

        argv = sys.argv[1:]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--experiment", "-e", default=None, metavar="PATH")
    pre_args, rest = pre.parse_known_args(argv)

    experiment_abspath: str | None = None
    experiment_type: str | None = None
    raw: dict[str, Any] = {}
    if pre_args.experiment:
        experiment_abspath = os.path.abspath(pre_args.experiment)
        full = dict(load_yaml(experiment_abspath))
        k = full.pop("type", None)
        if isinstance(k, str):
            experiment_type = k
        raw = full
        raw.pop("description", None)
        raw.pop("experiment", None)

    filtered = filter_for_parser(raw, parser)
    parser.set_defaults(**filtered)
    args = parser.parse_args(rest)
    setattr(args, "experiment_source_path", experiment_abspath)

    return args, raw, experiment_abspath, experiment_type


def new_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def save_experiment_start(
    run_dir: str,
    experiment_source_path: str | None,
    run_id: str,
    started_at: str,
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    meta = {
        "run_id": run_id,
        "started_at": started_at,
        "experiment_source_path": experiment_source_path,
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if experiment_source_path and os.path.isfile(experiment_source_path):
        shutil.copy2(
            experiment_source_path, os.path.join(run_dir, "experiment_source.yaml")
        )


def save_experiment_finish(run_dir: str, resolved: dict[str, Any]) -> None:
    """Write full resolved spec (hyperparameters + metrics + metadata)."""
    os.makedirs(run_dir, exist_ok=True)
    with open(
        os.path.join(run_dir, "experiment_resolved.yaml"), "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            resolved, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    with open(
        os.path.join(run_dir, "experiment_resolved.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(resolved, f, indent=2)


def snapshot_experiment(
    run_dir: str,
    experiment_source_path: str | None,
    resolved: dict[str, Any],
) -> None:
    """Backward-compatible single-shot snapshot (manifest + source copy + resolved)."""
    rid = resolved.get("run_id", new_run_id())
    started = resolved.get("started_at", datetime.now().isoformat())
    save_experiment_start(run_dir, experiment_source_path, rid, started)
    save_experiment_finish(run_dir, resolved)
