"""Round-sharded store for offline executed-window SFM expansion.

Each context contributes at most one resolved H=10 plan: the plan whose first
action was physically executed.  The unexecuted B-1 verifier queries are search
diagnostics and never enter this store.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch


class ExecutedRoundShard:
    VERSION = 1

    def __init__(self, round_i):
        self.round_i = int(round_i)
        self.contexts = []
        self.windows = []
        self.errors = []
        self._context_keys = set()
        self._window_contexts = set()

    def add_context(
        self, *, scenario_id, gamma, step, state, hp10, low5, hist,
        ped_xy, ped_vel,
    ):
        key = (int(scenario_id), round(float(gamma), 8), int(step))
        if key in self._context_keys:
            raise ValueError(f"context already stored: {key}")
        self._context_keys.add(key)
        context_id = len(self.contexts)
        self.contexts.append(dict(
            context_id=context_id,
            round=self.round_i,
            scenario_id=int(scenario_id),
            gamma=float(gamma),
            step=int(step),
            state=np.asarray(state, np.float32),
            hp10=np.asarray(hp10, np.float32),
            low5=np.asarray(low5, np.float32),
            hist=np.asarray(hist, np.float32),
            ped_xy=np.asarray(ped_xy, np.float32),
            ped_vel=np.asarray(ped_vel, np.float32),
        ))
        return context_id

    def add_executed_window(
        self, context_id, controls, result, *, execution_source, nvp_context,
        candidate_id=None, acquisition_step=None, sigma=None, hp_margin=None,
        mode=None,
    ):
        if not bool(result.get("resolved")):
            raise ValueError("unresolved executed verifier result cannot enter D")
        if int(result.get("y", -1)) not in (0, 1):
            raise ValueError("resolved executed window needs binary y")
        if not bool(result.get("full_h")) or int(result.get("terminal_step", -1)) != 10:
            raise ValueError("offline training D requires exact full-H=10 labels")
        controls = np.asarray(controls, np.float32)
        if tuple(controls.shape) != (10, 2) or not np.isfinite(controls).all():
            raise ValueError("offline training D requires finite controls [10,2]")
        components = (
            bool(result.get("taskspace")),
            bool(result.get("collision_free")),
            bool(result.get("certificate")),
        )
        if int(result["y"]) != int(all(components)):
            raise ValueError("verifier y disagrees with its exact components")
        context_id = int(context_id)
        if not 0 <= context_id < len(self.contexts):
            raise IndexError("unknown context")
        if context_id in self._window_contexts:
            raise ValueError("a context can contribute at most one executed window")
        self._window_contexts.add(context_id)
        window_id = len(self.windows)
        self.windows.append(dict(
            window_id=window_id,
            query_id=window_id,
            context_id=context_id,
            controls=controls,
            y=int(result["y"]),
            taskspace=bool(result["taskspace"]),
            collision_free=bool(result["collision_free"]),
            certificate=bool(result["certificate"]),
            full_h=True,
            terminal_step=10,
            train_eligible=bool(result["y"]),
            execution_source=str(execution_source),
            nvp_context=bool(nvp_context),
            candidate_id=None if candidate_id is None else int(candidate_id),
            acquisition_step=(
                None if acquisition_step is None else int(acquisition_step)
            ),
            sigma=None if sigma is None else float(sigma),
            hp_margin=None if hp_margin is None else float(hp_margin),
            mode=mode,
            verifier_diagnostics=dict(result["diagnostics"]),
        ))
        return window_id

    def add_error(self, *, context_id, candidate_id, execution_source, error):
        self.errors.append(dict(
            context_id=int(context_id),
            candidate_id=None if candidate_id is None else int(candidate_id),
            execution_source=str(execution_source),
            error=str(error),
        ))

    @property
    def D(self):
        return list(self.windows)

    @property
    def Dplus(self):
        return [row for row in self.windows if row["y"] == 1]

    @property
    def Dminus(self):
        return [row for row in self.windows if row["y"] == 0]

    def validate(self):
        for expected, context in enumerate(self.contexts):
            if int(context["context_id"]) != expected:
                raise AssertionError("context IDs are not dense")
        seen = set()
        for expected, window in enumerate(self.windows):
            if (
                int(window["window_id"]) != expected
                or int(window["query_id"]) != expected
            ):
                raise AssertionError("window IDs are not dense")
            context_id = int(window["context_id"])
            if not 0 <= context_id < len(self.contexts):
                raise AssertionError("window references missing context")
            if context_id in seen:
                raise AssertionError("multiple training windows share one context")
            seen.add(context_id)
            if not window["full_h"] or int(window["terminal_step"]) != 10:
                raise AssertionError("non-H10 window entered training D")
        if len(self.Dplus) + len(self.Dminus) != len(self.D):
            raise AssertionError("D+/D- must exactly partition resolved executed D")
        return dict(
            round=self.round_i,
            contexts=len(self.contexts),
            D=len(self.D),
            Dplus=len(self.Dplus),
            Dminus=len(self.Dminus),
            errors=len(self.errors),
            unresolved_contexts=len(self.contexts) - len(self.D),
        )

    def save(self, path):
        path = os.path.abspath(os.fspath(path))
        summary = self.validate()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = path + ".tmp"
        torch.save(dict(
            version=self.VERSION,
            round=self.round_i,
            contexts=self.contexts,
            windows=self.windows,
            errors=self.errors,
            summary=summary,
        ), temporary)
        os.replace(temporary, path)
        digest = sha256_file(path)
        complete = path + ".COMPLETE.json"
        with open(complete + ".tmp", "w") as stream:
            json.dump(dict(
                status="OFFLINE_EXECUTED_ROUND_SHARD_COMPLETE",
                file=path,
                sha256=digest,
                **summary,
            ), stream, indent=2)
        os.replace(complete + ".tmp", complete)
        return dict(path=path, sha256=digest, complete=complete, **summary)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["version"]) != cls.VERSION:
            raise ValueError("unsupported executed-round shard version")
        value = cls(payload["round"])
        value.contexts = payload["contexts"]
        value.windows = payload["windows"]
        value.errors = payload["errors"]
        value._context_keys = {
            (
                int(row["scenario_id"]),
                round(float(row["gamma"]), 8),
                int(row["step"]),
            )
            for row in value.contexts
        }
        value._window_contexts = {
            int(row["context_id"]) for row in value.windows
        }
        value.validate()
        return value


def context_for(shard, window):
    return shard.contexts[int(window["context_id"])]


def positive_records(shard):
    return [(shard, row) for row in shard.Dplus]


def negative_records(shard):
    return [(shard, row) for row in shard.Dminus]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
