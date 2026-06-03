#!/usr/bin/env python3
"""Cycle-accurate simulator for APS->CMT2 control/data transfer ideas.

This is intentionally small and model-driven:

- It simulates loop scopes, entry/body/next handoff, and frame ownership.
- It can model a nested loop either hierarchically or as a flattened single loop.
- It is meant for principle verification, not for semantic lowering or codegen.

The simulator keeps FIFO semantics explicit:

- current-cycle reads come from the current queue contents
- writes land in next-cycle queue contents
- all queues commit at the end of each cycle

The default examples are:

- a simple pipelined loop
- a nested hierarchical loop
- a flattened equivalent loop

Usage:

  python tools/aps_cmt2_cycle_sim.py --example simple-pipeline
  python tools/aps_cmt2_cycle_sim.py --example nested-hierarchical
  python tools/aps_cmt2_cycle_sim.py --example nested-flattened
  python tools/aps_cmt2_cycle_sim.py --example nested-hierarchical --trace
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

Payload = Dict[str, Any]
TransformFn = Callable[[Payload, int], Payload]


def identity_transform(payload: Payload, iter_idx: int) -> Payload:
    return dict(payload)


@dataclass
class Reg:
    """A cycle-accurate register: read current, write next, commit at cycle end."""

    name: str
    cur: Any
    nxt: Any = None
    writes_this_cycle: int = 0
    reads_this_cycle: int = 0
    max_writes_per_cycle: int = 0
    max_reads_per_cycle: int = 0

    def read(self) -> Any:
        self.reads_this_cycle += 1
        return self.cur

    def write(self, value: Any) -> None:
        self.writes_this_cycle += 1
        if self.writes_this_cycle > 1:
            raise RuntimeError(
                f"register {self.name} written multiple times in one cycle"
            )
        self.nxt = value

    def commit(self) -> None:
        self.max_writes_per_cycle = max(self.max_writes_per_cycle, self.writes_this_cycle)
        self.max_reads_per_cycle = max(self.max_reads_per_cycle, self.reads_this_cycle)
        if self.writes_this_cycle:
            self.cur = self.nxt
        self.nxt = None
        self.writes_this_cycle = 0
        self.reads_this_cycle = 0

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cur": self.cur,
            "max_reads_per_cycle": self.max_reads_per_cycle,
            "max_writes_per_cycle": self.max_writes_per_cycle,
        }


@dataclass
class Queue2:
    """A double-buffered FIFO: current-cycle reads, next-cycle writes."""

    name: str
    depth: int = 1
    cur: Deque[Any] = field(default_factory=deque)
    nxt: Deque[Any] = field(default_factory=deque)
    pushes_this_cycle: int = 0
    pops_this_cycle: int = 0
    max_pushes_per_cycle: int = 0
    max_pops_per_cycle: int = 0

    def has_cur(self) -> bool:
        return bool(self.cur)

    def push(self, item: Any) -> None:
        if self.depth <= 0:
            raise RuntimeError(f"queue {self.name} has invalid depth {self.depth}")
        if len(self.cur) + len(self.nxt) >= self.depth:
            raise RuntimeError(
                f"queue {self.name} overflow: depth={self.depth} "
                f"occupancy={len(self.cur) + len(self.nxt)}"
            )
        self.pushes_this_cycle += 1
        self.nxt.append(item)

    def pop(self) -> Any:
        if not self.cur:
            raise RuntimeError(f"queue {self.name} underflow")
        self.pops_this_cycle += 1
        return self.cur.popleft()

    def peek(self) -> Any:
        return self.cur[0]

    def commit(self) -> None:
        if self.nxt:
            self.cur.extend(self.nxt)
            self.nxt.clear()
        self.max_pushes_per_cycle = max(self.max_pushes_per_cycle, self.pushes_this_cycle)
        self.max_pops_per_cycle = max(self.max_pops_per_cycle, self.pops_this_cycle)
        self.pushes_this_cycle = 0
        self.pops_this_cycle = 0

    def empty(self) -> bool:
        return not self.cur and not self.nxt

    def occupancy(self) -> int:
        return len(self.cur) + len(self.nxt)

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "depth": self.depth,
            "cur_len": len(self.cur),
            "nxt_len": len(self.nxt),
            "max_pushes_per_cycle": self.max_pushes_per_cycle,
            "max_pops_per_cycle": self.max_pops_per_cycle,
        }


@dataclass
class BasicBodySpec:
    """A non-loop body modeled as a fixed-latency transform."""

    name: str
    latency: int = 1
    transform: TransformFn = identity_transform


BodyNode = Any


@dataclass
class LoopSpec:
    """A loop scope.

    The body is an ordered list of basic blocks and/or nested loop scopes.
    """

    name: str
    tripcount: int
    mode: str = "pipeline"  # "pipeline" or "non_pipeline"
    launch_mode: str = "carry"  # "carry" or "stream"
    body: List[BodyNode] = field(default_factory=lambda: [BasicBodySpec("body")])
    max_inflight: Optional[int] = None

    def is_pipeline(self) -> bool:
        return self.mode == "pipeline"

    def capacity(self) -> int:
        if self.max_inflight is not None:
            return self.max_inflight
        return self.tripcount if self.is_pipeline() else 1


@dataclass
class Frame:
    """One activation / iteration in a loop scope."""

    frame_id: int
    tag: str
    iter_idx: int
    payload: Payload
    step_idx: int = 0
    remaining: int = 0
    child: Optional["LoopRuntime"] = None
    started: bool = False


@dataclass
class FrameToken:
    """A completion token that carries the minimal information needed by next."""

    frame_id: int
    tag: str
    iter_idx: int
    payload: Payload


@dataclass
class LoopRuntime:
    """A cycle-accurate model of one loop scope."""

    spec: LoopSpec
    payload_seed: Payload
    tag_prefix: str = ""
    auto_prime: bool = True
    entry_token: Queue2 = field(init=False)
    loop_input_bundle: Queue2 = field(init=False)
    body_admit_fifo: Queue2 = field(init=False)
    body_done_token_fifo: Queue2 = field(init=False)
    loop_frame_to_next_fifo: Queue2 = field(init=False)
    loop_output_bundle: Queue2 = field(init=False)
    busy_reg: Reg = field(init=False)
    loop_state_reg: Reg = field(init=False)
    credit_reg: Optional[Reg] = field(init=False, default=None)
    tag_reg: Optional[Reg] = field(init=False, default=None)
    launch_cursor_reg: Optional[Reg] = field(init=False, default=None)
    active_frames: List[Frame] = field(default_factory=list)
    next_frame_id: int = 0
    max_live_contexts: int = 0
    max_active_frames: int = 0
    stall_count: int = 0
    cycle: int = 0

    def __post_init__(self) -> None:
        self.entry_token = Queue2(f"{self.spec.name}.entry_token", depth=1)
        self.loop_input_bundle = Queue2(f"{self.spec.name}.loop_input", depth=1)
        admitted_depth = self.spec.capacity() if self.spec.is_pipeline() and self.spec.launch_mode == "stream" else 1
        self.body_admit_fifo = Queue2(f"{self.spec.name}.body_admit", depth=admitted_depth)
        self.body_done_token_fifo = Queue2(f"{self.spec.name}.body_done", depth=admitted_depth)
        self.loop_frame_to_next_fifo = Queue2(f"{self.spec.name}.frame_to_next", depth=admitted_depth)
        self.loop_output_bundle = Queue2(f"{self.spec.name}.loop_output", depth=admitted_depth)
        self.busy_reg = Reg(f"{self.spec.name}.busy", False)
        self.loop_state_reg = Reg(f"{self.spec.name}.state", deepcopy(self.payload_seed))
        if self.spec.is_pipeline():
            self.credit_reg = Reg(f"{self.spec.name}.credit", self.spec.capacity())
            self.tag_reg = Reg(f"{self.spec.name}.tag", 0)
            if self.spec.launch_mode == "stream":
                self.launch_cursor_reg = Reg(f"{self.spec.name}.launch_cursor", 0)
        if self.auto_prime:
            self.prime(self.payload_seed)

    def prime(self, payload: Payload) -> None:
        """Seed the scope with its initial entry token and input bundle."""
        self.entry_token.push(1)
        self.loop_input_bundle.push(deepcopy(payload))
        self.commit()

    def _new_frame(self, iter_idx: int, payload: Payload) -> Frame:
        frame_id = self.next_frame_id
        self.next_frame_id += 1
        tag = f"{self.spec.name}:{frame_id}"
        return Frame(
            frame_id=frame_id,
            tag=tag,
            iter_idx=iter_idx,
            payload=deepcopy(payload),
            step_idx=0,
            remaining=0,
            child=None,
            started=False,
        )

    def _can_accept_new_frame(self) -> bool:
        if self.spec.mode == "non_pipeline":
            return len(self.active_frames) == 0
        return len(self.active_frames) < self.spec.capacity()

    def _body_complete(self, token: FrameToken, credit_accum: Optional[List[int]] = None) -> None:
        """Handle the frame after body completion and decide continue/exit."""
        if self.spec.is_pipeline() and self.spec.launch_mode == "stream":
            if self.loop_output_bundle.occupancy() >= self.loop_output_bundle.depth:
                self.stall_count += 1
                return
            self.loop_output_bundle.push(deepcopy(token.payload))
            if credit_accum is not None:
                credit_accum[0] += 1
            return

        next_iter = token.iter_idx + 1
        if next_iter < self.spec.tripcount:
            next_frame = self._new_frame(next_iter, deepcopy(token.payload))
            if self.body_admit_fifo.occupancy() >= self.body_admit_fifo.depth:
                self.stall_count += 1
                return
            self.body_admit_fifo.push(next_frame)
        else:
            if self.loop_output_bundle.occupancy() >= self.loop_output_bundle.depth:
                self.stall_count += 1
                return
            self.loop_output_bundle.push(deepcopy(token.payload))
        if self.spec.is_pipeline() and credit_accum is not None:
            credit_accum[0] += 1
        if not self.spec.is_pipeline():
            self.loop_state_reg.write(deepcopy(token.payload))
            self.busy_reg.write(False)

    def _live_contexts(self) -> int:
        total = len(self.active_frames)
        total += len(self.body_admit_fifo.cur) + len(self.body_admit_fifo.nxt)
        total += len(self.body_done_token_fifo.cur) + len(self.body_done_token_fifo.nxt)
        total += len(self.loop_frame_to_next_fifo.cur) + len(self.loop_frame_to_next_fifo.nxt)
        total += len(self.loop_output_bundle.cur) + len(self.loop_output_bundle.nxt)
        for frame in self.active_frames:
            if frame.child is not None:
                total += frame.child._live_contexts()
        return total

    def tick(self) -> List[str]:
        self.cycle += 1
        trial = deepcopy(self)
        trial.cycle = self.cycle
        try:
            events = trial._tick_body()
        except RuntimeError as exc:
            self.stall_count += 1
            return [f"{self.spec.name}: stall ({exc})"]
        self.__dict__.update(trial.__dict__)
        return events

    def _tick_body(self) -> List[str]:
        """Advance one cycle. Returns a list of human-readable events."""
        events: List[str] = []
        did_entry = False
        credit_next: Optional[List[int]] = None
        tag_next: Optional[List[int]] = None
        if self.credit_reg is not None:
            credit_next = [self.credit_reg.read()]
        if self.tag_reg is not None:
            tag_next = [self.tag_reg.read()]

        # 1) Progress active frames.
        for frame in list(self.active_frames):
            if frame.step_idx >= len(self.spec.body):
                self.body_done_token_fifo.push(
                    FrameToken(
                        frame_id=frame.frame_id,
                        tag=frame.tag,
                        iter_idx=frame.iter_idx,
                        payload=deepcopy(frame.payload),
                    )
                )
                self.active_frames.remove(frame)
                events.append(f"{self.spec.name}: body done {frame.tag}")
                continue

            current_step = self.spec.body[frame.step_idx]

            if not frame.started:
                if isinstance(current_step, BasicBodySpec):
                    frame.remaining = current_step.latency
                    frame.started = True
                    events.append(
                        f"{self.spec.name}: start basic step {frame.tag} step={frame.step_idx}"
                    )
                else:
                    assert isinstance(current_step, LoopSpec)
                    child_payload = deepcopy(frame.payload)
                    frame.child = LoopRuntime(current_step, child_payload, tag_prefix=f"{frame.tag}.")
                    frame.started = True
                    events.append(
                        f"{self.spec.name}: start child loop {frame.tag} step={frame.step_idx}"
                    )
                continue

            if frame.child is not None:
                frame.child.tick()
                if frame.child.is_done():
                    frame.child.commit()
                    child_outputs = list(frame.child.loop_output_bundle.cur) + list(frame.child.loop_output_bundle.nxt)
                    if child_outputs:
                        frame.payload = deepcopy(child_outputs[-1])
                    frame.child = None
                    frame.started = False
                    frame.step_idx += 1
                    events.append(
                        f"{self.spec.name}: child step done {frame.tag} step={frame.step_idx - 1}"
                    )
                continue

            if frame.remaining > 0:
                frame.remaining -= 1

            if frame.remaining == 0:
                assert isinstance(current_step, BasicBodySpec)
                frame.payload = current_step.transform(deepcopy(frame.payload), frame.iter_idx)
                frame.started = False
                frame.step_idx += 1
                events.append(
                    f"{self.spec.name}: basic step done {frame.tag} step={frame.step_idx - 1}"
                )

        # 2) Start new frames from body admission FIFO.
        while self.body_admit_fifo.has_cur() and self._can_accept_new_frame():
            frame = self.body_admit_fifo.pop()
            if credit_next is not None:
                credit_next[0] -= 1
            if tag_next is not None:
                tag_next[0] += 1
            self.active_frames.append(frame)
            events.append(f"{self.spec.name}: admit {frame.tag} iter={frame.iter_idx}")
        if self.body_admit_fifo.has_cur() and not self._can_accept_new_frame():
            self.stall_count += 1
            events.append(f"{self.spec.name}: stall body admission")

        # 3) Launch one new frame per cycle for stream pipelines.
        launch_idx: Optional[int] = None
        if (
            self.spec.is_pipeline()
            and self.spec.launch_mode == "stream"
            and self.launch_cursor_reg is not None
            and not did_entry
            and self.entry_token.empty()
        ):
            launch_idx = self.launch_cursor_reg.read()
            if launch_idx < self.spec.tripcount:
                next_frame = self._new_frame(launch_idx, self.payload_seed)
                self.body_admit_fifo.push(next_frame)
                self.launch_cursor_reg.write(launch_idx + 1)
                events.append(f"{self.spec.name}: launch {next_frame.tag} iter={launch_idx}")

        # 4) Consume body-done frames from the current cycle boundary.
        while self.body_done_token_fifo.has_cur():
            token = self.body_done_token_fifo.pop()
            self.loop_frame_to_next_fifo.push(token)
            events.append(f"{self.spec.name}: frame_to_next {token.tag}")

        # 5) next rule: decide continue / exit for completed frames.
        while self.loop_frame_to_next_fifo.has_cur():
            token = self.loop_frame_to_next_fifo.pop()
            events.append(
                f"{self.spec.name}: next decide {token.tag} iter={token.iter_idx}"
            )
            self._body_complete(token, credit_accum=credit_next)

        # 6) entry rule: only once per scope invocation.
        if self.entry_token.has_cur() and self.loop_input_bundle.has_cur():
            self.entry_token.pop()
            seed = self.loop_input_bundle.pop()
            initial = self._new_frame(0, seed)
            if self.body_admit_fifo.occupancy() >= self.body_admit_fifo.depth:
                self.stall_count += 1
                events.append(f"{self.spec.name}: stall entry admission")
                self.entry_token.push(1)
                self.loop_input_bundle.push(seed)
            else:
                self.body_admit_fifo.push(initial)
                events.append(f"{self.spec.name}: entry seed {initial.tag}")
            if not self.spec.is_pipeline():
                self.busy_reg.write(True)
                self.loop_state_reg.write(deepcopy(seed))
            elif self.spec.launch_mode == "stream" and self.launch_cursor_reg is not None:
                self.launch_cursor_reg.write(1)
            did_entry = True

        if self.credit_reg is not None and credit_next is not None and credit_next[0] != self.credit_reg.cur:
            self.credit_reg.write(credit_next[0])
        if self.tag_reg is not None and tag_next is not None and tag_next[0] != self.tag_reg.cur:
            self.tag_reg.write(tag_next[0])

        self.max_active_frames = max(self.max_active_frames, len(self.active_frames))
        self.max_live_contexts = max(self.max_live_contexts, self._live_contexts())
        return events

    def commit(self) -> None:
        """Commit next-cycle queue contents for this runtime and all children."""
        self.entry_token.commit()
        self.loop_input_bundle.commit()
        self.body_admit_fifo.commit()
        self.body_done_token_fifo.commit()
        self.loop_frame_to_next_fifo.commit()
        self.loop_output_bundle.commit()
        self.busy_reg.commit()
        self.loop_state_reg.commit()
        if self.credit_reg is not None:
            self.credit_reg.commit()
        if self.tag_reg is not None:
            self.tag_reg.commit()
        if self.launch_cursor_reg is not None:
            self.launch_cursor_reg.commit()
        for frame in self.active_frames:
            if frame.child is not None:
                frame.child.commit()

    def is_done(self) -> bool:
        queues_empty = (
            self.entry_token.empty()
            and self.loop_input_bundle.empty()
            and self.body_admit_fifo.empty()
            and self.body_done_token_fifo.empty()
            and self.loop_frame_to_next_fifo.empty()
        )
        children_done = all(frame.child is None or frame.child.is_done() for frame in self.active_frames)
        if self.spec.is_pipeline() and self.spec.launch_mode == "stream" and self.launch_cursor_reg is not None:
            launched_all = self.launch_cursor_reg.cur >= self.spec.tripcount
        else:
            launched_all = True
        return queues_empty and not self.active_frames and children_done and launched_all

    def render(self, indent: int = 0) -> List[str]:
        pad = " " * indent
        lines = [
            f"{pad}{self.spec.name}: cycle={self.cycle} active={len(self.active_frames)} "
            f"done={self.is_done()} busy={self.busy_reg.cur} "
            f"state={self.loop_state_reg.cur} "
            f"credit={None if self.credit_reg is None else self.credit_reg.cur} "
            f"tag={None if self.tag_reg is None else self.tag_reg.cur} "
            f"launch={None if self.launch_cursor_reg is None else self.launch_cursor_reg.cur} "
            f"out={len(self.loop_output_bundle.cur)} stall={self.stall_count} "
            f"live_ctx={self._live_contexts()} max_live={self.max_live_contexts}",
        ]
        for frame in self.active_frames:
            child_desc = ""
            if frame.child is not None:
                child_desc = f" child=[{frame.child.spec.name} active={len(frame.child.active_frames)} done={frame.child.is_done()}]"
            lines.append(
                f"{pad}  frame {frame.tag} iter={frame.iter_idx} rem={frame.remaining}{child_desc}"
            )
        for frame in self.active_frames:
            if frame.child is not None:
                lines.extend(frame.child.render(indent + 4))
        return lines

    def resource_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "busy_reg": self.busy_reg.stats(),
            "loop_state_reg": self.loop_state_reg.stats(),
            "entry_token": self.entry_token.stats(),
            "loop_input_bundle": self.loop_input_bundle.stats(),
            "body_admit_fifo": self.body_admit_fifo.stats(),
            "body_done_token_fifo": self.body_done_token_fifo.stats(),
            "loop_frame_to_next_fifo": self.loop_frame_to_next_fifo.stats(),
            "loop_output_bundle": self.loop_output_bundle.stats(),
        }
        if self.credit_reg is not None:
            stats["credit_reg"] = self.credit_reg.stats()
        if self.tag_reg is not None:
            stats["tag_reg"] = self.tag_reg.stats()
        if self.launch_cursor_reg is not None:
            stats["launch_cursor_reg"] = self.launch_cursor_reg.stats()
        stats["stall_count"] = self.stall_count
        if self.active_frames:
            stats["children"] = [frame.child.resource_stats() for frame in self.active_frames if frame.child is not None]
        return stats


@dataclass
class ProgramRuntime:
    """A sequential composition of loop scopes."""

    steps: List[LoopRuntime]
    step_idx: int = 0
    current: Optional[LoopRuntime] = None
    completed_payload: Optional[Payload] = None
    cycle: int = 0
    max_live_contexts: int = 0
    max_active_frames: int = 0

    def tick(self) -> List[str]:
        events: List[str] = []
        self.cycle += 1
        if self.current is None and self.step_idx < len(self.steps):
            self.current = self.steps[self.step_idx]
            events.append(f"program: launch {self.current.spec.name}")
            if self.current.entry_token.empty() and self.current.loop_input_bundle.empty():
                seed = self.completed_payload if self.completed_payload is not None else deepcopy(self.current.payload_seed)
                self.current.prime(seed)
                events.append(f"program: prime {self.current.spec.name}")

        if self.current is not None:
            events.extend(self.current.tick())
            if self.current.is_done():
                self.current.commit()
                current_outputs = list(self.current.loop_output_bundle.cur) + list(self.current.loop_output_bundle.nxt)
                if current_outputs:
                    self.completed_payload = deepcopy(current_outputs[-1])
                events.append(f"program: complete {self.current.spec.name}")
                self.step_idx += 1
                self.current = None
                if self.step_idx < len(self.steps):
                    next_step = self.steps[self.step_idx]
                    if next_step.entry_token.empty() and next_step.loop_input_bundle.empty():
                        if self.completed_payload is not None:
                            next_step.prime(self.completed_payload)
                        else:
                            next_step.prime(deepcopy(next_step.payload_seed))
                    events.append(f"program: armed {next_step.spec.name}")
        self.max_live_contexts = max(
            self.max_live_contexts,
            sum(step._live_contexts() for step in self.steps),
        )
        self.max_active_frames = max(
            self.max_active_frames,
            sum(len(step.active_frames) for step in self.steps),
        )
        return events

    def commit(self) -> None:
        for step in self.steps:
            step.commit()

    def is_done(self) -> bool:
        return self.step_idx >= len(self.steps) and self.current is None

    def render(self, indent: int = 0) -> List[str]:
        pad = " " * indent
        lines = [
            f"{pad}program: cycle={self.cycle} step_idx={self.step_idx} "
            f"done={self.is_done()} max_live={self.max_live_contexts}",
        ]
        if self.current is not None:
            lines.append(f"{pad}  current={self.current.spec.name}")
            lines.extend(self.current.render(indent + 4))
        return lines

    def resource_stats(self) -> Dict[str, Any]:
        return {
            "steps": [step.resource_stats() for step in self.steps],
            "current_step": None if self.current is None else self.current.spec.name,
        }


def run_simulation(root: Any, max_cycles: int = 256, trace: bool = False) -> Dict[str, Any]:
    events_by_cycle: List[List[str]] = []
    for cycle in range(max_cycles):
        events = root.tick()
        root.commit()
        events_by_cycle.append(events)
        if trace:
            print(f"cycle {cycle:04d}")
            for line in root.render(indent=2):
                print(line)
            for event in events:
                print(f"  - {event}")
            print()
        if root.is_done():
            break
    outputs: List[Payload] = []
    if hasattr(root, "completed_payload"):
        if getattr(root, "completed_payload") is not None:
            outputs = [deepcopy(getattr(root, "completed_payload"))]
    elif hasattr(root, "loop_output_bundle"):
        outputs = list(root.loop_output_bundle.cur) + list(root.loop_output_bundle.nxt)
    return {
        "cycles": root.cycle,
        "done": root.is_done(),
        "outputs": outputs,
        "max_active_frames": root.max_active_frames,
        "max_live_contexts": root.max_live_contexts,
        "resource_stats": root.resource_stats(),
        "events_by_cycle": events_by_cycle,
    }


def make_simple_pipeline_example(tripcount: int = 8, latency: int = 3) -> LoopRuntime:
    def transform(payload: Payload, iter_idx: int) -> Payload:
        out = dict(payload)
        out["acc"] = out.get("acc", 0) + 1
        out["last_iter"] = iter_idx
        return out

    spec = LoopSpec(
        name="simple_pipeline",
        tripcount=tripcount,
        mode="pipeline",
        launch_mode="carry",
        body=[BasicBodySpec("simple_body", latency=latency, transform=transform)],
    )
    return LoopRuntime(spec, {"acc": 0})


def make_pure_non_pipeline_example(tripcount: int = 8, latency: int = 3) -> LoopRuntime:
    def transform(payload: Payload, iter_idx: int) -> Payload:
        out = dict(payload)
        out["acc"] = out.get("acc", 0) + 1
        out["last_iter"] = iter_idx
        return out

    spec = LoopSpec(
        name="pure_non_pipeline",
        tripcount=tripcount,
        mode="non_pipeline",
        launch_mode="carry",
        body=[BasicBodySpec("non_pipeline_body", latency=latency, transform=transform)],
    )
    return LoopRuntime(spec, {"acc": 0})


def make_nested_hierarchical_example(
    outer_tripcount: int = 4,
    inner_tripcount: int = 5,
    inner_latency: int = 2,
    outer_latency: int = 1,
) -> LoopRuntime:
    def inner_transform(payload: Payload, iter_idx: int) -> Payload:
        out = dict(payload)
        out["inner_acc"] = out.get("inner_acc", 0) + 1
        out["inner_last_iter"] = iter_idx
        return out

    inner = LoopSpec(
        name="inner_loop",
        tripcount=inner_tripcount,
        mode="pipeline",
        launch_mode="carry",
        body=[BasicBodySpec("inner_body", latency=inner_latency, transform=inner_transform)],
    )
    outer = LoopSpec(
        name="outer_loop",
        tripcount=outer_tripcount,
        mode="pipeline",
        launch_mode="carry",
        body=[inner],
    )
    root = LoopRuntime(outer, {"outer_acc": 0, "inner_acc": 0})
    return root


def make_nested_flattened_example(
    outer_tripcount: int = 4,
    inner_tripcount: int = 5,
    body_latency: int = 2,
) -> LoopRuntime:
    total_tripcount = outer_tripcount * inner_tripcount

    def transform(payload: Payload, iter_idx: int) -> Payload:
        out = dict(payload)
        out["flat_acc"] = out.get("flat_acc", 0) + 1
        out["flat_last_iter"] = iter_idx
        return out

    spec = LoopSpec(
        name="flattened_loop",
        tripcount=total_tripcount,
        mode="pipeline",
        launch_mode="carry",
        body=[BasicBodySpec("flat_body", latency=body_latency, transform=transform)],
    )
    return LoopRuntime(spec, {"flat_acc": 0})


def make_multi_basic_block_nested_example(
    outer_tripcount: int = 3,
    inner_tripcount: int = 4,
) -> LoopRuntime:
    def outer_pre(payload: Payload, iter_idx: int) -> Payload:
        out = dict(payload)
        out["outer_pre"] = out.get("outer_pre", 0) + 1
        out["outer_pre_last_iter"] = iter_idx
        return out

    def inner_a(payload: Payload, iter_idx: int) -> Payload:
        out = dict(payload)
        out["inner_a"] = out.get("inner_a", 0) + 2
        out["inner_a_last_iter"] = iter_idx
        return out

    def inner_b(payload: Payload, iter_idx: int) -> Payload:
        out = dict(payload)
        out["inner_b"] = out.get("inner_b", 0) + 3
        out["inner_b_last_iter"] = iter_idx
        return out

    def outer_post(payload: Payload, iter_idx: int) -> Payload:
        out = dict(payload)
        out["outer_post"] = out.get("outer_post", 0) + 1
        out["outer_post_last_iter"] = iter_idx
        return out

    inner = LoopSpec(
        name="nested_inner_loop",
        tripcount=inner_tripcount,
        mode="pipeline",
        launch_mode="carry",
        body=[
            BasicBodySpec("inner_block_a", latency=1, transform=inner_a),
            BasicBodySpec("inner_block_b", latency=2, transform=inner_b),
        ],
    )
    outer = LoopSpec(
        name="nested_outer_loop",
        tripcount=outer_tripcount,
        mode="pipeline",
        launch_mode="carry",
        body=[
            BasicBodySpec("outer_block_pre", latency=1, transform=outer_pre),
            inner,
            BasicBodySpec("outer_block_post", latency=1, transform=outer_post),
        ],
    )
    return LoopRuntime(
        outer,
        {
            "outer_pre": 0,
            "inner_a": 0,
            "inner_b": 0,
            "outer_post": 0,
        },
    )


def make_sequential_two_loop_example() -> ProgramRuntime:
    def pipeline_transform(payload: Payload, iter_idx: int) -> Payload:
        out = dict(payload)
        out["pipe_acc"] = out.get("pipe_acc", 0) + 1
        out["pipe_last_iter"] = iter_idx
        return out

    def serial_transform(payload: Payload, iter_idx: int) -> Payload:
        out = dict(payload)
        out["serial_acc"] = out.get("serial_acc", 0) + 1
        out["serial_last_iter"] = iter_idx
        return out

    first = LoopRuntime(
        LoopSpec(
            name="first_pipeline_loop",
            tripcount=6,
            mode="pipeline",
            launch_mode="carry",
            body=[BasicBodySpec("first_pipe_body", latency=2, transform=pipeline_transform)],
        ),
        {"pipe_acc": 0},
        auto_prime=False,
    )
    second = LoopRuntime(
        LoopSpec(
            name="second_non_pipeline_loop",
            tripcount=5,
            mode="non_pipeline",
            launch_mode="carry",
            body=[BasicBodySpec("second_serial_body", latency=3, transform=serial_transform)],
        ),
        {"serial_acc": 0},
        auto_prime=False,
    )
    return ProgramRuntime([first, second])


def make_stream_pipeline_example(tripcount: int = 8, latency: int = 3) -> LoopRuntime:
    def transform(payload: Payload, iter_idx: int) -> Payload:
        out = dict(payload)
        out["sample"] = iter_idx
        out["done"] = True
        return out

    spec = LoopSpec(
        name="stream_pipeline",
        tripcount=tripcount,
        mode="pipeline",
        launch_mode="stream",
        body=[BasicBodySpec("stream_body", latency=latency, transform=transform)],
    )
    return LoopRuntime(spec, {"sample": -1, "done": False})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cycle-accurate APS/CMT2 loop simulator")
    p.add_argument(
        "--example",
        choices=[
            "simple-pipeline",
            "stream-pipeline",
            "pure-non-pipeline",
            "nested-hierarchical",
            "nested-flattened",
            "multi-basic-block-nested",
            "sequential-two-loop",
        ],
        default="simple-pipeline",
        help="Built-in scenario to run",
    )
    p.add_argument("--cycles", type=int, default=128, help="Maximum number of cycles")
    p.add_argument("--trace", action="store_true", help="Print per-cycle trace")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.example == "simple-pipeline":
        root = make_simple_pipeline_example()
    elif args.example == "stream-pipeline":
        root = make_stream_pipeline_example()
    elif args.example == "pure-non-pipeline":
        root = make_pure_non_pipeline_example()
    elif args.example == "nested-hierarchical":
        root = make_nested_hierarchical_example()
    elif args.example == "nested-flattened":
        root = make_nested_flattened_example()
    elif args.example == "multi-basic-block-nested":
        root = make_multi_basic_block_nested_example()
    elif args.example == "sequential-two-loop":
        root = make_sequential_two_loop_example()
    else:
        raise AssertionError(args.example)

    result = run_simulation(root, max_cycles=args.cycles, trace=args.trace)
    print(
        {
            "example": args.example,
            "cycles": result["cycles"],
            "done": result["done"],
            "max_active_frames": result["max_active_frames"],
            "max_live_contexts": result["max_live_contexts"],
            "outputs": result["outputs"],
            "resource_stats": result["resource_stats"],
        }
    )
    return 0 if result["done"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
