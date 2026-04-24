"""Continuous-batching scheduler.

Manages the lifecycle of inference requests through three states:

  WAITING  -> prefill pending (prompt not yet processed)
  RUNNING  -> actively generating (in decode batch)
  FINISHED -> generation complete (EOS or max_tokens reached)

Each scheduling step:
  1. Admit WAITING requests into RUNNING (allocate KV blocks).
  2. Return the RUNNING set for the engine to execute.
  3. After execution, the engine reports finished sequence ids;
     the scheduler frees their KV blocks.

The scheduler also enforces memory limits by refusing to admit new
requests when the block allocator cannot satisfy the prompt's block
requirement.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional

from .debug import tracer
from .kv_cache import KVCacheManager


class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: List[int]
    generated_token_ids: List[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    max_tokens: int = 256


@dataclass
class SchedulerOutput:
    """What the scheduler hands to the engine for one step."""
    prefill_seqs: List[Sequence]       # sequences needing prefill this step
    decode_seqs: List[Sequence]        # sequences doing decode this step


class Scheduler:
    def __init__(
        self,
        kv_manager: KVCacheManager,
        max_num_seqs: int = 256,
    ):
        self.kv_manager = kv_manager
        self.max_num_seqs = max_num_seqs

        self.waiting: Deque[Sequence] = deque()
        self.running: Dict[int, Sequence] = {}
        self.finished: Dict[int, Sequence] = {}
        self._next_id = 0

    def add_request(self, prompt_token_ids: List[int], max_tokens: int = 256) -> int:
        seq_id = self._next_id
        self._next_id += 1
        seq = Sequence(
            seq_id=seq_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
        )
        self.waiting.append(seq)
        tracer.on_add_request(seq_id, prompt_token_ids, max_tokens)
        return seq_id

    def schedule(self) -> SchedulerOutput:
        prefill: List[Sequence] = []
        decode: List[Sequence] = list(self.running.values())

        while self.waiting and len(self.running) + len(prefill) < self.max_num_seqs:
            seq = self.waiting[0]
            num_blocks_needed = (
                (len(seq.prompt_token_ids) + self.kv_manager.block_size - 1)
                // self.kv_manager.block_size
            )
            if self.kv_manager.allocator.num_free < num_blocks_needed:
                break
            seq = self.waiting.popleft()
            self.kv_manager.allocate_seq(seq.seq_id, len(seq.prompt_token_ids))
            seq.status = SeqStatus.RUNNING
            prefill.append(seq)

        return SchedulerOutput(prefill_seqs=prefill, decode_seqs=decode)

    def after_step(
        self,
        prefill_seqs: List[Sequence],
        finished_ids: List[int],
    ) -> None:
        """Called by the engine after executing a step."""
        for seq in prefill_seqs:
            self.running[seq.seq_id] = seq

        for sid in finished_ids:
            seq = self.running.pop(sid, None)
            if seq is not None:
                seq.status = SeqStatus.FINISHED
                self.finished[sid] = seq
                self.kv_manager.free_seq(sid)

    @property
    def has_pending(self) -> bool:
        return bool(self.waiting) or bool(self.running)
