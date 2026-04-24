"""Debug tracer for KV cache data flow.

Belongs to the `debug/kv-cache-trace` branch. The tracer is **off by default**
(no overhead) and turned on by either:

  - Environment variable: `NDS_KV_DEBUG=1`
  - Programmatic call:    `from nano_dist_spec.debug import tracer; tracer.enable()`

Why a tracer rather than scattered `print`s? KV cache state lives in many
places (allocator free-list, manager block_tables, GPU tensors keyed by
`slot_mapping`). A single tracer with semantic methods (`on_allocate_seq`,
`on_kv_write`, ...) lets every call site stay one line, while the formatting,
verbosity gating, and indentation logic live here.

Mid-level verbosity policy:
  - Block-allocator: log alloc/free with current pool size.
  - Manager: log allocate_seq / append_slots / rollback / free_seq with the
    block table snapshot before/after.
  - compute_slot_mapping: log the (pos -> (blk, off) -> slot) folded table.
  - KV write: log only for layer 0 to avoid num_layers x num_steps spam.
  - Engine + speculative: log step boundaries and per-phase events.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, List, Optional, Sequence


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


class KVTracer:
    """Singleton-style tracer with cheap no-op gating.

    Each `on_*` method first checks `self.enabled`; when off the call is a
    bool test plus a function return — no string formatting cost.
    """

    def __init__(self) -> None:
        self.enabled: bool = _env_truthy("NDS_KV_DEBUG")
        self._stream = sys.stderr
        self._step_idx = 0

    # ------------------------------------------------------------------
    # Toggles
    # ------------------------------------------------------------------

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _PREFIX = "[KV-TRACE]"

    def _emit(self, category: str, msg: str) -> None:
        # Categories padded to 12 cols so columns align across event types.
        print(f"{self._PREFIX} {category:<12} {msg}", file=self._stream, flush=True)

    def _emit_continuation(self, msg: str) -> None:
        # Indent under the previous line's category column for readability.
        pad = " " * (len(self._PREFIX) + 1 + 12 + 1)
        print(f"{pad}{msg}", file=self._stream, flush=True)

    @staticmethod
    def _fmt_list(xs: Iterable[int], max_items: int = 8) -> str:
        xs = list(xs)
        if len(xs) <= max_items:
            return "[" + ",".join(str(x) for x in xs) + "]"
        head = ",".join(str(x) for x in xs[: max_items // 2])
        tail = ",".join(str(x) for x in xs[-max_items // 2 :])
        return f"[{head}, ... ,{tail}]  (len={len(xs)})"

    @staticmethod
    def _fmt_slot_table(
        start_pos: int,
        slots: Sequence[int],
        block_table: Sequence[int],
        block_size: int,
        max_rows: int = 8,
    ) -> List[str]:
        """Format a (pos -> (blk, off) -> slot) table, folding if too long."""
        n = len(slots)
        if n == 0:
            return ["(empty)"]

        def row(i: int) -> str:
            pos = start_pos + i
            blk_idx = pos // block_size
            blk_off = pos % block_size
            phys = block_table[blk_idx]
            return f"{pos:>4}->({blk_idx:>2},{blk_off:>2})[blk={phys:>3}]->slot={slots[i]:>5}"

        if n <= max_rows:
            indices = range(n)
        else:
            half = max_rows // 2
            indices = list(range(half)) + [-1] + list(range(n - half, n))

        lines = ["pos->(blk,off)[blk=phys]->slot:"]
        for i in indices:
            if i == -1:
                lines.append("    ...")
            else:
                lines.append("    " + row(i))
        return lines

    # ------------------------------------------------------------------
    # BlockAllocator
    # ------------------------------------------------------------------

    def on_allocate(self, block_id: int, num_free: int, num_total: int) -> None:
        if not self.enabled:
            return
        self._emit("alloc", f"+block={block_id:<3}  free_pool={num_free}/{num_total}")

    def on_free(self, block_id: int, num_free: int, num_total: int) -> None:
        if not self.enabled:
            return
        self._emit("free", f"-block={block_id:<3}  free_pool={num_free}/{num_total}")

    # ------------------------------------------------------------------
    # KVCacheManager
    # ------------------------------------------------------------------

    def on_allocate_seq(
        self,
        seq_id: int,
        prompt_len: int,
        blocks: Sequence[int],
        free_after: int,
        num_total: int,
    ) -> None:
        if not self.enabled:
            return
        self._emit(
            "allocate_seq",
            f"seq={seq_id} prompt_len={prompt_len} "
            f"blocks={self._fmt_list(blocks)} ctx={prompt_len} "
            f"free_pool={free_after}/{num_total}",
        )

    def on_append_slots(
        self,
        seq_id: int,
        old_len: int,
        new_len: int,
        block_table: Sequence[int],
        block_size: int,
        new_block_id: Optional[int],
        free_after: int,
        num_total: int,
    ) -> None:
        if not self.enabled:
            return
        if new_block_id is None:
            blk_idx = (new_len - 1) // block_size
            blk_off = (new_len - 1) % block_size
            phys = block_table[blk_idx]
            detail = f"(no new block; blk={phys} off={blk_off})"
        else:
            detail = (
                f"(NEW block={new_block_id} allocated; "
                f"free_pool={free_after}/{num_total})"
            )
        self._emit(
            "append_slots",
            f"seq={seq_id} ctx {old_len}->{new_len} {detail}",
        )

    def on_rollback(
        self,
        seq_id: int,
        old_len: int,
        new_len: int,
        freed_blocks: Sequence[int],
        block_table_after: Sequence[int],
        free_after: int,
        num_total: int,
    ) -> None:
        if not self.enabled:
            return
        self._emit(
            "rollback",
            f"seq={seq_id} ctx {old_len}->{new_len} "
            f"freed={self._fmt_list(freed_blocks)} "
            f"table={self._fmt_list(block_table_after)} "
            f"free_pool={free_after}/{num_total}",
        )

    def on_free_seq(
        self,
        seq_id: int,
        freed_blocks: Sequence[int],
        free_after: int,
        num_total: int,
    ) -> None:
        if not self.enabled:
            return
        self._emit(
            "free_seq",
            f"seq={seq_id} freed={self._fmt_list(freed_blocks)} "
            f"free_pool={free_after}/{num_total}",
        )

    def on_slot_mapping(
        self,
        seq_id: int,
        start_pos: int,
        num_tokens: int,
        slots: Sequence[int],
        block_table: Sequence[int],
        block_size: int,
    ) -> None:
        if not self.enabled:
            return
        end = start_pos + num_tokens
        self._emit(
            "slot_map",
            f"seq={seq_id} range=[{start_pos},{end}) "
            f"table={self._fmt_list(block_table)}",
        )
        for line in self._fmt_slot_table(start_pos, slots, block_table, block_size):
            self._emit_continuation(line)

    # ------------------------------------------------------------------
    # Model attention KV write (only called for layer 0)
    # ------------------------------------------------------------------

    def on_kv_write(self, layer_idx: int, slots: Sequence[int]) -> None:
        if not self.enabled:
            return
        self._emit(
            "kv_write",
            f"layer={layer_idx} n={len(slots)} slots={self._fmt_list(slots)}",
        )

    # ------------------------------------------------------------------
    # Engine / speculative step boundaries
    # ------------------------------------------------------------------

    def on_add_request(
        self, seq_id: int, prompt_token_ids: Sequence[int], max_tokens: int,
    ) -> None:
        if not self.enabled:
            return
        self._emit(
            "add_request",
            f"seq={seq_id} max_tokens={max_tokens} "
            f"prompt_len={len(prompt_token_ids)} "
            f"token_ids={self._fmt_list(prompt_token_ids, max_items=16)}",
        )

    def on_step(self, kind: str, **kwargs) -> None:
        if not self.enabled:
            return
        self._step_idx += 1
        body = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self._emit("STEP", f"#{self._step_idx} {kind:<7} {body}")

    def on_spec_event(self, event: str, **kwargs) -> None:
        if not self.enabled:
            return
        body = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self._emit("SPEC", f"{event:<13} {body}")


tracer = KVTracer()
