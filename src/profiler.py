import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional


class PipelineProfiler:
    """Lean profiler for pipeline stages - tracks compute, comm, and bubbles."""

    def __init__(self, rank: int, enabled: bool = True):
        self.rank = rank
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.active_timers: Dict[str, float] = {}
        self.stage_start: Optional[float] = None

    @contextmanager
    def time_block(self, name: str):
        """Context manager to time a code block."""
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[name].append(elapsed)

    def start_stage(self, stage_name: str):
        """Mark the start of a pipeline stage (e.g., 'forward', 'backward')."""
        if not self.enabled:
            return
        self.stage_start = time.perf_counter()
        self.active_timers[stage_name] = self.stage_start

    def end_stage(self, stage_name: str):
        """Mark the end of a pipeline stage."""
        if not self.enabled or self.stage_start is None:
            return
        elapsed = time.perf_counter() - self.stage_start
        self.timings[f"stage_{stage_name}"].append(elapsed)
        self.stage_start = None

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated statistics: mean, total, count."""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "count": len(times),
                    "min": min(times),
                    "max": max(times),
                }
        return stats

    def print_summary(self):
        """Print a summary of timings and utilization."""
        if not self.enabled:
            return

        stats = self.get_stats()
        print(f"\n--- Pipeline Profiler (Rank {self.rank}) ---")

        # Use stage_step as the actual wall-clock time per step
        if "stage_step" not in stats:
            print("No stage_step timings found")
            return

        step_time = stats["stage_step"]["total"]  # Total wall-clock for all steps
        num_steps = stats["stage_step"]["count"]
        avg_step_time = stats["stage_step"]["mean"]

        # Compute utilization (active compute + comm within steps)
        compute_time = sum(stats[k]["total"] for k in stats.keys() if "compute" in k)
        comm_time = sum(
            stats[k]["total"] for k in stats.keys() if "send" in k or "recv" in k
        )

        # Bubbles = total wall-clock time - active work time
        bubble_time = step_time - compute_time - comm_time

        if step_time > 0:
            compute_pct = (compute_time / step_time) * 100
            comm_pct = (comm_time / step_time) * 100
            bubble_pct = (bubble_time / step_time) * 100

            print(f"Steps: {num_steps} | Avg step time: {avg_step_time:.4f}s")
            print(f"Compute: {compute_time:.4f}s ({compute_pct:.1f}%)")
            print(f"Comm:    {comm_time:.4f}s ({comm_pct:.1f}%)")
            print(f"Bubbles: {bubble_time:.4f}s ({bubble_pct:.1f}%)")
            print(f"Total:   {step_time:.4f}s")

        # Detailed breakdown
        for name, stat in sorted(stats.items()):
            if name != "stage_step":  # Already shown above
                print(f"  {name:20s}: {stat['mean']:.4f}s avg ({stat['count']} calls)")

    def reset(self):
        """Reset all timings."""
        self.timings.clear()
        self.active_timers.clear()
