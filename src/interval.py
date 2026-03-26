from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .utils import K_EPS

@dataclass
class Interval:
    lb: float = 1.0
    ub: float = 0.0

    @staticmethod
    def empty():
        return Interval()

    def is_empty(self) -> bool:
        return self.lb > self.ub

    def length(self) -> float:
        return 0.0 if self.is_empty() else self.ub - self.lb

    def contains(self, x: float) -> bool:
        if self.is_empty():
            return False
        return self.lb - K_EPS <= x <= self.ub + K_EPS

    def contains_interval(self, other: "Interval") -> bool:
        if self.is_empty() or other.is_empty():
            return False
        return (self.lb <= other.lb + K_EPS and
                self.ub >= other.ub - K_EPS)

    def overlaps(self, other: "Interval") -> bool:
        if self.is_empty() or other.is_empty():
            return False
        return not (self.ub + K_EPS < other.lb or
                    other.ub + K_EPS < self.lb)

    def intersect(self, other: "Interval") -> "Interval":
        if self.is_empty() or other.is_empty():
            return Interval.empty()
        return Interval(max(self.lb, other.lb),
                        min(self.ub, other.ub))

    def sample_uniform_by_n(self, n: int, include_endpoints: bool = True) -> list[float]:
        if self.is_empty() or n == 0:
            return []

        if include_endpoints:
            if n == 1:
                return [(self.lb + self.ub) / 2.0]
            step = (self.ub - self.lb) / (n - 1)
            return [self.lb + i * step for i in range(n)]
        else:
            step = (self.ub - self.lb) / (n + 1)
            return [self.lb + (i + 1) * step for i in range(n)]

    def sample_uniform_by_step(self, step: float, include_endpoints: bool = True) -> list[float]:
        if self.is_empty() or step <= 0:
            return []

        samples = []
        x = self.lb

        if include_endpoints:
            while x <= self.ub + K_EPS:
                samples.append(x)
                x += step
        else:
            x += step
            while x < self.ub - K_EPS:
                samples.append(x)
                x += step

        return samples
    
    def __repr__(self):
        return f"[{self.lb:.6f}, {self.ub:.6f}]"


@dataclass
class Intervals:
    intervals: list[Interval] = field(default_factory=list)

    def __post_init__(self):
        self.intervals = self._normalize(self.intervals)

    @staticmethod
    def empty() -> "Intervals":
        return Intervals()

    @staticmethod
    def _normalize(intervals: Iterable[Interval]) -> list[Interval]:
        valid_intervals = [Interval(interval.lb, interval.ub) for interval in intervals if not interval.is_empty()]
        if not valid_intervals:
            return []

        valid_intervals.sort(key=lambda interval: (interval.lb, interval.ub))

        merged = [valid_intervals[0]]
        for current in valid_intervals[1:]:
            last = merged[-1]
            if current.lb <= last.ub + K_EPS:
                last.ub = max(last.ub, current.ub)
            else:
                merged.append(Interval(current.lb, current.ub))
        return merged

    def copy(self) -> "Intervals":
        return Intervals(self.intervals)

    def is_empty(self) -> bool:
        return not self.intervals

    def add(self, other: Interval | "Intervals") -> "Intervals":
        if isinstance(other, Interval):
            self.intervals = self._normalize([*self.intervals, other])
            return self

        self.intervals = self._normalize([*self.intervals, *other.intervals])
        return self

    def contains(self, x: float) -> bool:
        return any(interval.contains(x) for interval in self.intervals)

    def contains_interval(self, other: Interval) -> bool:
        return any(interval.contains_interval(other) for interval in self.intervals)

    def intersect(self, other: Interval | "Intervals") -> "Intervals":
        other_intervals = [other] if isinstance(other, Interval) else other.intervals
        if not self.intervals or not other_intervals:
            return Intervals.empty()

        result: list[Interval] = []
        i = 0
        j = 0

        while i < len(self.intervals) and j < len(other_intervals):
            left = self.intervals[i]
            right = other_intervals[j]

            intersection = left.intersect(right)
            if not intersection.is_empty():
                result.append(intersection)

            if left.ub < right.ub - K_EPS:
                i += 1
            else:
                j += 1

        return Intervals(result)

    def union(self, other: Interval | "Intervals") -> "Intervals":
        result = self.copy()
        return result.add(other)

    def intersection(self, other: Interval | "Intervals") -> "Intervals":
        return self.intersect(other)

    def difference(self, other: Interval | "Intervals") -> "Intervals":
        other_intervals = [other] if isinstance(other, Interval) else other.intervals
        if not self.intervals:
            return Intervals.empty()
        if not other_intervals:
            return self.copy()

        result: list[Interval] = []
        j = 0

        for current in self.intervals:
            start = current.lb

            while j < len(other_intervals) and other_intervals[j].ub < current.lb - K_EPS:
                j += 1

            k = j
            while k < len(other_intervals) and other_intervals[k].lb <= current.ub + K_EPS:
                blocker = other_intervals[k]

                if blocker.lb > start + K_EPS:
                    result.append(Interval(start, min(blocker.lb - K_EPS, current.ub)))

                if blocker.ub >= current.ub - K_EPS:
                    start = current.ub
                    break

                start = max(start, blocker.ub + K_EPS)
                k += 1
            else:
                if start <= current.ub + K_EPS:
                    result.append(Interval(start, current.ub))
                continue

            if start < current.ub - K_EPS:
                result.append(Interval(start, current.ub))

        return Intervals(result)

    def complement(self, universe: Interval) -> "Intervals":
        if universe.is_empty():
            return Intervals.empty()
        return Intervals([universe]).difference(self)

    def __add__(self, other: Interval | "Intervals") -> "Intervals":
        return self.union(other)

    def __or__(self, other: Interval | "Intervals") -> "Intervals":
        return self.union(other)

    def __and__(self, other: Interval | "Intervals") -> "Intervals":
        return self.intersect(other)

    def __sub__(self, other: Interval | "Intervals") -> "Intervals":
        return self.difference(other)

    def to_list(self) -> list[Interval]:
        return [Interval(interval.lb, interval.ub) for interval in self.intervals]

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __len__(self) -> int:
        return len(self.intervals)

    def __iter__(self) -> Iterator[Interval]:
        return iter(self.intervals)

    def __getitem__(self, index: int) -> Interval:
        return self.intervals[index]

    def __repr__(self):
        return "{" + ", ".join(repr(interval) for interval in self.intervals) + "}"
