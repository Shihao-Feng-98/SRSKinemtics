from utils import K_EPS_SMALL
from dataclasses import dataclass

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
        return self.lb - K_EPS_SMALL <= x <= self.ub + K_EPS_SMALL

    def contains_interval(self, other: "Interval") -> bool:
        if self.is_empty() or other.is_empty():
            return False
        return (self.lb <= other.lb + K_EPS_SMALL and
                self.ub >= other.ub - K_EPS_SMALL)

    def overlaps(self, other: "Interval") -> bool:
        if self.is_empty() or other.is_empty():
            return False
        return not (self.ub + K_EPS_SMALL < other.lb or
                    other.ub + K_EPS_SMALL < self.lb)

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
            while x <= self.ub + K_EPS_SMALL:
                samples.append(x)
                x += step
        else:
            x += step
            while x < self.ub - K_EPS_SMALL:
                samples.append(x)
                x += step

        return samples
    
    def __repr__(self):
        return f"[{self.lb:.6f}, {self.ub:.6f}]"

        
