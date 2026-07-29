"""Mutable, fixed-capacity collision caches with explicit generations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from curobo_metal.ops.world_collision import Mesh, VoxelGrid

T = TypeVar("T")


class CacheCapacityError(RuntimeError):
    pass


class _SlotCache(Generic[T]):
    def __init__(self, environments: int, capacity: int, kind: str) -> None:
        self.environments, self.capacity, self.kind = environments, capacity, kind
        self._slots: list[list[T | None]] = [
            [None for _ in range(capacity)] for _ in range(environments)
        ]
        self._active = torch.zeros((environments, capacity), dtype=torch.bool)
        self.generation = 0

    @property
    def active(self) -> torch.Tensor:
        return self._active.clone()

    def _check(self, environment: int, slot: int) -> None:
        if not 0 <= environment < self.environments:
            raise IndexError(f"environment index out of range: {environment}")
        if not 0 <= slot < self.capacity:
            raise IndexError(f"{self.kind} cache slot out of range: {slot}")

    def update(self, environment: int, slot: int, value: T, *, active: bool = True) -> None:
        self._check(environment, slot)
        self._slots[environment][slot] = value
        self._active[environment, slot] = active
        self.generation += 1

    def append(self, environment: int, value: T, *, active: bool = True) -> int:
        if not 0 <= environment < self.environments:
            raise IndexError(f"environment index out of range: {environment}")
        for slot, current in enumerate(self._slots[environment]):
            if current is None:
                self.update(environment, slot, value, active=active)
                return slot
        raise CacheCapacityError(f"{self.kind} cache capacity {self.capacity} exceeded")

    def set_active(self, environment: int, slot: int, active: bool) -> None:
        self._check(environment, slot)
        if self._slots[environment][slot] is None and active:
            raise ValueError(f"cannot activate empty {self.kind} slot {slot}")
        self._active[environment, slot] = active
        self.generation += 1

    def remove(self, environment: int, slot: int) -> None:
        self._check(environment, slot)
        self._slots[environment][slot] = None
        self._active[environment, slot] = False
        self.generation += 1

    def clear(self, environment: int | None = None) -> None:
        environments = range(self.environments) if environment is None else (environment,)
        for e in environments:
            if not 0 <= e < self.environments:
                raise IndexError(f"environment index out of range: {e}")
            for slot in range(self.capacity):
                self._slots[e][slot] = None
                self._active[e, slot] = False
        self.generation += 1


@dataclass(frozen=True)
class Cuboid:
    center: torch.Tensor
    rotation: torch.Tensor
    half_extents: torch.Tensor


class PrimitiveCache(_SlotCache[Cuboid]):
    def __init__(self, environments: int, capacity: int) -> None:
        super().__init__(environments, capacity, "primitive")


class MeshCache(_SlotCache[Mesh]):
    def __init__(self, environments: int, capacity: int) -> None:
        super().__init__(environments, capacity, "mesh")


class VoxelCache(_SlotCache[VoxelGrid]):
    def __init__(self, environments: int, capacity: int) -> None:
        super().__init__(environments, capacity, "voxel")
