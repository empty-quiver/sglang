# SPDX-License-Identifier: Apache-2.0
"""Runtime controller for KT CPU/GPU expert residency.

The KT wrapper owns tensors, CUDA streams, and commit timing. This module keeps
the policy-only state used at scheduler-safe boundaries: route counts/scores,
placement scoring, cooldowns, and epoch-level replacement planning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


@dataclass(frozen=True)
class KTStagingCommitStats:
    staged_score: float
    evicted_score: float
    net_score: float
    cost_ms: float
    effective_score: float


@dataclass(frozen=True)
class KTStagingPlacementCandidate:
    staged_expert: int
    evicted_expert: int
    gpu_index: int
    staged_score: float
    evicted_score: float
    net_score: float
    effective_score: float
    route_score: float
    route_count: int


@dataclass
class KTStagingPlacementPlan:
    """Telemetry-rich replacement plan for one planning epoch."""

    observation_count: int
    epoch: int
    interval_observations: int
    target_residents: Tuple[int, ...]
    replacements: Tuple[KTStagingPlacementCandidate, ...]
    considered_pairs: int
    skipped_cold_score: int
    skipped_evict_score: int
    skipped_cooldown: int
    skipped_residency: int
    skipped_hot: int
    skipped_inflight: int

    def replacement_count(self) -> int:
        return len(self.replacements)


class KTStagingController:
    """Per-layer KT residency policy state.

    This class deliberately has no torch dependency. The wrapper converts route
    tensors into counts, asks for candidates/evictions, and commits swaps after
    a safe scheduler boundary.
    """

    def __init__(self, num_experts: int):
        self.num_experts = int(num_experts)
        self.route_scores: List[float] = [0.0] * self.num_experts
        self.cpu_scores: List[float] = [0.0] * self.num_experts
        self.gpu_scores: List[float] = [0.0] * self.num_experts
        self.route_counts: List[int] = [0] * self.num_experts
        self.cooldown_until: List[int] = [0] * self.num_experts
        self.route_last_seen: List[int] = [-10**9] * self.num_experts
        self.residency_started_at: List[int] = [-1] * self.num_experts
        self._observation_count: int = 0
        self._plan_epoch: int = 0
        self._plan_epoch_observation: int = -10**9
        self._plan_resident_signature: Tuple[int, ...] = tuple()
        self._active_plan: Optional[KTStagingPlacementPlan] = None
        self._active_plan_cursor: int = 0

    @property
    def cooldown_observations(self) -> int:
        return _env_int("SGLANG_KT_STAGING_RUNTIME_COOLDOWN_OBSERVATIONS", 2)

    @property
    def min_residency_observations(self) -> int:
        return _env_int("SGLANG_KT_STAGING_RUNTIME_MIN_RESIDENCY_OBSERVATIONS", 8)

    @property
    def hysteresis_score(self) -> float:
        return _env_float("SGLANG_KT_STAGING_RUNTIME_HYSTERESIS_SCORE", 0.5)

    @property
    def eviction_route_weight(self) -> float:
        return _env_float("SGLANG_KT_STAGING_RUNTIME_EVICTION_ROUTE_WEIGHT", 0.25)

    @property
    def placement_plan_observations(self) -> int:
        return _env_int("SGLANG_KT_STAGING_RUNTIME_PLAN_OBSERVATIONS", 32, minimum=1)

    @property
    def max_replacements_per_epoch(self) -> int:
        return _env_int(
            "SGLANG_KT_STAGING_RUNTIME_MAX_REPLACEMENTS_PER_EPOCH", 2, minimum=1
        )

    @property
    def max_swaps_per_policy_epoch(self) -> int:
        return self.max_replacements_per_epoch

    @property
    def recent_hot_window(self) -> int:
        return _env_int("SGLANG_KT_STAGING_RUNTIME_HOT_RESIDENCY_WINDOW", 12, minimum=1)

    @property
    def target_resident_bias(self) -> float:
        return _env_float("SGLANG_KT_STAGING_RUNTIME_TARGET_RESIDENT_BIAS", 0.2)

    @property
    def target_route_weight(self) -> float:
        return _env_float("SGLANG_KT_STAGING_RUNTIME_TARGET_ROUTE_WEIGHT", 0.35)

    def apply_counts(
        self,
        route_counts: Sequence[float],
        cpu_counts: Sequence[float],
        gpu_counts: Sequence[float],
        decay: float,
    ) -> float:
        self._observation_count += 1
        total_count = 0.0
        for expert, route_count in enumerate(list(route_counts)[: self.num_experts]):
            route_count_f = float(route_count)
            cpu_count_f = float(cpu_counts[expert]) if expert < len(cpu_counts) else 0.0
            gpu_count_f = float(gpu_counts[expert]) if expert < len(gpu_counts) else 0.0
            total_count += route_count_f
            self.route_scores[expert] = self.route_scores[expert] * decay + route_count_f
            self.cpu_scores[expert] = self.cpu_scores[expert] * decay + cpu_count_f
            self.gpu_scores[expert] = self.gpu_scores[expert] * decay + gpu_count_f
            self.route_counts[expert] += int(route_count_f)
            if route_count_f > 0.0 or cpu_count_f > 0.0 or gpu_count_f > 0.0:
                self.route_last_seen[expert] = self._observation_count
        return total_count

    def cooldown_remaining(self, expert: int, observation_count: int) -> int:
        if not self._valid_expert(expert):
            return 0
        return max(0, int(self.cooldown_until[expert]) - int(observation_count))

    def layer_scores(self, gpu_mask: Iterable[bool]) -> Tuple[float, float, float, int]:
        route_score = 0.0
        cpu_score = 0.0
        gpu_score = 0.0
        route_count = 0
        for expert, is_gpu in enumerate(gpu_mask):
            if expert >= self.num_experts:
                break
            route_score += self.route_scores[expert]
            route_count += self.route_counts[expert]
            if bool(is_gpu):
                gpu_score += self.gpu_scores[expert]
            else:
                cpu_score += self.cpu_scores[expert]
        return route_score, cpu_score, gpu_score, route_count

    def layer_debt(
        self,
        gpu_mask: Iterable[bool],
        swaps: int,
        predicted_total_cost_ms: float,
        swap_penalty: float,
    ) -> float:
        _, cpu_score, _, _ = self.layer_scores(gpu_mask)
        cost_ms = max(1.0, float(predicted_total_cost_ms))
        return (cpu_score / cost_ms) / (1.0 + swap_penalty * float(swaps))

    def eviction_value(self, expert: int) -> float:
        if not self._valid_expert(expert):
            return float("inf")
        return (
            self.gpu_scores[expert]
            + self.eviction_route_weight * self.route_scores[expert]
        )

    def commit_stats(
        self,
        staged_expert: int,
        evicted_expert: int,
        cost_ms: float,
        commit_margin: float,
    ) -> KTStagingCommitStats:
        staged_score = self.cpu_scores[staged_expert]
        evicted_score = self.eviction_value(evicted_expert)
        net_score = (
            staged_score
            - float(commit_margin) * evicted_score
            - self.hysteresis_score
        )
        cost_ms = max(1.0, float(cost_ms))
        return KTStagingCommitStats(
            staged_score=staged_score,
            evicted_score=evicted_score,
            net_score=net_score,
            cost_ms=cost_ms,
            effective_score=net_score / cost_ms,
        )

    def choose_candidate(
        self,
        cold_experts: Iterable[int],
        inflight_experts: set[int],
        evict_for_expert: Callable[[int], Optional[Tuple[int, int]]],
        min_score: float,
        min_effective_score: float,
        predicted_total_cost_ms: float,
        commit_margin: float,
        observation_count: int = 0,
    ) -> Optional[Tuple[int, Tuple[float, ...]]]:
        best = None
        for expert in cold_experts:
            expert = int(expert)
            if (
                not self._valid_expert(expert)
                or expert in inflight_experts
                or self.cooldown_remaining(expert, observation_count) > 0
            ):
                continue
            cpu_score = self.cpu_scores[expert]
            if cpu_score < min_score:
                continue
            evict = evict_for_expert(expert)
            if evict is None:
                continue
            evicted_expert, _ = evict
            if self.cooldown_remaining(evicted_expert, observation_count) > 0:
                continue
            if evicted_expert in inflight_experts:
                continue
            stats = self.commit_stats(
                expert,
                evicted_expert,
                predicted_total_cost_ms,
                commit_margin,
            )
            if (
                stats.net_score <= 0.0
                or stats.effective_score < min_effective_score
            ):
                continue
            score_key = (
                stats.effective_score,
                stats.net_score,
                cpu_score,
                -stats.evicted_score,
                -stats.cost_ms,
                self.route_scores[expert],
                float(self.route_counts[expert]),
                float(-expert),
            )
            if best is None or score_key > best[1]:
                best = (expert, score_key)
        return best

    def choose_evict(
        self,
        staged_expert: int,
        gpu_index_to_logical: Sequence[int],
        observation_count: int = 0,
    ) -> Optional[Tuple[int, int]]:
        candidates = []
        for gpu_idx, logical in enumerate(gpu_index_to_logical):
            logical_id = int(logical)
            if not self._valid_expert(logical_id) or logical_id == int(staged_expert):
                continue
            if not self._is_eviction_allowed(logical_id, observation_count):
                continue
            candidates.append(
                (
                    self.eviction_value(logical_id),
                    self.gpu_scores[logical_id],
                    self.route_scores[logical_id],
                    logical_id,
                    int(gpu_idx),
                )
            )
        if not candidates:
            return None
        candidates.sort()
        _, _, _, logical_id, gpu_idx = candidates[0]
        return int(logical_id), int(gpu_idx)

    def compute_placement_plan(
        self,
        gpu_mask: Sequence[bool],
        gpu_index_to_logical: Sequence[int],
        observation_count: int,
        inflight_experts: Optional[Set[int]] = None,
        min_score: float = 0.0,
        min_effective_score: float = 0.0,
        predicted_total_cost_ms: float = 1.0,
        commit_margin: float = 0.0,
        force_recompute: bool = False,
    ) -> KTStagingPlacementPlan:
        return self._get_or_refresh_plan(
            gpu_mask=gpu_mask,
            gpu_index_to_logical=gpu_index_to_logical,
            observation_count=observation_count,
            inflight_experts=(
                set(inflight_experts)
                if inflight_experts is not None
                else set()
            ),
            min_score=min_score,
            min_effective_score=min_effective_score,
            predicted_total_cost_ms=predicted_total_cost_ms,
            commit_margin=commit_margin,
            force_recompute=force_recompute,
        )

    # Alias used by callers that prefer explicit "replacements" naming.
    def plan_replacements(
        self,
        gpu_mask: Sequence[bool],
        gpu_index_to_logical: Sequence[int],
        observation_count: int,
        inflight_experts: Optional[Set[int]] = None,
        min_score: float = 0.0,
        min_effective_score: float = 0.0,
        predicted_total_cost_ms: float = 1.0,
        commit_margin: float = 0.0,
        force_recompute: bool = False,
    ) -> KTStagingPlacementPlan:
        return self.compute_placement_plan(
            gpu_mask,
            gpu_index_to_logical,
            observation_count,
            inflight_experts,
            min_score=min_score,
            min_effective_score=min_effective_score,
            predicted_total_cost_ms=predicted_total_cost_ms,
            commit_margin=commit_margin,
            force_recompute=force_recompute,
        )

    def next_prepared_replacement(
        self,
        gpu_mask: Sequence[bool],
        gpu_index_to_logical: Sequence[int],
        observation_count: int,
        inflight_experts: Optional[Set[int]] = None,
        min_score: float = 0.0,
        min_effective_score: float = 0.0,
        predicted_total_cost_ms: float = 1.0,
        commit_margin: float = 0.0,
        force_recompute: bool = False,
    ) -> Optional[KTStagingPlacementCandidate]:
        plan = self.compute_placement_plan(
            gpu_mask,
            gpu_index_to_logical,
            observation_count,
            inflight_experts,
            min_score=min_score,
            min_effective_score=min_effective_score,
            predicted_total_cost_ms=predicted_total_cost_ms,
            commit_margin=commit_margin,
            force_recompute=force_recompute,
        )

        while self._active_plan_cursor < plan.replacement_count():
            cand = plan.replacements[self._active_plan_cursor]
            self._active_plan_cursor += 1
            if inflight_experts is None:
                inflight_experts = set()
            if (
                cand.staged_expert in inflight_experts
                or cand.evicted_expert in inflight_experts
            ):
                continue
            if cand.staged_expert >= len(gpu_mask) or cand.evicted_expert >= len(gpu_mask):
                continue
            if bool(gpu_mask[cand.staged_expert]) or not bool(gpu_mask[cand.evicted_expert]):
                continue
            if self.cooldown_remaining(cand.staged_expert, observation_count) > 0:
                continue
            if self.cooldown_remaining(cand.evicted_expert, observation_count) > 0:
                continue
            if not self._is_residency_stable(cand.evicted_expert, observation_count):
                continue
            if self._is_recently_hot(cand.evicted_expert, observation_count):
                continue
            return cand
        return None

    # Alias for callers that prefer a shorter term.
    def next_replacement(self, *args, **kwargs) -> Optional[KTStagingPlacementCandidate]:
        return self.next_prepared_replacement(*args, **kwargs)

    def record_prepare(
        self, staged_expert: int, evicted_expert: Optional[int] = None
    ) -> None:  # no-op by design
        return

    def should_commit_prepared(
        self,
        staged_expert: int,
        evicted_expert: int,
        observation_count: Optional[int] = None,
    ) -> Optional[bool]:
        if not self._valid_expert(staged_expert) or not self._valid_expert(
            evicted_expert
        ):
            return False

        if observation_count is None:
            return None

        if self.cooldown_remaining(staged_expert, observation_count) > 0:
            return False
        if self.cooldown_remaining(evicted_expert, observation_count) > 0:
            return False
        if not self._is_residency_stable(evicted_expert, observation_count):
            return False
        return True

    def record_resident_skip(self, expert: int) -> None:
        if not self._valid_expert(expert):
            return
        self.cpu_scores[expert] = 0.0
        self.gpu_scores[expert] = max(
            self.gpu_scores[expert], self.route_scores[expert]
        )

    def record_commit(
        self, staged_expert: int, evicted_expert: int, observation_count: int
    ) -> None:
        if not self._valid_expert(staged_expert) or not self._valid_expert(
            evicted_expert
        ):
            return
        self.cpu_scores[staged_expert] = 0.0
        self.cpu_scores[evicted_expert] = 0.0
        self.gpu_scores[evicted_expert] = 0.0
        self.gpu_scores[staged_expert] = max(
            self.gpu_scores[staged_expert], self.route_scores[staged_expert]
        )
        until = int(observation_count) + self.cooldown_observations
        self.cooldown_until[staged_expert] = max(
            self.cooldown_until[staged_expert], until
        )
        self.cooldown_until[evicted_expert] = max(
            self.cooldown_until[evicted_expert], until
        )
        self.residency_started_at[staged_expert] = int(observation_count)
        self.residency_started_at[evicted_expert] = -1
        # A commit changes the live mask semantics; keep the existing plan fresh
        # if wrapper chooses to materialize more in the same epoch.
        self._plan_epoch_observation = int(observation_count) - (
            self.placement_plan_observations - 1
        )

    def _get_or_refresh_plan(
        self,
        gpu_mask: Sequence[bool],
        gpu_index_to_logical: Sequence[int],
        observation_count: int,
        inflight_experts: Set[int],
        min_score: float,
        min_effective_score: float,
        predicted_total_cost_ms: float,
        commit_margin: float,
        force_recompute: bool = False,
    ) -> KTStagingPlacementPlan:
        current_signature = self._resident_signature(gpu_mask)
        stale = (
            force_recompute
            or self._active_plan is None
            or int(observation_count) - self._plan_epoch_observation
            >= self.placement_plan_observations
            or current_signature != self._plan_resident_signature
        )
        if not stale:
            return self._active_plan
        return self._build_plan(
            gpu_mask=gpu_mask,
            gpu_index_to_logical=gpu_index_to_logical,
            observation_count=observation_count,
            inflight_experts=inflight_experts,
            min_score=min_score,
            min_effective_score=min_effective_score,
            predicted_total_cost_ms=predicted_total_cost_ms,
            commit_margin=commit_margin,
            current_resident_signature=current_signature,
        )

    def _build_plan(
        self,
        gpu_mask: Sequence[bool],
        gpu_index_to_logical: Sequence[int],
        observation_count: int,
        inflight_experts: Set[int],
        min_score: float,
        min_effective_score: float,
        predicted_total_cost_ms: float,
        commit_margin: float,
        current_resident_signature: Tuple[int, ...],
    ) -> KTStagingPlacementPlan:
        obs = int(observation_count)
        self._observation_count = max(self._observation_count, obs)
        self._plan_epoch += 1
        self._plan_epoch_observation = obs
        self._plan_resident_signature = current_resident_signature
        self._active_plan_cursor = 0

        residents = self._resident_experts(gpu_mask)
        if not residents:
            plan = KTStagingPlacementPlan(
                observation_count=obs,
                epoch=self._plan_epoch,
                interval_observations=self.placement_plan_observations,
                target_residents=tuple(),
                replacements=tuple(),
                considered_pairs=0,
                skipped_cold_score=0,
                skipped_evict_score=0,
                skipped_cooldown=0,
                skipped_residency=0,
                skipped_hot=0,
                skipped_inflight=0,
            )
            self._active_plan = plan
            return plan

        capacity = len(residents)
        for expert in residents:
            if self.residency_started_at[expert] < 0:
                self.residency_started_at[expert] = obs

        target_residents = tuple(
            self._rank_experts_for_residency(
                gpu_mask, residents, obs
            )[:capacity]
        )
        cold_candidates = tuple(sorted(set(target_residents) - set(residents)))
        evict_candidates = tuple(sorted(set(residents) - set(target_residents)))

        if not cold_candidates or not evict_candidates:
            plan = KTStagingPlacementPlan(
                observation_count=obs,
                epoch=self._plan_epoch,
                interval_observations=self.placement_plan_observations,
                target_residents=target_residents,
                replacements=tuple(),
                considered_pairs=0,
                skipped_cold_score=0,
                skipped_evict_score=0,
                skipped_cooldown=0,
                skipped_residency=0,
                skipped_hot=0,
                skipped_inflight=0,
            )
            self._active_plan = plan
            return plan

        logical_to_gpu_idx = self._logical_to_gpu_index(gpu_index_to_logical)
        pair_scores: List[Tuple[Tuple[float, ...], KTStagingPlacementCandidate]] = []
        skipped_cold_score = 0
        skipped_evict_score = 0
        skipped_cooldown = 0
        skipped_residency = 0
        skipped_hot = 0
        skipped_inflight = 0
        considered_pairs = 0

        for staged in cold_candidates:
            if staged in inflight_experts:
                skipped_inflight += 1
                continue
            staged_score = self.cpu_scores[staged]
            if staged_score < min_score:
                skipped_cold_score += 1
                continue
            if self.cooldown_remaining(staged, obs) > 0:
                skipped_cooldown += 1
                continue
            for evicted in evict_candidates:
                if evicted in inflight_experts:
                    skipped_inflight += 1
                    continue
                considered_pairs += 1
                if self.cooldown_remaining(evicted, obs) > 0:
                    skipped_cooldown += 1
                    continue
                if self._is_recently_hot(evicted, obs):
                    skipped_hot += 1
                    continue
                if not self._is_residency_stable(evicted, obs):
                    skipped_residency += 1
                    continue
                gpu_idx = logical_to_gpu_idx.get(evicted)
                if gpu_idx is None:
                    skipped_evict_score += 1
                    continue
                stats = self.commit_stats(
                    staged,
                    evicted,
                    predicted_total_cost_ms,
                    commit_margin,
                )
                if (
                    stats.net_score <= 0.0
                    or stats.effective_score < min_effective_score
                ):
                    skipped_evict_score += 1
                    continue
                pair_scores.append(
                    (
                        self._placement_sort_key(stats, staged),
                        int(staged),
                        int(evicted),
                        KTStagingPlacementCandidate(
                            staged_expert=staged,
                            evicted_expert=evicted,
                            gpu_index=gpu_idx,
                            staged_score=stats.staged_score,
                            evicted_score=stats.evicted_score,
                            net_score=stats.net_score,
                            effective_score=stats.effective_score,
                            route_score=self.route_scores[staged],
                            route_count=self.route_counts[staged],
                        ),
                    )
                )

        pair_scores.sort(reverse=True)
        selected: List[KTStagingPlacementCandidate] = []
        used_staged: Set[int] = set()
        used_evicted: Set[int] = set()
        for _, __, ___, cand in pair_scores:
            if len(selected) >= self.max_replacements_per_epoch:
                break
            if (
                cand.staged_expert in used_staged
                or cand.evicted_expert in used_evicted
            ):
                continue
            used_staged.add(cand.staged_expert)
            used_evicted.add(cand.evicted_expert)
            selected.append(cand)

        plan = KTStagingPlacementPlan(
            observation_count=obs,
            epoch=self._plan_epoch,
            interval_observations=self.placement_plan_observations,
            target_residents=target_residents,
            replacements=tuple(selected),
            considered_pairs=considered_pairs,
            skipped_cold_score=skipped_cold_score,
            skipped_evict_score=skipped_evict_score,
            skipped_cooldown=skipped_cooldown,
            skipped_residency=skipped_residency,
            skipped_hot=skipped_hot,
            skipped_inflight=skipped_inflight,
        )
        self._active_plan = plan
        return plan

    def _placement_sort_key(
        self, stats: KTStagingCommitStats, staged_expert: int
    ) -> Tuple[float, float, float, float, float, float]:
        return (
            stats.effective_score,
            stats.net_score,
            stats.staged_score,
            -stats.evicted_score,
            self.route_scores[staged_expert],
            self.route_counts[staged_expert],
        )

    def _rank_experts_for_residency(
        self, gpu_mask: Sequence[bool], residents: Tuple[int, ...], observation: int
    ) -> List[int]:
        del residents
        ranked = []
        current = self._is_resident_flags(gpu_mask)
        for expert in range(self.num_experts):
            if not self._valid_expert(expert):
                continue
            value = self.cpu_scores[expert] + self.target_route_weight * self.route_scores[
                expert
            ]
            if current[expert]:
                value += self.target_resident_bias
            ranked.append((value, expert))
        ranked.sort(reverse=True)
        return [expert for _, expert in ranked]

    def _is_eviction_allowed(self, expert: int, observation_count: int) -> bool:
        if not self._valid_expert(expert):
            return False
        if self.cooldown_remaining(expert, observation_count) > 0:
            return False
        if not self._is_residency_stable(expert, observation_count):
            return False
        if self._is_recently_hot(expert, observation_count):
            return False
        return True

    def _is_residency_stable(self, expert: int, observation_count: int) -> bool:
        started = self.residency_started_at[expert]
        if started < 0:
            return True
        return int(observation_count) - int(started) >= self.min_residency_observations

    def _is_recently_hot(self, expert: int, observation_count: int) -> bool:
        if self.recent_hot_window <= 0:
            return False
        if (
            self.route_last_seen[expert] < 0
            or self.route_scores[expert] <= 0.0
            or self.gpu_scores[expert] <= 0.0
        ):
            return False
        return (int(observation_count) - int(self.route_last_seen[expert])) <= int(
            self.recent_hot_window
        )

    def _resident_signature(self, gpu_mask: Sequence[bool]) -> Tuple[int, ...]:
        return tuple(
            expert
            for expert, is_resident in enumerate(gpu_mask[: self.num_experts])
            if bool(is_resident)
        )

    def _resident_experts(self, gpu_mask: Sequence[bool]) -> Tuple[int, ...]:
        return self._resident_signature(gpu_mask)

    def _is_resident_flags(self, gpu_mask: Sequence[bool]) -> List[bool]:
        flags = [False] * self.num_experts
        for expert in self._resident_signature(gpu_mask):
            flags[expert] = True
        return flags

    def _logical_to_gpu_index(self, gpu_index_to_logical: Sequence[int]) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for gpu_idx, logical in enumerate(gpu_index_to_logical):
            logical_id = int(logical)
            if self._valid_expert(logical_id):
                mapping.setdefault(logical_id, gpu_idx)
        return mapping

    def _valid_expert(self, expert: int) -> bool:
        return 0 <= int(expert) < self.num_experts
