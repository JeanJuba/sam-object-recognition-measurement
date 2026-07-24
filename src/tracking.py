"""Rastreamento de objetos entre frames (Fase 2).

Tracker por centroide: associa detecções a tracks existentes pela menor
distância euclidiana, mantendo IDs estáveis enquanto a peça atravessa a
cena. Suficiente para esteiras/bancadas com movimento suave.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    track_id: int
    centroid: tuple[int, int]
    hits: int = 1                 # nº de frames em que foi detectado
    missed: int = 0               # frames consecutivos sem detecção
    measurements: list = field(default_factory=list)  # histórico de medições


class CentroidTracker:
    def __init__(self, max_distance_px: float = 100, max_missed_frames: int = 15, min_hits: int = 3):
        self.max_distance = max_distance_px
        self.max_missed = max_missed_frames
        self.min_hits = min_hits
        self._next_id = 1
        self.active: dict[int, Track] = {}
        self.finished: list[Track] = []

    def update(self, centroids: list[tuple[int, int]]) -> dict[int, int]:
        """Atualiza os tracks com os centroides do frame atual.

        Returns:
            Mapeamento índice_da_detecção -> track_id.
        """
        assignment: dict[int, int] = {}
        unmatched = set(range(len(centroids)))

        if self.active and centroids:
            track_ids = list(self.active.keys())
            t_pos = np.array([self.active[t].centroid for t in track_ids], dtype=float)
            d_pos = np.array(centroids, dtype=float)
            dist = np.linalg.norm(t_pos[:, None, :] - d_pos[None, :, :], axis=2)

            # associação gulosa: pares de menor distância primeiro
            pairs = sorted(
                ((dist[i, j], i, j) for i in range(len(track_ids)) for j in range(len(centroids))),
            )
            used_tracks: set[int] = set()
            for d, i, j in pairs:
                if d > self.max_distance or i in used_tracks or j not in unmatched:
                    continue
                tid = track_ids[i]
                track = self.active[tid]
                track.centroid = centroids[j]
                track.hits += 1
                track.missed = 0
                assignment[j] = tid
                used_tracks.add(i)
                unmatched.discard(j)

        # novas detecções viram novos tracks
        for j in unmatched:
            tid = self._next_id
            self._next_id += 1
            self.active[tid] = Track(track_id=tid, centroid=centroids[j])
            assignment[j] = tid

        # tracks não atualizados neste frame
        matched_ids = set(assignment.values())
        for tid in list(self.active.keys()):
            if tid not in matched_ids:
                track = self.active[tid]
                track.missed += 1
                if track.missed > self.max_missed:
                    self._finish(tid)

        return assignment

    def _finish(self, tid: int) -> None:
        track = self.active.pop(tid)
        if track.hits >= self.min_hits:
            self.finished.append(track)

    def finalize(self) -> list[Track]:
        """Encerra todos os tracks ativos e devolve os válidos."""
        for tid in list(self.active.keys()):
            self._finish(tid)
        return self.finished
