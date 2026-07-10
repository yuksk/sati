from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi
from skimage.segmentation import flood

__all__ = ["Responsibility"]


BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int_]
UIntArray: TypeAlias = NDArray[np.uint16]
FloatArray: TypeAlias = NDArray[np.float64]
BBox: TypeAlias = tuple[int, int, int, int]
Edge: TypeAlias = tuple[int, float, float]


@dataclass(slots=True)
class _FloodRegion:
    mask: BoolArray
    area: int
    bbox: BBox
    seed: tuple[int, int]


@dataclass(slots=True)
class _TerraceCandidate:
    mask: BoolArray
    support: UIntArray
    area: int
    bbox: BBox
    members: tuple[int, ...]


class _UnionFind:
    """Union-Find for finding connected components in the overlap graph."""

    def __init__(self, n: int) -> None:
        self.parent: IntArray = np.arange(n, dtype=int)
        self.rank: NDArray[np.int8] = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def _bounding_box(mask: BoolArray) -> BBox:
    """Return the bounding box of a Boolean mask."""
    rows, cols = np.nonzero(mask)

    return (
        int(rows.min()),
        int(rows.max()) + 1,
        int(cols.min()),
        int(cols.max()) + 1,
    )


def _boxes_overlap(box_a: BBox, box_b: BBox) -> bool:
    """Return whether two bounding boxes overlap."""
    ar0, ar1, ac0, ac1 = box_a
    br0, br1, bc0, bc1 = box_b

    return ar0 < br1 and br0 < ar1 and ac0 < bc1 and bc0 < ac1


def _intersection_size(
    mask_a: BoolArray,
    box_a: BBox,
    mask_b: BoolArray,
    box_b: BBox,
) -> int:
    """Return the number of pixels shared by two masks."""
    row0 = max(box_a[0], box_b[0])
    row1 = min(box_a[1], box_b[1])
    col0 = max(box_a[2], box_b[2])
    col1 = min(box_a[3], box_b[3])

    if row0 >= row1 or col0 >= col1:
        return 0

    overlap = mask_a[row0:row1, col0:col1] & mask_b[row0:row1, col0:col1]
    return int(np.count_nonzero(overlap))


def _robust_location(values: FloatArray) -> float:
    """Return the median of the finite values."""
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return np.nan

    return float(np.median(finite_values))


def _robust_scale(values: FloatArray) -> float:
    """Return a robust standard-deviation estimate based on the MAD."""
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return np.nan

    median = np.median(finite_values)
    mad = np.median(np.abs(finite_values - median))
    return float(1.4826 * mad)


def _generate_flood_regions(
    image: FloatArray,
    tolerance: float,
    *,
    stride: int,
    min_flood_area: int,
    connectivity: int,
) -> list[_FloodRegion]:
    """Generate flood regions from regularly spaced seed points."""
    ny, nx = image.shape
    finite = np.isfinite(image)

    regions: list[_FloodRegion] = []
    seen_masks: set[bytes] = set()

    row_positions = list(range(stride // 2, ny, stride)) or [ny // 2]
    col_positions = list(range(stride // 2, nx, stride)) or [nx // 2]

    for row in row_positions:
        for col in col_positions:
            if not finite[row, col]:
                continue

            mask = np.asarray(
                flood(
                    image,
                    seed_point=(row, col),
                    tolerance=tolerance,
                    connectivity=connectivity,
                ),
                dtype=bool,
            )
            mask &= finite

            area = int(np.count_nonzero(mask))
            if area < min_flood_area:
                continue

            # Keep only one copy of each identical region.
            packed = np.packbits(mask, axis=None).tobytes()
            if packed in seen_masks:
                continue
            seen_masks.add(packed)

            regions.append(
                _FloodRegion(
                    mask=mask,
                    area=area,
                    bbox=_bounding_box(mask),
                    seed=(row, col),
                )
            )

    return regions


def _merge_overlapping_flood_regions(
    flood_regions: list[_FloodRegion],
    *,
    min_overlap_pixels: int,
    min_overlap_fraction: float,
    min_candidate_area: int,
) -> list[_TerraceCandidate]:
    """
    Merge overlapping flood regions into the same terrace candidate.

    If R1 overlaps R2 and R2 overlaps R3, all three regions are merged
    even when R1 and R3 do not overlap directly.
    """
    n_regions = len(flood_regions)
    union_find = _UnionFind(n_regions)

    for i in range(n_regions):
        region_i = flood_regions[i]

        for j in range(i + 1, n_regions):
            region_j = flood_regions[j]

            if not _boxes_overlap(region_i.bbox, region_j.bbox):
                continue

            overlap = _intersection_size(
                region_i.mask,
                region_i.bbox,
                region_j.mask,
                region_j.bbox,
            )

            if overlap < min_overlap_pixels:
                continue

            overlap_fraction = overlap / min(region_i.area, region_j.area)

            if overlap_fraction >= min_overlap_fraction:
                union_find.union(i, j)

    groups: dict[int, list[int]] = {}

    for index in range(n_regions):
        root = union_find.find(index)
        groups.setdefault(root, []).append(index)

    candidates: list[_TerraceCandidate] = []

    for members in groups.values():
        union_mask = np.zeros_like(flood_regions[0].mask, dtype=bool)
        support = np.zeros_like(union_mask, dtype=np.uint16)

        for index in members:
            mask = flood_regions[index].mask
            union_mask |= mask
            support += mask.astype(np.uint16)

        area = int(np.count_nonzero(union_mask))
        if area < min_candidate_area:
            continue

        candidates.append(
            _TerraceCandidate(
                mask=union_mask,
                support=support,
                area=area,
                bbox=_bounding_box(union_mask),
                members=tuple(members),
            )
        )

    candidates.sort(key=lambda candidate: candidate.area, reverse=True)
    return candidates


def _candidate_label_image(
    candidates: list[_TerraceCandidate],
    shape: tuple[int, int],
) -> IntArray:
    """Convert all terrace candidates into an integer label image."""
    labels: IntArray = np.zeros(shape, dtype=int)

    for candidate_id, candidate in enumerate(candidates, start=1):
        labels[candidate.mask] = candidate_id

    return labels


def _measure_candidate_boundary(
    image: FloatArray,
    mask_i: BoolArray,
    mask_j: BoolArray,
    *,
    adjacency_gap: float,
    edge_band: float,
) -> tuple[bool, float, float]:
    """
    Determine whether two candidates are adjacent and return their local
    height difference.

    A positive delta means that candidate j is higher than candidate i.
    A negative delta means that candidate i is higher than candidate j.
    """
    distance_to_j: FloatArray = np.empty(
        mask_j.shape,
        dtype=np.float64,
    )
    distance_to_i: FloatArray = np.empty(
        mask_i.shape,
        dtype=np.float64,
    )

    ndi.distance_transform_edt(
        ~mask_j,
        distances=distance_to_j,
    )
    ndi.distance_transform_edt(
        ~mask_i,
        distances=distance_to_i,
    )

    min_i_to_j = float(np.min(distance_to_j[mask_i]))
    min_j_to_i = float(np.min(distance_to_i[mask_j]))
    minimum_distance = min(min_i_to_j, min_j_to_i)

    if minimum_distance > adjacency_gap:
        return False, np.nan, 0.0

    side_i = mask_i & (distance_to_j <= min_i_to_j + edge_band)
    side_j = mask_j & (distance_to_i <= min_j_to_i + edge_band)

    values_i = np.asarray(image[side_i], dtype=float)
    values_j = np.asarray(image[side_j], dtype=float)

    height_i = _robust_location(values_i)
    height_j = _robust_location(values_j)

    if not np.isfinite(height_i) or not np.isfinite(height_j):
        return False, np.nan, 0.0

    delta = height_j - height_i

    noise_i = _robust_scale(values_i)
    noise_j = _robust_scale(values_j)

    if not np.isfinite(noise_i):
        noise_i = 0.0
    if not np.isfinite(noise_j):
        noise_j = 0.0

    combined_noise = float(np.hypot(noise_i, noise_j))
    confidence = abs(delta) / (combined_noise + np.finfo(float).eps)

    return True, float(delta), float(confidence)


def _build_directed_adjacency(
    image: FloatArray,
    candidates: list[_TerraceCandidate],
    *,
    adjacency_gap: float,
    edge_band: float,
    min_edge_confidence: float,
) -> list[list[Edge]]:
    """Build a directed adjacency graph from lower to higher candidates."""
    outgoing: list[list[Edge]] = [[] for _ in range(len(candidates))]

    for i, candidate_i in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            candidate_j = candidates[j]

            adjacent, delta, confidence = _measure_candidate_boundary(
                image,
                candidate_i.mask,
                candidate_j.mask,
                adjacency_gap=adjacency_gap,
                edge_band=edge_band,
            )

            if not adjacent or confidence < min_edge_confidence:
                continue

            if delta > 0:
                outgoing[i].append((j, delta, confidence))
            elif delta < 0:
                outgoing[j].append((i, -delta, confidence))

    return outgoing


def _find_best_increasing_path(
    candidates: list[_TerraceCandidate],
    outgoing: list[list[Edge]],
    n_terraces: int,
    *,
    edge_weight: float,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Find a path of n_terraces adjacent candidates with increasing height."""
    areas = np.asarray(
        [candidate.area for candidate in candidates],
        dtype=float,
    )

    best_path: tuple[int, ...] | None = None
    best_deltas: tuple[float, ...] | None = None
    best_score = -np.inf

    def search(
        path: list[int],
        deltas: list[float],
        confidences: list[float],
    ) -> None:
        nonlocal best_path, best_deltas, best_score

        if len(path) == n_terraces:
            area_score = float(np.sum(np.log(areas[path])))
            edge_score = float(np.sum(np.log1p(confidences)))
            score = area_score + edge_weight * edge_score

            if score > best_score:
                best_score = score
                best_path = tuple(path)
                best_deltas = tuple(deltas)
            return

        current = path[-1]

        for next_index, delta, confidence in outgoing[current]:
            if next_index in path:
                continue

            path.append(next_index)
            deltas.append(delta)
            confidences.append(confidence)

            search(path, deltas, confidences)

            path.pop()
            deltas.pop()
            confidences.pop()

    for start in range(len(candidates)):
        search([start], [], [])

    if best_path is None or best_deltas is None:
        raise RuntimeError(
            "Could not find an increasing sequence with the requested "
            "number of adjacent terraces. Increase adjacency_gap or reduce "
            "min_edge_confidence."
        )

    return best_path, best_deltas


class Responsibility:
    """Represent per-class responsibility maps for a two-dimensional image.

    The object stores the input topography as a float array and allocates a
    responsibility array of shape ``(n, height, width)``, where ``n`` is the
    number of terrace classes. Each slice along the first axis corresponds to
    one terrace class.

    Parameters
    ----------
    image : ``numpy.typing.ArrayLike``
        Two-dimensional image.
    n : ``int``
        Number of terrace classes.

    Attributes
    ----------
    values : ``numpy.ndarray``
        Responsibility array for each terrace class and pixel.
    initial_candidates : ``numpy.ndarray``
        Label image of all candidate regions obtained by merging overlapping flood regions.
    """

    def __init__(self, image: ArrayLike, n: int) -> None:
        array = np.asarray(image, dtype=float)

        if array.ndim != 2:
            raise ValueError("image must be a two-dimensional array.")
        if n < 1:
            raise ValueError("n must be at least 1.")

        self.__array: FloatArray = array
        self.n: int = n

        # Leave all terrace-axis values at zero for unclassified pixels.
        self.values: FloatArray = np.zeros(
            (n,) + array.shape,
            dtype=np.float64,
        )

        # Zero denotes no candidate; 1, 2, ... label the merged candidates.
        self.initial_candidates: IntArray = np.zeros(
            array.shape,
            dtype=int,
        )

    def initial_guess(
        self,
        tolerance: float,
        *,
        stride: int = 16,
        min_flood_area: int = 100,
        min_overlap_pixels: int = 1,
        min_overlap_fraction: float = 0.0,
        min_candidate_area: int = 500,
        adjacency_gap: float = 8.0,
        edge_band: float = 4.0,
        min_edge_confidence: float = 1.0,
        edge_weight: float = 1.0,
        connectivity: int = 1,
        use_core: bool = False,
        core_support_fraction: float = 0.2,
    ) -> None:
        """
        Initialize the responsibility from overlapping flood regions.

        If a classified pixel (j, i) belongs to terrace k, this method sets

            values[k, j, i] = 1

        and sets the values for all other terraces at the same pixel to zero.
        All values in values[:, j, i] remain zero for an unclassified pixel.

        initial_candidates stores the label image of all candidate regions
        obtained by merging overlapping flood regions.

        Parameters
        ----------
        tolerance : ``float``
            Height tolerance used by the flood operation.
        stride : ``int``, optional, default 16
            Spacing between flood seed points.
        min_flood_area : ``int``, optional, default 100
            Minimum area of an individual flood region.
        min_overlap_pixels : ``int``, optional, default 1
            Minimum number of overlapping pixels required to merge two regions.
        min_overlap_fraction : ``float``, optional, default 0.0
            Minimum overlap fraction relative to the smaller region.
        min_candidate_area : ``int``, optional, default 500
            Minimum area of a terrace candidate after merging.
        adjacency_gap : ``float``, optional, default 8.0
            Maximum distance for considering two candidates adjacent.
        edge_band : ``float``, optional, default 4.0
            Width of the bands used to compare heights across a boundary.
        min_edge_confidence : ``float``, optional, default 1.0
            Minimum confidence required for the height-ordering decision.
        edge_weight : ``float``, optional, default 1.0
            Weight of boundary confidence in the path score.
        connectivity : ``int``, optional, default 1
            Connectivity used by flood. Use 1 or 2 for a two-dimensional image.
        use_core : ``bool``, optional, default ``False``
            If True, use only pixels supported by multiple overlapping flood
            regions for the initial responsibility.
        core_support_fraction : ``float``, optional, default 0.2
            Relative support threshold used when use_core is True.
        """
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if stride < 1:
            raise ValueError("stride must be at least 1.")
        if min_flood_area < 1:
            raise ValueError("min_flood_area must be at least 1.")
        if min_overlap_pixels < 1:
            raise ValueError("min_overlap_pixels must be at least 1.")
        if not 0.0 <= min_overlap_fraction <= 1.0:
            raise ValueError("min_overlap_fraction must be between 0 and 1.")
        if min_candidate_area < 1:
            raise ValueError("min_candidate_area must be at least 1.")
        if adjacency_gap < 0:
            raise ValueError("adjacency_gap must be non-negative.")
        if edge_band < 0:
            raise ValueError("edge_band must be non-negative.")
        if min_edge_confidence < 0:
            raise ValueError("min_edge_confidence must be non-negative.")
        if connectivity not in (1, 2):
            raise ValueError("connectivity must be 1 or 2.")
        if not 0.0 < core_support_fraction <= 1.0:
            raise ValueError(
                "core_support_fraction must be greater than 0 " "and at most 1."
            )

        flood_regions = _generate_flood_regions(
            self.__array,
            tolerance,
            stride=stride,
            min_flood_area=min_flood_area,
            connectivity=connectivity,
        )

        if not flood_regions:
            raise RuntimeError(
                "No flood regions were found. Reduce min_flood_area or "
                "stride, or increase tolerance."
            )

        candidates = _merge_overlapping_flood_regions(
            flood_regions,
            min_overlap_pixels=min_overlap_pixels,
            min_overlap_fraction=min_overlap_fraction,
            min_candidate_area=min_candidate_area,
        )

        if len(candidates) < self.n:
            raise RuntimeError(
                f"Only {len(candidates)} terrace candidates were found, "
                f"but n={self.n}. Reduce min_candidate_area or use stricter "
                "overlap-merging criteria."
            )

        outgoing = _build_directed_adjacency(
            self.__array,
            candidates,
            adjacency_gap=adjacency_gap,
            edge_band=edge_band,
            min_edge_confidence=min_edge_confidence,
        )

        selected, _ = _find_best_increasing_path(
            candidates,
            outgoing,
            self.n,
            edge_weight=edge_weight,
        )

        # Clear the previous initialization before recomputing it.
        self.values.fill(0.0)

        for level, candidate_index in enumerate(selected):
            candidate = candidates[candidate_index]
            mask = candidate.mask

            if use_core:
                maximum_support = int(np.max(candidate.support[mask]))
                minimum_support = max(
                    1,
                    int(np.ceil(core_support_fraction * maximum_support)),
                )
                mask = mask & (candidate.support >= minimum_support)

            # Set a one-hot vector for each classified pixel.
            self.values[:, mask] = 0.0
            self.values[level, mask] = 1.0

        self.initial_candidates[...] = _candidate_label_image(
            candidates,
            self.__array.shape,
        )

    def classify(self, threshold: float) -> np.ndarray:
        """Assign each pixel to the terrace with the highest responsibility.

        Pixels whose maximum responsibility across all terraces is below
        ``threshold`` are left unassigned (``nan``).

        Parameters
        ----------
        threshold : ``float``
            Minimum responsibility value required to assign a pixel to a terrace.

        Returns
        -------
        ``numpy.ndarray``, shape (height, width)
            Terrace index exceeding the threshold at each pixel. Pixels below
             ``threshold`` are ``nan``.
        """
        maximum = self.values.max(axis=0)
        return np.where(maximum >= threshold, np.argmax(self.values, axis=0), np.nan)
