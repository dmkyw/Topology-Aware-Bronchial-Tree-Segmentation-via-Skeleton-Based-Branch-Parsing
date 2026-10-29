#!/usr/bin/env python
"""
Bronchial Branch Labeler
========================

A lightweight utility for anatomical labeling of bronchial-tree branches from a
binary airway segmentation stored as NRRD / Slicer .seg.nrrd.

Pipeline
--------
1. Load a binary bronchial-tree mask.
2. Skeletonize the airway using kimimaro TEASAR.
3. Build a graph from the airway skeleton.
4. Trace and classify major bronchial branches using graph topology and PCA-based
   anatomical directions.
5. Assign airway voxels to the nearest labeled skeleton branch.
6. Export a 3D Slicer-compatible .seg.nrrd and a text label table.

Example
-------
python bronchial_branch_labeler.py \
    --input "D:/LyNoS_dataset/Benchmark/Pat2/lung_vessels_segmentation/lung_vessels_segmentation_bronchial_ori.seg.nrrd" \
    --ct "D:/LyNoS_dataset/Benchmark/Pat2/pat2_data.nii.gz" \
    --output "D:/LyNoS_dataset/Benchmark/Pat2/lung_vessels_segmentation/bronchial_BRANCHES.seg.nrrd"

Notes
-----
- The input airway mask is expected to contain airway voxels with value 1.
- AP orientation may vary between datasets. Use --ap-sign -1 if anterior/posterior
  assignment is reversed.
"""

from __future__ import annotations

import argparse
import colorsys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import kimimaro
import nibabel as nib
import nrrd
import numpy as np
from scipy.spatial import cKDTree


# bid -> (anatomical name, RGB color for Slicer)
LABELS: Dict[int, Tuple[str, Tuple[float, float, float]]] = {
    0: ("Trachea", (1.00, 1.00, 0.00)),
    1: ("LMB", (0.00, 0.00, 1.00)),
    2: ("RMB", (1.00, 0.00, 0.00)),
    3: ("RUL", (1.00, 0.50, 0.00)),
    4: ("BI", (0.60, 0.00, 0.60)),
    5: ("LUL", (0.00, 1.00, 1.00)),
    6: ("LLL", (0.00, 0.60, 0.00)),
    7: ("Rt_B1", (1.00, 0.00, 0.20)),
    8: ("Rt_B2", (1.00, 0.30, 0.30)),
    9: ("Rt_B3", (1.00, 0.60, 0.60)),
    10: ("RML", (0.80, 0.00, 0.80)),
    11: ("RLL", (0.50, 0.00, 0.50)),
    12: ("UDB", (0.00, 1.00, 0.80)),
    13: ("LIN", (0.00, 0.80, 0.50)),
    14: ("Rt_B2+B3", (1.00, 0.45, 0.10)),
    15: ("Rt_B1+B2", (1.00, 0.15, 0.65)),
    16: ("Rt_B1+B3", (0.75, 0.10, 0.10)),
    17: ("Lt_B1", (0.10, 0.70, 1.00)),
    18: ("Lt_B2", (0.10, 0.45, 0.90)),
    19: ("Lt_B3", (0.30, 0.85, 1.00)),
    20: ("Lt_B2+B3", (0.00, 0.55, 0.85)),
    21: ("Lt_B1+B2", (0.20, 0.35, 0.85)),
    22: ("Lt_B1+B3", (0.00, 0.75, 0.65)),
    23: ("Rt_B4", (0.90, 0.10, 0.90)),
    24: ("Rt_B5", (0.95, 0.45, 0.95)),
    25: ("Rt_B4+B5", (0.75, 0.25, 0.75)),
    26: ("Lt_B4", (0.10, 0.95, 0.95)),
    27: ("Lt_B5", (0.35, 0.75, 0.95)),
    28: ("Lt_B4+B5", (0.15, 0.65, 0.75)),
    29: ("Rt_B6", (0.95, 0.20, 0.05)),
    30: ("Rt_B7", (0.95, 0.45, 0.05)),
    31: ("Rt_B8", (0.95, 0.70, 0.05)),
    32: ("Rt_B9", (0.65, 0.15, 0.05)),
    33: ("Rt_B10", (0.45, 0.05, 0.02)),
    34: ("Rt_B9+B10", (0.70, 0.30, 0.10)),
    35: ("Lt_B6", (0.05, 0.55, 1.00)),
    36: ("Lt_B8", (0.20, 0.75, 1.00)),
    37: ("Lt_B9", (0.00, 0.45, 0.80)),
    38: ("Lt_B10", (0.00, 0.25, 0.60)),
    39: ("Lt_B9+B10", (0.10, 0.45, 0.65)),
}
NAME_TO_BID = {name: bid for bid, (name, _) in LABELS.items()}

DEFAULT_EXPORT_BIDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17,
    20, 23, 24, 26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 38,
]


class TopologyStructureError(RuntimeError):
    """An explicit limitation of the observed skeleton, not an internal crash."""

    def __init__(self, code: str, message: str, diagnostics=None):
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics or {}


def branch_color(bid: int) -> Tuple[float, float, float]:
    if bid in LABELS:
        return LABELS[bid][1]
    return colorsys.hsv_to_rgb(((bid - 1000) * 0.61803398875) % 1.0, 0.75, 0.95)


class BronchialBranchLabeler:
    def __init__(
        self,
        mask: np.ndarray,
        ap_sign: float = 1.0,
        min_subtree_size: int = 5,
        voxel_directions_ras_mm: Optional[np.ndarray] = None,
        strict_anatomy: bool = False,
    ):
        self.mask = mask.astype(bool)
        self.ap_sign = float(ap_sign)
        self.min_subtree_size = int(min_subtree_size)
        self.strict_anatomy = bool(strict_anatomy)
        self.parsing_mode = "NOT_PARSED"
        self.diagnostics = {"algorithm": "rooted-ras-topology/2", "skeleton_attempts": []}
        self.voxel_directions_ras_mm = np.asarray(
            voxel_directions_ras_mm if voxel_directions_ras_mm is not None else np.eye(3),
            dtype=float,
        )
        if self.voxel_directions_ras_mm.shape != (3, 3) or not np.isfinite(self.voxel_directions_ras_mm).all():
            raise ValueError("Voxel geometry must contain three finite spatial axes.")
        self.voxel_spacing_mm = np.linalg.norm(self.voxel_directions_ras_mm, axis=1)
        if np.any(self.voxel_spacing_mm <= 0):
            raise ValueError("Voxel geometry must contain three finite, nonzero spatial axes.")
        self.voxel_axis_directions_ras = (
            self.voxel_directions_ras_mm / self.voxel_spacing_mm[:, None]
        )

        self.verts: np.ndarray
        self.edges: np.ndarray
        self.adj: List[List[int]]
        self.degree: np.ndarray
        self.dist: np.ndarray
        self.pc1: np.ndarray
        self.pc2: np.ndarray
        self.pc3: np.ndarray
        self.carina: int

    def run(self) -> Dict[int, Set[int]]:
        self._skeletonize()
        self._build_graph()
        # A 50 mm TEASAR invalidation radius can suppress short, existing limbs.
        # Retry the skeleton only; the mask is never filled or altered.
        for const_mm in (10.0,):
            if np.any(self.degree >= 3):
                break
            self._skeletonize(const_mm=const_mm)
            self._build_graph()
        self._estimate_axes()
        self.diagnostics.update({"skeleton_vertices": len(self.verts),
                                 "skeleton_edges": len(self.edges),
                                 "branch_points": int(np.count_nonzero(self.degree >= 3)),
                                 "connected_components": self.graph_components})
        if not np.any(self.degree >= 3):
            raise TopologyStructureError("NO_BRANCH_POINTS", "No graph branching point was found after skeleton scale retries. No anatomical branches were invented.", self.diagnostics)
        try:
            if self.graph_components != 1 or len(self.edges) != len(self.verts)-1:
                raise TopologyStructureError("NON_TREE_SKELETON", "The observed skeleton is disconnected or cyclic; anatomical naming is unresolved.")
            self._find_carina()
            segments = self._discover_segments()
        except TopologyStructureError as error:
            self.diagnostics.update({"anatomy_unresolved_code": error.code,
                                     "anatomy_unresolved_reason": str(error)})
            if self.strict_anatomy:
                raise
            self.parsing_mode = "TOPOLOGY_ONLY"
            segments = self._topological_segments()
        else:
            self.parsing_mode = "ANATOMICAL_CANDIDATES"
        self.diagnostics["parsing_mode"] = self.parsing_mode
        return segments

    def export_items(self, segment_nodes: Dict[int, Set[int]]) -> List[Tuple[int, str]]:
        if self.parsing_mode == "TOPOLOGY_ONLY":
            return [(bid, f"Topology_Branch_{i:03d}")
                    for i, bid in enumerate(sorted(segment_nodes), 1)]
        return build_export_items(segment_nodes, DEFAULT_EXPORT_BIDS)

    def _skeletonize(self, const_mm: float = 50.0) -> None:
        skeletons = kimimaro.skeletonize(
            self.mask.astype(np.uint8),
            teasar_params={
                "scale": 1.0,
                "const": const_mm,
                "pdrf_exponent": 4,
                "pdrf_scale": 100000,
            },
            dust_threshold=0,
            anisotropy=tuple(float(v) for v in self.voxel_spacing_mm),
            fix_branching=True,
            fix_borders=True,
        )
        if not skeletons:
            raise TopologyStructureError("EMPTY_SKELETON", "No skeleton was generated. Please check the input airway mask.")
        # Kimimaro returns vertices in physical units along the array axes.
        # Rotate/reflect those axes into RAS so anatomical heuristics and voxel
        # assignment use the same physical coordinate system.
        vertices, edges, offset = [], [], 0
        for skeleton in skeletons.values():
            vertices.append(skeleton.vertices.astype(np.float64) @ self.voxel_axis_directions_ras)
            edges.append(skeleton.edges.astype(np.int64).reshape(-1, 2) + offset)
            offset += len(skeleton.vertices)
        self.verts = np.concatenate(vertices)
        self.edges = np.concatenate(edges).reshape(-1, 2)
        self.diagnostics["skeleton_attempts"].append({"const_mm": const_mm,
            "vertices": len(self.verts), "components": len(skeletons)})

    def _build_graph(self) -> None:
        self.adj = [[] for _ in range(len(self.verts))]
        for a, b in self.edges:
            self.adj[int(a)].append(int(b))
            self.adj[int(b)].append(int(a))
        self.degree = np.array([len(nbs) for nbs in self.adj])
        remaining = set(range(len(self.verts)))
        self.graph_components = 0
        while remaining:
            stack = [min(remaining)]
            self.graph_components += 1
            while stack:
                node = stack.pop()
                if node in remaining:
                    remaining.remove(node)
                    stack.extend(self.adj[node])

    def _estimate_axes(self) -> None:
        # Vertices are already physical RAS. PCA signs/order are not anatomical
        # axes and change when branches are added, cropped, or reflected.
        self.pc1 = np.array([0.0, 0.0, 1.0])  # superior
        self.pc2 = np.array([1.0, 0.0, 0.0])  # right
        self.pc3 = np.array([0.0, 1.0, 0.0])  # anterior

    def _find_carina(self) -> None:
        branch_idx = np.where(self.degree >= 3)[0]
        if len(branch_idx) == 0:
            raise TopologyStructureError("NO_BRANCH_POINTS",
                "No graph branching point was found in the airway mask. "
                "Provide the complete bifurcating tracheobronchial tree; an isolated airway segment or "
                "single centerline cannot be parsed into anatomical branches. No output was written."
            )
        endpoints = np.flatnonzero(self.degree == 1)
        if not len(endpoints):
            raise TopologyStructureError("NO_TRACHEAL_ENDPOINT", "No terminal tracheal candidate is available.")
        self.trachea_root = int(max(endpoints, key=lambda n: (self.verts[n, 2], -abs(self.verts[n, 0]))))
        path, previous, current = [], -1, self.trachea_root
        while current not in path:
            path.append(current)
            children = [n for n in self.adj[current] if n != previous]
            if len(children) >= 2:
                self.carina = current
                self._trachea_path = path
                self.dist = self.bfs_distance(self.carina)
                self.diagnostics.update({"root_method": "superior_endpoint_to_first_junction",
                    "trachea_root": self.trachea_root, "carina": self.carina})
                return
            if not children:
                break
            previous, current = current, children[0]
        raise TopologyStructureError("CARINA_UNRESOLVED", "The tracheal candidate does not reach a branching point.")

    def _select_main_neighbors(self) -> Tuple[int, int]:
        parent = self._trachea_path[-2]
        children = [n for n in self.adj[self.carina] if n != parent]
        if len(children) < 2:
            raise TopologyStructureError("MAIN_BRANCHES_UNRESOLVED", "Two distal bronchial limbs were not observed.")
        # Rank actual distal subtrees, excluding the known proximal limb.
        # The old minimum-angle rule frequently selected the trachea instead.
        sizes = {n: len(self.trace_component(n, blocked={self.carina})) for n in children}
        return max(((a,b) for i,a in enumerate(children) for b in children[i+1:]),
                   key=lambda pair: (min(sizes[pair[0]], sizes[pair[1]]),
                                     sizes[pair[0]]+sizes[pair[1]], -min(pair)))

    def _topological_segments(self) -> Dict[int, Set[int]]:
        """Partition observed graph paths without assigning anatomical names."""
        visited, paths = set(), []
        key = lambda n: (-float(self.verts[n,2]), float(self.verts[n,0]), float(self.verts[n,1]), n)
        critical = sorted(np.flatnonzero(self.degree != 2).tolist(), key=key)
        for start in critical + sorted(np.flatnonzero(self.degree == 2).tolist(), key=key):
            if self.degree[start] == 0:
                paths.append([start])
            for neighbor in sorted(self.adj[start], key=key):
                edge = tuple(sorted((start,neighbor)))
                if edge in visited:
                    continue
                path, previous, current = [start], start, neighbor
                visited.add(edge)
                while True:
                    path.append(current)
                    if self.degree[current] != 2:
                        break
                    following = next(n for n in self.adj[current] if n != previous)
                    edge = tuple(sorted((current,following)))
                    if edge in visited:
                        break
                    visited.add(edge)
                    previous, current = current, following
                paths.append(path)
        return {1000+i: set(path) for i,path in enumerate(paths)}

    def unit_vec(self, a: int, b: int) -> np.ndarray:
        v = self.verts[a] - self.verts[b]
        return v / (np.linalg.norm(v) + 1e-8)

    def bfs_distance(self, root: int) -> np.ndarray:
        dist = np.full(len(self.verts), -1, dtype=np.int32)
        dist[root] = 0
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def shortest_path(self, start: Optional[int], end: Optional[int]) -> Optional[List[int]]:
        if start is None or end is None:
            return None
        parent = {int(start): -1}
        queue = deque([int(start)])
        while queue:
            u = queue.popleft()
            if u == end:
                break
            for v in self.adj[u]:
                if v not in parent:
                    parent[v] = u
                    queue.append(v)
        if end not in parent:
            return None

        path, cur = [], int(end)
        while cur != -1:
            path.append(cur)
            cur = parent[cur]
        return path[::-1]

    def trace_until_branch(self, start: int, parent: int) -> Optional[int]:
        cur, prev = int(start), int(parent)
        while True:
            if self.degree[cur] >= 3:
                return cur
            nxts = [n for n in self.adj[cur] if n != prev and self.dist[n] > self.dist[cur]]
            if not nxts:
                return None
            cur, prev = max(nxts, key=lambda x: self.dist[x]), cur

    def first_branch(self, node: int, parent: int) -> Optional[int]:
        return self.trace_until_branch(node, parent)

    def branch_children(self, node: int, parent: int) -> List[int]:
        return list({
            c for c in (
                self.trace_until_branch(n, node)
                for n in self.adj[node]
                if n != parent and self.dist[n] > self.dist[node]
            )
            if c is not None
        })

    def split_by_up(self, children: Sequence[int], parent: int) -> Tuple[Optional[int], Optional[int]]:
        if len(children) < 2:
            return None, None
        scored = sorted(
            ((c, float(np.dot(self.unit_vec(c, parent), self.pc1))) for c in children),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[0][0], scored[-1][0]

    def trace_component(
        self,
        seed: int,
        blocked: Optional[Iterable[int]] = None,
        allowed: Optional[Iterable[int]] = None,
    ) -> Set[int]:
        blocked_set = set(blocked or [])
        allowed_set = None if allowed is None else set(allowed)
        visited, stack = set(), [int(seed)]
        while stack:
            cur = stack.pop()
            if cur in visited or cur in blocked_set or (allowed_set is not None and cur not in allowed_set):
                continue
            visited.add(cur)
            for n in self.adj[cur]:
                if n not in visited and n not in blocked_set and (allowed_set is None or n in allowed_set):
                    stack.append(n)
        return visited

    def direction_info(self, nodes: Set[int], root_node: int, seed: Optional[int] = None, attach: Optional[int] = None) -> dict:
        node_list = list(nodes)
        pts = self.verts[node_list]
        root = self.verts[root_node]
        d = np.linalg.norm(pts - root, axis=1) + 1e-6
        w = d ** 1.5
        wc = (pts * w[:, None]).sum(axis=0) / w.sum()
        vec = wc - root
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return {
            "seed": seed,
            "attach": attach,
            "nodes": nodes,
            "weighted_centroid": wc,
            "vec": vec,
            "up_score": float(np.dot(vec, self.pc1)),
            "ap_score": float(self.ap_sign * np.dot(vec, self.pc3)),
            "x_score": float(vec[0]),
            "z_score": float(wc[2]),
            "size": len(nodes),
            "farthest": node_list[int(np.argmax(d))],
        }

    def distal_subtrees(self, root_node: int, parent_node: int, keep: Optional[int] = None) -> List[dict]:
        infos = []
        for seed in [n for n in self.adj[root_node] if n != parent_node and self.dist[n] > self.dist[root_node]]:
            nodes = self.trace_component(seed, blocked={root_node})
            if len(nodes) >= self.min_subtree_size:
                infos.append(self.direction_info(nodes, root_node, seed=seed, attach=root_node))
        infos.sort(key=lambda x: x["size"], reverse=True)
        return infos[:keep] if keep else infos

    def collect_side_subtrees(
        self,
        trunk_nodes: Sequence[int],
        extra_blocked: Optional[Iterable[int]] = None,
    ) -> List[dict]:
        trunk_set = set(trunk_nodes)
        blocked = trunk_set | set(extra_blocked or [])
        out, seen = [], set()
        for attach in trunk_nodes:
            for nb in self.adj[attach]:
                if nb in blocked:
                    continue
                nodes = self.trace_component(nb, blocked=blocked)
                key = frozenset(nodes)
                if len(nodes) < self.min_subtree_size or key in seen:
                    continue
                seen.add(key)
                out.append(self.direction_info(nodes, attach, seed=nb, attach=attach))
        return out

    def trace_trachea(self) -> List[int]:
        return list(reversed(self._trachea_path))

    def classify_b123(self, root_node: int, parent_node: int, side_prefix: str) -> Dict[str, Set[int]]:
        infos = self.distal_subtrees(root_node, parent_node, keep=3)
        labels = {
            "B1": f"{side_prefix}_B1",
            "B2": f"{side_prefix}_B2",
            "B3": f"{side_prefix}_B3",
            "B23": f"{side_prefix}_B2+B3",
            "B12": f"{side_prefix}_B1+B2",
            "B13": f"{side_prefix}_B1+B3",
        }

        if len(infos) >= 3:
            b1 = max(infos, key=lambda x: x["up_score"])
            rest = [x for x in infos if x["seed"] != b1["seed"]]
            b3 = max(rest, key=lambda x: x["ap_score"])
            b2 = min(rest, key=lambda x: x["ap_score"])
            assignment = {labels["B1"]: b1, labels["B2"]: b2, labels["B3"]: b3}
        elif len(infos) == 2:
            up_sorted = sorted(infos, key=lambda x: x["up_score"], reverse=True)
            ap_sorted = sorted(infos, key=lambda x: x["ap_score"], reverse=True)
            if up_sorted[0]["up_score"] > 0.45 and up_sorted[0]["up_score"] - up_sorted[1]["up_score"] > 0.20:
                assignment = {labels["B1"]: up_sorted[0], labels["B23"]: up_sorted[1]}
            elif ap_sorted[0]["ap_score"] > 0.35 and ap_sorted[0]["ap_score"] - ap_sorted[1]["ap_score"] > 0.20:
                assignment = {labels["B3"]: ap_sorted[0], labels["B12"]: ap_sorted[1]}
            elif ap_sorted[1]["ap_score"] < -0.35 and ap_sorted[0]["ap_score"] - ap_sorted[1]["ap_score"] > 0.20:
                assignment = {labels["B2"]: ap_sorted[1], labels["B13"]: ap_sorted[0]}
            else:
                assignment = {labels["B1"]: up_sorted[0], labels["B23"]: up_sorted[1]}
        else:
            return {}
        return {label: info["nodes"] for label, info in assignment.items()}

    def classify_b45(self, root_node: int, parent_node: int, side_prefix: str) -> Dict[str, Set[int]]:
        infos = self.distal_subtrees(root_node, parent_node, keep=2)
        if len(infos) >= 2:
            ap_sorted = sorted(infos, key=lambda x: x["ap_score"])
            assignment = {f"{side_prefix}_B4": ap_sorted[0], f"{side_prefix}_B5": ap_sorted[-1]}
        elif len(infos) == 1:
            assignment = {f"{side_prefix}_B4+B5": infos[0]}
        else:
            return {}
        return {label: info["nodes"] for label, info in assignment.items()}

    def trunk_with_distal_extension(self, prox_node: int, root_node: int) -> List[int]:
        base_path = self.shortest_path(prox_node, root_node) or []
        seeds = [n for n in self.adj[root_node] if n not in base_path and self.dist[n] > self.dist[root_node]]
        distal_nodes: List[int] = []
        if seeds:
            comps = [(s, self.trace_component(s, blocked=set(base_path))) for s in seeds]
            _, trunk_nodes = max(comps, key=lambda x: len(x[1]))
            terminal = max(trunk_nodes, key=lambda x: self.dist[x])
            distal_nodes = self.shortest_path(root_node, terminal) or []
        return list(dict.fromkeys(base_path + distal_nodes))

    @staticmethod
    def merge_used_nodes(segment_nodes: Dict[int, Set[int]]) -> Set[int]:
        used: Set[int] = set()
        for nodes in segment_nodes.values():
            used |= set(nodes)
        return used

    @staticmethod
    def add_named_nodes(segment_nodes: Dict[int, Set[int]], node_map: Dict[str, Set[int]]) -> None:
        for label_name, nodes in node_map.items():
            bid = NAME_TO_BID.get(label_name)
            if bid is not None and nodes:
                segment_nodes[bid] = set(nodes)

    @staticmethod
    def merge_trunk_extension(
        segment_nodes: Dict[int, Set[int]],
        target_bid: int,
        node_map: Dict[str, Set[int]],
        preferred_keys: Sequence[str],
    ) -> None:
        for key in preferred_keys:
            nodes = node_map.get(key)
            if nodes:
                segment_nodes.setdefault(target_bid, set()).update(nodes)
                return

    def classify_lower_lobe(
        self,
        side_prefix: str,
        prox_node: int,
        lower_node: int,
        already_used_nodes: Set[int],
        terminal_sort_key: str,
    ) -> Dict[str, Set[int]]:
        node_map: Dict[str, Set[int]] = {}
        trunk = self.trunk_with_distal_extension(prox_node, lower_node)
        middle = self.collect_side_subtrees(trunk, extra_blocked=already_used_nodes)
        if not middle:
            return node_map

        middle_by_z = sorted(middle, key=lambda x: x["z_score"], reverse=True)
        b6_infos = [middle_by_z[0]]
        if len(middle_by_z) >= 2:
            z_gap = abs(middle_by_z[0]["z_score"] - middle_by_z[1]["z_score"])
            if z_gap < 25 and middle_by_z[1]["ap_score"] < 0:
                b6_infos.append(middle_by_z[1])

        node_map[f"{side_prefix}_B6"] = set().union(*(x["nodes"] for x in b6_infos))
        used_seeds = {x["seed"] for x in b6_infos}
        remaining = [x for x in middle_by_z if x["seed"] not in used_seeds]

        b8_info = None
        if side_prefix == "Rt":
            if len(remaining) >= 2:
                assignments = [("Rt_B7", remaining[0]), ("Rt_B8", remaining[1])]
            elif len(remaining) == 1:
                one = remaining[0]
                assignments = [("Rt_B7" if one["ap_score"] > abs(one["x_score"]) else "Rt_B8", one)]
            else:
                assignments = []
        else:
            assignments = [("Lt_B8", remaining[0])] if remaining else []

        for label, info in assignments:
            node_map[label] = info["nodes"]
            if label.endswith("B8"):
                b8_info = info

        if b8_info is not None:
            self.classify_terminal_b9b10(
                side_prefix=side_prefix,
                prox_node=prox_node,
                lower_node=lower_node,
                b8_info=b8_info,
                already_used_nodes=already_used_nodes,
                current_node_map=node_map,
                node_map=node_map,
                sort_key=terminal_sort_key,
            )
        return node_map

    def classify_terminal_b9b10(
        self,
        side_prefix: str,
        prox_node: int,
        lower_node: int,
        b8_info: dict,
        already_used_nodes: Set[int],
        current_node_map: Dict[str, Set[int]],
        node_map: Dict[str, Set[int]],
        sort_key: str,
    ) -> None:
        b8_attach = b8_info["attach"]
        trunk_to_b8 = self.shortest_path(lower_node, b8_attach)
        if trunk_to_b8:
            node_map[f"{side_prefix}_trunk_to_B8_attach"] = set(trunk_to_b8)

        upstream = set(self.shortest_path(prox_node, b8_attach) or [])
        blocked = upstream | set(already_used_nodes)
        for nodes in current_node_map.values():
            blocked |= set(nodes)

        comps = []
        for nb in self.adj[b8_attach]:
            if nb in blocked:
                continue
            nodes = self.trace_component(nb, blocked=blocked)
            if len(nodes) >= self.min_subtree_size:
                comps.append(self.direction_info(nodes, b8_attach, seed=nb, attach=b8_attach))
        if not comps:
            return

        territory_info = max(comps, key=lambda x: x["size"])
        territory = set(territory_info["nodes"])
        branch_cands = [n for n in territory if sum(1 for nb in self.adj[n] if nb in territory) >= 3]

        if not branch_cands:
            node_map[f"{side_prefix}_B9+B10"] = territory
            return

        split_node = min(branch_cands, key=lambda x: self.dist[x])
        trunk_to_split = self.shortest_path(lower_node, split_node)
        if trunk_to_split and side_prefix == "Rt":
            node_map[f"{side_prefix}_trunk_to_B9B10_split"] = set(trunk_to_split)

        path_to_split = self.shortest_path(b8_attach, split_node) or []
        upstream_prev = path_to_split[-2] if len(path_to_split) >= 2 else None
        allowed = territory - {split_node}

        terminal = []
        for nb in self.adj[split_node]:
            if nb == upstream_prev or nb not in territory:
                continue
            nodes = self.trace_component(nb, allowed=allowed)
            if len(nodes) >= self.min_subtree_size:
                terminal.append(self.direction_info(nodes, split_node, seed=nb, attach=split_node))

        if len(terminal) >= 2:
            sorted_terms = sorted(terminal, key=lambda x: x[sort_key])
            if side_prefix == "Rt":
                b9, b10 = sorted_terms[0], sorted_terms[-1]
            else:
                b10, b9 = sorted_terms[0], sorted_terms[-1]
            node_map[f"{side_prefix}_B9"] = b9["nodes"]
            node_map[f"{side_prefix}_B10"] = b10["nodes"]
        elif len(terminal) == 1:
            node_map[f"{side_prefix}_B9+B10"] = terminal[0]["nodes"]
        else:
            node_map[f"{side_prefix}_B9+B10"] = territory_info["nodes"]

    def _discover_segments(self) -> Dict[int, Set[int]]:
        l2_a, l2_b = self._select_main_neighbors()

        l3_a, l3_b = self.first_branch(l2_a, self.carina), self.first_branch(l2_b, self.carina)
        if l3_a is None or l3_b is None:
            raise TopologyStructureError("MAIN_BRANCHES_UNRESOLVED", "Main bronchial branching nodes are unresolved in the observed skeleton.")

        lateral = self.verts[[l3_a, l3_b], 0] - self.verts[self.carina, 0]
        if min(lateral) >= -1e-6 or max(lateral) <= 1e-6:
            raise TopologyStructureError("LEFT_RIGHT_UNRESOLVED", "The observed proximal limbs do not establish distinct left/right territories in RAS.")

        if np.dot(self.verts[l3_a] - self.verts[self.carina], self.pc2) < np.dot(self.verts[l3_b] - self.verts[self.carina], self.pc2):
            lmb_l3, rmb_l3, lmb_l2, rmb_l2 = l3_a, l3_b, l2_a, l2_b
        else:
            lmb_l3, rmb_l3, lmb_l2, rmb_l2 = l3_b, l3_a, l2_b, l2_a

        raw_lul, raw_lll = self.split_by_up(self.branch_children(lmb_l3, lmb_l2), lmb_l3)
        raw_rul, raw_bi = self.split_by_up(self.branch_children(rmb_l3, rmb_l2), rmb_l3)

        lmb_node, rmb_node = lmb_l3, rmb_l3
        rul_node, bi_node = raw_rul, raw_bi
        lul_node, lll_node = raw_lul, raw_lll

        if None in [rul_node, bi_node, lul_node, lll_node]:
            raise TopologyStructureError("LOBAR_BRANCHES_UNRESOLVED", "Lobar branching nodes are not all observed; use neutral topology labels.")

        rul_map = self.classify_b123(rul_node, rmb_node, "Rt")
        rml, rll = self.split_by_up(self.branch_children(bi_node, rmb_node), bi_node)
        udb, lin = self.split_by_up(self.branch_children(lul_node, lmb_node), lul_node)

        udb_map = self.classify_b123(udb, lul_node, "Lt") if udb is not None else {}
        rml_map = self.classify_b45(rml, bi_node, "Rt") if rml is not None else {}
        lin_map = self.classify_b45(lin, lul_node, "Lt") if lin is not None else {}

        segment_nodes: Dict[int, Set[int]] = {0: set(self.trace_trachea())}
        base_paths = {
            1: (self.carina, lmb_node),
            2: (self.carina, rmb_node),
            3: (rmb_node, rul_node),
            4: (rmb_node, bi_node),
            5: (lmb_node, lul_node),
            6: (lmb_node, lll_node),
            10: (bi_node, rml),
            11: (bi_node, rll),
            12: (lul_node, udb),
            13: (lul_node, lin),
        }

        for bid, (start, end) in base_paths.items():
            path = self.shortest_path(start, end)
            if path:
                segment_nodes[bid] = set(path)

        for node_map in (rul_map, udb_map, rml_map, lin_map):
            self.add_named_nodes(segment_nodes, node_map)

        if rll is not None:
            rll_map = self.classify_lower_lobe(
                side_prefix="Rt",
                prox_node=bi_node,
                lower_node=rll,
                already_used_nodes=self.merge_used_nodes(segment_nodes),
                terminal_sort_key="x_score",
            )
            self.merge_trunk_extension(segment_nodes, 11, rll_map, ["Rt_trunk_to_B9B10_split", "Rt_trunk_to_B8_attach"])
            self.add_named_nodes(segment_nodes, rll_map)

        if lll_node is not None:
            lll_map = self.classify_lower_lobe(
                side_prefix="Lt",
                prox_node=lmb_node,
                lower_node=lll_node,
                already_used_nodes=self.merge_used_nodes(segment_nodes),
                terminal_sort_key="ap_score",
            )
            self.merge_trunk_extension(segment_nodes, 6, lll_map, ["Lt_trunk_to_B8_attach"])
            self.add_named_nodes(segment_nodes, lll_map)

        return {bid: nodes for bid, nodes in segment_nodes.items() if bid in LABELS and nodes}


def load_airway_mask(input_path: Path, airway_value: int = 1) -> Tuple[np.ndarray, dict]:
    data, header = nrrd.read(str(input_path))
    if data.ndim != 3:
        raise ValueError(
            f"Expected one 3D airway labelmap, got shape {data.shape}. "
            "Extract a single 3D mask from multi-layer/4D Slicer segmentations first."
        )
    if not np.issubdtype(data.dtype, np.number) and data.dtype != np.bool_:
        raise ValueError(f"Airway labelmap must be numeric or boolean, got {data.dtype}.")
    if not np.isfinite(data).all():
        raise ValueError("Airway labelmap contains non-finite voxel values.")
    mask = data == airway_value
    if not mask.any():
        raise ValueError(f"No airway voxels equal to --airway-value {airway_value} were found.")
    return mask, header


def _unit_scale_to_mm(unit: str) -> float:
    unit = unit.strip().lower()
    scales = {
        "mm": 1.0,
        "millimeter": 1.0,
        "millimeters": 1.0,
        "millimetre": 1.0,
        "millimetres": 1.0,
        "cm": 10.0,
        "centimeter": 10.0,
        "centimeters": 10.0,
        "centimetre": 10.0,
        "centimetres": 10.0,
        "m": 1000.0,
        "meter": 1000.0,
        "meters": 1000.0,
        "metre": 1000.0,
        "metres": 1000.0,
        "um": 0.001,
        "µm": 0.001,
        "micron": 0.001,
        "microns": 0.001,
        "micrometer": 0.001,
        "micrometers": 0.001,
        "micrometre": 0.001,
        "micrometres": 0.001,
    }
    try:
        return scales[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported or unknown spatial unit {unit!r}; use a length unit.") from exc


def _validate_orthogonal_axes(axis_vectors: np.ndarray, description: str) -> None:
    if axis_vectors.shape != (3, 3):
        raise ValueError(f"{description} must contain exactly three spatial axes.")
    lengths = np.linalg.norm(axis_vectors, axis=1)
    if not np.isfinite(axis_vectors).all() or np.any(lengths <= 0):
        raise ValueError(f"{description} has invalid or zero-length spatial axes.")
    unit_vectors = axis_vectors / lengths[:, None]
    if not np.allclose(unit_vectors @ unit_vectors.T, np.eye(3), atol=1e-4, rtol=0):
        raise ValueError(
            f"{description} contains non-orthogonal/sheared axes, which this workflow does not support."
        )


def nrrd_geometry_ras_mm(header: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return NRRD origin and per-array-axis vectors in RAS millimetres."""
    space_signs = {
        "right-anterior-superior": np.array([1.0, 1.0, 1.0]),
        "left-posterior-superior": np.array([-1.0, -1.0, 1.0]),
        "left-anterior-superior": np.array([-1.0, 1.0, 1.0]),
        "right-posterior-superior": np.array([1.0, -1.0, 1.0]),
    }
    space = str(header.get("space", "")).strip().lower()
    if space not in space_signs:
        raise ValueError(
            f"Unsupported or missing NRRD space {space!r}; expected an explicit RAS/LPS-family space."
        )
    try:
        directions = np.asarray(header["space directions"], dtype=float)
        origin = np.asarray(header["space origin"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Input NRRD must include 3D space directions and a space origin.") from exc
    if directions.shape != (3, 3) or origin.shape != (3,):
        raise ValueError("Input NRRD must have exactly three spatial directions and a 3D origin.")

    units = header.get("space units")
    # NRRD commonly omits space units for medical volumes; the pipeline's
    # coordinate contract is millimetres in that case.
    if units is None:
        scale = 1.0
    else:
        units = list(units)
        if len(units) != 3:
            raise ValueError("Input NRRD space units must contain three entries.")
        unit_scales = np.array([_unit_scale_to_mm(str(unit).strip('"')) for unit in units])
        if not np.allclose(unit_scales, unit_scales[0], atol=0, rtol=0):
            raise ValueError("Mixed spatial units across NRRD axes are not supported.")
        scale = float(unit_scales[0])

    signs = space_signs[space]
    origin_ras_mm = origin * signs * scale
    directions_ras_mm = directions * signs[None, :] * scale
    _validate_orthogonal_axes(directions_ras_mm, "Input NRRD geometry")
    return origin_ras_mm, directions_ras_mm


def validate_ct_geometry(
    mask_shape: Sequence[int],
    input_header: dict,
    ct_path: Optional[Path],
) -> Optional[np.ndarray]:
    """Validate exact input-mask/CT voxel correspondence; never resample silently."""
    source_origin_ras_mm, source_directions_ras_mm = nrrd_geometry_ras_mm(input_header)
    if ct_path is None:
        return None

    ct_img = nib.load(str(ct_path))
    if len(ct_img.shape) != 3:
        raise ValueError(f"Reference CT must be a 3D NIfTI volume, got shape {ct_img.shape}.")
    if tuple(ct_img.shape) != tuple(mask_shape):
        raise ValueError(
            f"Input mask shape {tuple(mask_shape)} does not match CT shape {tuple(ct_img.shape)}; "
            "resampling is not performed."
        )
    spatial_unit = ct_img.header.get_xyzt_units()[0]
    ct_scale = _unit_scale_to_mm(spatial_unit) if spatial_unit else None
    if ct_scale is None:
        raise ValueError("Reference CT must declare its spatial units; unknown units cannot be matched safely.")

    affine_ras_mm = np.asarray(ct_img.affine, dtype=float).copy()
    if affine_ras_mm.shape != (4, 4) or not np.isfinite(affine_ras_mm).all():
        raise ValueError("Reference CT has an invalid voxel-to-world affine.")
    affine_ras_mm[:3, :] *= ct_scale
    ct_directions_ras_mm = affine_ras_mm[:3, :3].T
    _validate_orthogonal_axes(ct_directions_ras_mm, "Reference CT geometry")
    if not np.allclose(
        source_directions_ras_mm,
        ct_directions_ras_mm,
        atol=1e-3,
        rtol=1e-5,
    ) or not np.allclose(
        source_origin_ras_mm,
        affine_ras_mm[:3, 3],
        atol=1e-3,
        rtol=1e-5,
    ):
        raise ValueError(
            "Input mask and CT do not share the same RAS voxel geometry; "
            "the tool will not copy CT geometry onto a differently aligned mask."
        )
    return affine_ras_mm


def build_export_items(segment_nodes: Dict[int, Set[int]], export_bids: Sequence[int]) -> List[Tuple[int, str]]:
    return [(bid, LABELS[bid][0]) for bid in export_bids if bid in segment_nodes and len(segment_nodes[bid]) > 0]


def assign_voxel_labels(
    mask: np.ndarray,
    verts: np.ndarray,
    segment_nodes: Dict[int, Set[int]],
    export_items: Sequence[Tuple[int, str]],
    voxel_directions_ras_mm: np.ndarray,
) -> np.ndarray:
    all_pts, all_lbl = [], []
    if len(export_items) > np.iinfo(np.uint16).max:
        raise ValueError("Too many observed branches for a uint16 labelmap.")
    for export_label, (bid, _) in enumerate(export_items, start=1):
        pts = verts[list(segment_nodes[bid])]
        all_pts.append(pts)
        all_lbl.append(np.full(len(pts), export_label, dtype=np.uint16))

    if not all_pts:
        raise RuntimeError("No valid segment nodes are available for export.")

    points, first = np.unique(np.vstack(all_pts), axis=0, return_index=True)
    # Shared junction nodes go to the first exported segment, deterministically.
    # Lexicographic physical coordinates also stabilize equal-distance queries.
    tree = cKDTree(points)
    labels = np.concatenate(all_lbl)[first]
    voxels = np.argwhere(mask)
    labelmap = np.zeros(mask.shape, dtype=np.uint16)
    voxels_ras_mm = voxels @ voxel_directions_ras_mm
    _, nearest = tree.query(voxels_ras_mm, k=1)
    labelmap[tuple(voxels.T)] = labels[nearest]
    return labelmap


def make_slicer_segmentation(
    labelmap: np.ndarray,
    export_items: Sequence[Tuple[int, str]],
    ct_affine_ras_mm: Optional[np.ndarray] = None,
    fallback_header: Optional[dict] = None,
) -> Tuple[np.ndarray, dict, List[Tuple[int, str]]]:
    segment_arrays = []
    segment_meta = []

    for export_label, (bid, name) in enumerate(export_items, start=1):
        seg = (labelmap == export_label).astype(np.uint8)
        if seg.sum() > 0:
            segment_arrays.append(seg)
            segment_meta.append((bid, name))

    if not segment_arrays:
        raise RuntimeError("All exported segments are empty.")

    seg_data = np.stack(segment_arrays, axis=0)
    header = {
        "type": "uint8",
        "dimension": 4,
        "sizes": np.array(seg_data.shape, dtype=np.int64),
        "kinds": ["list", "domain", "domain", "domain"],
        "encoding": "gzip",
        "Segmentation_MasterRepresentation": "Binary labelmap",
        "Segmentation_ContainedRepresentationNames": "Binary labelmap",
        "space units": ["mm", "mm", "mm"],
    }

    if ct_affine_ras_mm is not None:
        affine = ct_affine_ras_mm
        header["space"] = "right-anterior-superior"
        header["space origin"] = affine[:3, 3].astype(float)
        header["space directions"] = np.vstack([
            [np.nan, np.nan, np.nan],
            affine[:3, 0],
            affine[:3, 1],
            affine[:3, 2],
        ]).astype(float)
    elif fallback_header is not None:
        origin_ras_mm, directions_ras_mm = nrrd_geometry_ras_mm(fallback_header)
        header["space"] = "right-anterior-superior"
        header["space origin"] = origin_ras_mm
        header["space directions"] = np.vstack(
            (np.full((1, 3), np.nan), directions_ras_mm)
        )

    for i, (bid, name) in enumerate(segment_meta):
        color = branch_color(bid)
        nz = np.argwhere(segment_arrays[i] > 0)
        xmin, ymin, zmin = nz.min(axis=0)
        xmax, ymax, zmax = nz.max(axis=0)

        header[f"Segment{i}_ID"] = f"Segment_{i + 1}"
        header[f"Segment{i}_Name"] = name
        header[f"Segment{i}_NameAutoGenerated"] = "0"
        header[f"Segment{i}_Color"] = f"{color[0]} {color[1]} {color[2]}"
        header[f"Segment{i}_ColorAutoGenerated"] = "0"
        header[f"Segment{i}_Layer"] = str(i)
        header[f"Segment{i}_LabelValue"] = "1"
        header[f"Segment{i}_Extent"] = f"{xmin} {xmax} {ymin} {ymax} {zmin} {zmax}"

    return seg_data, header, segment_meta


def write_segment_table(txt_path: Path, segment_meta: Sequence[Tuple[int, str]]) -> None:
    lines = ["=== FINAL SEGMENTS ==="]
    for i, (bid, name) in enumerate(segment_meta, start=1):
        lines.append(f"{i:2d}. {name} (bid={bid})")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_segment_table(segment_meta: Sequence[Tuple[int, str]]) -> None:
    print("\n=== FINAL SEGMENTS ===")
    for i, (bid, name) in enumerate(segment_meta, start=1):
        print(f"{i:2d}. {name} (bid={bid})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label bronchial-tree branches from an airway .nrrd / .seg.nrrd mask."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input airway .nrrd or .seg.nrrd file.")
    parser.add_argument("--output", type=Path, default=None, help="Output Slicer .seg.nrrd path.")
    parser.add_argument("--txt", type=Path, default=None, help="Output text table path.")
    parser.add_argument("--ct", type=Path, default=None, help="Reference CT .nii/.nii.gz for geometry metadata.")
    parser.add_argument("--airway-value", type=int, default=1, help="Voxel value of the airway mask. Default: 1.")
    parser.add_argument("--ap-sign", type=float, default=1.0, choices=[-1.0, 1.0], help="Use -1 if AP assignment is reversed.")
    parser.add_argument("--min-subtree-size", type=int, default=5, help="Minimum skeleton component size for branch classification.")
    parser.add_argument("--strict-anatomy", action="store_true", help="Fail instead of exporting neutral topology when anatomical naming is unresolved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path.with_name(input_path.stem.replace(".seg", "") + "_BRANCHES.seg.nrrd")
    txt_path = args.txt or output_path.with_suffix(".txt")
    print("The entire process may take around 2-3 minutes depending on the airway complexity and hardware. Please wait until completion.")

    mask, input_header = load_airway_mask(input_path, airway_value=args.airway_value)
    _, voxel_directions_ras_mm = nrrd_geometry_ras_mm(input_header)
    ct_affine_ras_mm = validate_ct_geometry(mask.shape, input_header, args.ct)

    protected_inputs = {input_path.resolve()}
    if args.ct is not None:
        protected_inputs.add(args.ct.resolve())
    resolved_outputs = {output_path.resolve(), txt_path.resolve()}
    if len(resolved_outputs) != 2 or resolved_outputs & protected_inputs:
        raise ValueError("Output paths must be distinct from each other, the input mask, and the reference CT.")

    labeler = BronchialBranchLabeler(
        mask,
        ap_sign=args.ap_sign,
        min_subtree_size=args.min_subtree_size,
        voxel_directions_ras_mm=voxel_directions_ras_mm,
        strict_anatomy=args.strict_anatomy,
    )
    segment_nodes = labeler.run()
    print("Parsing mode:", labeler.parsing_mode)
    if labeler.parsing_mode == "TOPOLOGY_ONLY":
        print("Anatomical naming unresolved:", labeler.diagnostics.get("anatomy_unresolved_reason", ""))

    export_items = labeler.export_items(segment_nodes)
    labelmap = assign_voxel_labels(
        mask,
        labeler.verts,
        segment_nodes,
        export_items,
        voxel_directions_ras_mm,
    )
    seg_data, header, segment_meta = make_slicer_segmentation(
        labelmap,
        export_items,
        ct_affine_ras_mm=ct_affine_ras_mm,
        fallback_header=input_header,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(output_path), seg_data, header)
    write_segment_table(txt_path, segment_meta)
    print_segment_table(segment_meta)
    print(f"\nSaved NRRD: {output_path}")
    print(f"Saved TXT : {txt_path}")


if __name__ == "__main__":
    main()
