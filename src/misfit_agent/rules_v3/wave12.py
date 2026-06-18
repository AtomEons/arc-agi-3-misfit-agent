"""Wave 12 — per-task object-correspondence program synthesizer.

This is a fundamentally different inference engine from Waves 4-11.

Waves 4-11 each contribute ONE global rule with parameters fitted on all
train pairs simultaneously. The 980 / 1000 fit-contract bottleneck shows
that's too restrictive: most tasks have a per-OBJECT logic, not a global
parameter.

Wave 12 fits a per-task PROGRAM whose structure is:

  for each input object o:
    classify(o) -> class_id (small integer)
    transform(o, class_id) -> placed_object | dropped

The classifier and per-class transforms are discovered from the train pairs:

  1. Per train pair, extract objects from input and output.
  2. Greedy-match input objects to output objects by signature similarity.
  3. For each matched pair, record a transform record.
  4. Cluster transforms by an OBJECT-CLASS signature (size, color, shape
     hash, position class). Classes must yield IDENTICAL transforms across
     all train pairs (Tier-1 honest — no soft consistency).
  5. The program is the discovered (classifier, per-class transform) tuple.
  6. predict(test_input):
       extract objects; for each, classify and apply its transform; render.

Tier-1 strict throughout:
  - No LLM, no pretrained weights.
  - Classifier is a deterministic function of object features.
  - Transforms are enumerated atoms: identity, color_remap, shift,
    delete, swap_colors, scale, paint_at_anchor.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..perceptor import _background_color


Grid = np.ndarray


# ---------------------------------------------------------------------------
# Object extraction with the perceptor.
# We keep the (mask, color, bbox, area) tuple instead of the dataclass so we
# can mutate/cluster more freely. Mask is a 2-D bool array sized to the grid.
# ---------------------------------------------------------------------------


def _flood_components(grid: Grid, bg: int) -> list[tuple[np.ndarray, int, tuple[int, int, int, int]]]:
    """Return list of (mask, color, bbox) for each 4-connected non-bg blob."""
    rows, cols = grid.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    nxt = 0
    out = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == bg or labels[r, c] != 0:
                continue
            nxt += 1
            stack = [(r, c)]
            target = int(grid[r, c])
            cells = []
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                if labels[rr, cc] != 0:
                    continue
                if int(grid[rr, cc]) != target:
                    continue
                labels[rr, cc] = nxt
                cells.append((rr, cc))
                stack.extend([(rr+1, cc), (rr-1, cc), (rr, cc+1), (rr, cc-1)])
            if not cells:
                continue
            mask = np.zeros_like(grid, dtype=bool)
            for rr, cc in cells:
                mask[rr, cc] = True
            ys, xs = np.where(mask)
            bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
            out.append((mask, target, bbox))
    return out


def _normalized_shape_key(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple:
    """Hash of the shape (mask cropped to bbox, ignoring color & position)."""
    r0, c0, r1, c1 = bbox
    sub = mask[r0:r1+1, c0:c1+1].astype(np.uint8)
    return (sub.shape, sub.tobytes())


# ---------------------------------------------------------------------------
# Object descriptors used by the classifier and transform inference.
# ---------------------------------------------------------------------------


@dataclass
class ObjDesc:
    color: int
    area: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray  # full-grid bool
    shape_key: tuple

    @property
    def shape_hw(self) -> tuple[int, int]:
        r0, c0, r1, c1 = self.bbox
        return (r1 - r0 + 1, c1 - c0 + 1)

    @property
    def centroid(self) -> tuple[float, float]:
        ys, xs = np.where(self.mask)
        return (float(ys.mean()), float(xs.mean()))


def _describe(grid: Grid) -> list[ObjDesc]:
    bg = _background_color(grid)
    raw = _flood_components(grid, bg)
    out = []
    for mask, color, bbox in raw:
        out.append(ObjDesc(
            color=color, area=int(mask.sum()), bbox=bbox, mask=mask,
            shape_key=_normalized_shape_key(mask, bbox),
        ))
    # sort by area descending
    out.sort(key=lambda o: (-o.area, o.bbox))
    return out


# ---------------------------------------------------------------------------
# Object class signatures.
#
# The CLASSIFIER is a deterministic function of object features that yields a
# small-integer class id. We try several class-signature schemes and pick the
# coarsest one that yields a consistent transform-per-class across all train
# pairs.
# ---------------------------------------------------------------------------


def _class_by_shape(o: ObjDesc) -> tuple:
    return ("shape", o.shape_key)


def _class_by_area_rank(o: ObjDesc, descs: list[ObjDesc]) -> tuple:
    areas = sorted({d.area for d in descs}, reverse=True)
    rank = areas.index(o.area)
    return ("area_rank", rank)


def _class_by_color(o: ObjDesc) -> tuple:
    return ("color", o.color)


def _class_by_touches_edge(o: ObjDesc, grid_shape: tuple[int, int]) -> tuple:
    rows, cols = grid_shape
    r0, c0, r1, c1 = o.bbox
    return ("touches_edge", r0 == 0 or c0 == 0 or r1 == rows-1 or c1 == cols-1)


def _class_by_singleton(o: ObjDesc, descs: list[ObjDesc]) -> tuple:
    # "the only one of its color"
    counter = Counter(d.color for d in descs)
    return ("unique_color", counter[o.color] == 1)


CLASSIFIER_NAMES = [
    "shape",
    "area_rank",
    "color",
    "touches_edge",
    "unique_color",
]


def _classify(o: ObjDesc, descs: list[ObjDesc], grid_shape: tuple[int, int],
              scheme: str) -> tuple:
    if scheme == "shape":
        return _class_by_shape(o)
    if scheme == "area_rank":
        return _class_by_area_rank(o, descs)
    if scheme == "color":
        return _class_by_color(o)
    if scheme == "touches_edge":
        return _class_by_touches_edge(o, grid_shape)
    if scheme == "unique_color":
        return _class_by_singleton(o, descs)
    raise ValueError(scheme)


# ---------------------------------------------------------------------------
# Atomic per-object transforms.
#
# Each transform is a callable (obj, grid_shape) -> placed_object | None.
# A "placed object" is a (target_color, target_mask) tuple where target_mask
# is sized to the OUTPUT grid (we assume same shape as input for v1).
#
# Transform parameters are inferred from the matched (in_obj, out_obj)
# pairs. We enumerate a small atom set; for each candidate transform we
# check whether it is the SAME function across every (in, out) pair of
# its discovered class.
# ---------------------------------------------------------------------------


@dataclass
class TransformAtom:
    kind: str  # identity / recolor / shift / delete / paint_at
    params: dict = field(default_factory=dict)

    def apply(self, obj: ObjDesc, grid_shape: tuple[int, int]
              ) -> Optional[tuple[int, np.ndarray]]:
        if self.kind == "delete":
            return None
        rows, cols = grid_shape
        if self.kind == "identity":
            return obj.color, obj.mask.copy()
        if self.kind == "recolor":
            return self.params["dst"], obj.mask.copy()
        if self.kind == "shift":
            dy, dx = self.params["dy"], self.params["dx"]
            new_mask = np.zeros_like(obj.mask)
            ys, xs = np.where(obj.mask)
            for y, x in zip(ys, xs):
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    new_mask[ny, nx] = True
            return obj.color, new_mask
        if self.kind == "shift_recolor":
            dy, dx = self.params["dy"], self.params["dx"]
            new_mask = np.zeros_like(obj.mask)
            ys, xs = np.where(obj.mask)
            for y, x in zip(ys, xs):
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    new_mask[ny, nx] = True
            return self.params["dst"], new_mask
        return None


def _match_objects(in_descs: list[ObjDesc], out_descs: list[ObjDesc]
                   ) -> list[tuple[ObjDesc, Optional[ObjDesc]]]:
    """Greedy match input objects to output objects by signature."""
    pairs = []
    used_out = set()
    # 1) exact shape_key match preferring same color
    for io in in_descs:
        best = None
        best_score = (-1, -1)
        for j, oo in enumerate(out_descs):
            if j in used_out:
                continue
            shape_match = int(io.shape_key == oo.shape_key)
            color_match = int(io.color == oo.color)
            score = (shape_match, color_match)
            if score > best_score:
                best_score = score
                best = j
        if best is not None and best_score[0] == 1:
            pairs.append((io, out_descs[best]))
            used_out.add(best)
            continue
        # 2) fallback to same-color match
        best = None
        for j, oo in enumerate(out_descs):
            if j in used_out:
                continue
            if io.color == oo.color and abs(io.area - oo.area) <= max(1, io.area // 2):
                best = j
                break
        if best is not None:
            pairs.append((io, out_descs[best]))
            used_out.add(best)
        else:
            pairs.append((io, None))
    return pairs


def _infer_atom(io: ObjDesc, oo: Optional[ObjDesc]) -> Optional[TransformAtom]:
    if oo is None:
        return TransformAtom("delete")
    if io.shape_key == oo.shape_key and io.bbox == oo.bbox:
        if io.color == oo.color:
            return TransformAtom("identity")
        return TransformAtom("recolor", {"dst": oo.color})
    if io.shape_key == oo.shape_key:
        dy = oo.bbox[0] - io.bbox[0]
        dx = oo.bbox[1] - io.bbox[1]
        if io.color == oo.color:
            return TransformAtom("shift", {"dy": dy, "dx": dx})
        return TransformAtom("shift_recolor", {"dy": dy, "dx": dx, "dst": oo.color})
    return None


def _atom_key(atom: TransformAtom) -> tuple:
    return (atom.kind,) + tuple(sorted(atom.params.items()))


# ---------------------------------------------------------------------------
# The synthesizer rule.
# ---------------------------------------------------------------------------


@dataclass
class ObjectCorrespondenceProgram:
    """For each train pair: extract objects from input and output, match,
    infer per-object atoms. Group atoms by classifier-scheme class. Accept
    the coarsest classifier scheme under which all train pairs yield
    IDENTICAL (class -> atom) mappings.

    predict: extract objects from test input, classify, apply atoms, render.

    Tier-1 strict: deterministic enumeration over a small classifier set
    (shape, color, area_rank, touches_edge, unique_color) and a small
    transform-atom set (identity, recolor, shift, delete, shift_recolor).
    No learned parameters at eval. The "program" is just the (scheme,
    class_to_atom) tuple — completely interpretable.
    """
    scheme: str = "shape"
    class_to_atom: dict = field(default_factory=dict)
    background_keep: bool = True
    fitted: bool = False

    def _try_scheme(self, scheme: str, train_pairs: list[tuple[Grid, Grid]]
                    ) -> Optional[dict]:
        """Try to derive a class_to_atom mapping under `scheme` that is
        consistent across ALL train pairs. Return mapping or None."""
        bookkeeping: dict[tuple, str] = {}  # class -> atom_key (string)
        atom_objs: dict[tuple, TransformAtom] = {}
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return None
            in_descs = _describe(inp)
            out_descs = _describe(out)
            pairs = _match_objects(in_descs, out_descs)
            for io, oo in pairs:
                atom = _infer_atom(io, oo)
                if atom is None:
                    return None
                klass = _classify(io, in_descs, inp.shape, scheme)
                key = _atom_key(atom)
                if klass in bookkeeping:
                    if bookkeeping[klass] != key:
                        return None
                else:
                    bookkeeping[klass] = key
                    atom_objs[klass] = atom
        return atom_objs if bookkeeping else None

    def _render(self, grid: Grid) -> Grid:
        bg = _background_color(grid)
        out = np.full_like(grid, bg) if not self.background_keep else grid.copy()
        # If background_keep True, we *start* from the input grid but we
        # OVERWRITE only the cells we touch (placed objects). Deletes simply
        # blank the original mask back to bg.
        in_descs = _describe(grid)
        # Clear all object cells first (we'll redraw)
        for o in in_descs:
            out[o.mask] = bg
        for o in in_descs:
            klass = _classify(o, in_descs, grid.shape, self.scheme)
            atom = self.class_to_atom.get(klass)
            if atom is None:
                # No transform for this class -> keep identity
                out[o.mask] = o.color
                continue
            placed = atom.apply(o, grid.shape)
            if placed is None:
                continue
            color, mask = placed
            out[mask] = color
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for scheme in CLASSIFIER_NAMES:
            mapping = self._try_scheme(scheme, train_pairs)
            if mapping is None:
                continue
            # Check the program actually solves every train pair
            self.scheme = scheme
            self.class_to_atom = mapping
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                pred = self._render(inp)
                if pred.shape != out.shape or not np.array_equal(pred, out):
                    ok = False
                    break
            if ok:
                # Reject trivial all-identity programs (covered by Identity)
                only_identity = all(a.kind == "identity" for a in mapping.values())
                if only_identity:
                    continue
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._render(np.asarray(grid))

    def signature(self) -> tuple:
        return ("ObjectCorrespondenceProgram", self.scheme,
                tuple(sorted((str(k), _atom_key(v))
                             for k, v in self.class_to_atom.items())))


ALL_WAVE12_RULES = [ObjectCorrespondenceProgram]
