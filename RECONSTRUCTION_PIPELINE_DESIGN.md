# Reconstruction Pipeline Redesign — Hand-Drawn Sketch → Digital Floor Plan

**Scope:** full redesign of the reconstruction pipeline (model unchanged).
**Model:** YOLO11-seg, 7 classes: `room, door, wall, floor, stairs, loft_hatch, acm`
**Metrics:** box P≈0.90 / R≈0.85 / mAP50≈0.88 / mAP50-95≈0.68; mask P≈0.79 / R≈0.74 / mAP50≈0.72 / mAP50-95≈0.48
**Reading of the metrics:** boxes are ~16 points better than masks at mAP50 and ~20 points at mAP50-95. The model knows *where* things are far better than it knows their *exact outline*. The pipeline below is designed around exactly that asymmetry: use boxes and topology for structure, masks only where area/shape genuinely matters, and **never let a raw mask boundary reach the exported file**.

---

## 1. Per-Class Prediction Strategy (the decision matrix)

| Class | Primary signal | Secondary signal | Geometry extraction | Topology role | Export representation |
|---|---|---|---|---|---|
| **wall** | **MASK → SKELETON → CENTERLINE** | box (existence filter) | binarize → thin → vectorize → merge → snap | *the* backbone: every other element attaches to the wall graph | centerline polyline + thickness (DXF: polyline pair or solid hatch; VSDX: wall shapes) |
| **room** | **TOPOLOGY (wall-loop faces)** | box (seed/count/label anchor), mask (fallback) | enclosed faces of the planar wall graph | node in room-adjacency graph | closed LWPOLYLINE / Visio room shape |
| **door** | **BOX** | mask centroid | project box centre onto nearest wall centerline; width = box extent along wall | edge connecting the two faces sharing that wall | wall gap + swing-arc block/master |
| **floor** | **MASK** | — | outer envelope polygon (largest contour, hole-filled) | global constraint: all geometry ⊂ envelope; fills unlabeled space | building outline polyline (optional layer) |
| **stairs** | **MASK (locate only)** | box | min-area oriented rectangle + flight direction (long axis) | attached to containing/overlapping room face | **template stair symbol** (Visio master / DXF block) placed+scaled into the located rect — never the raw mask outline |
| **loft_hatch** | **BOX** | — | centroid → point symbol | attached to containing room | ⊠ symbol block/master at point |
| **acm** | **MASK** | box | polygon clipped to room faces; ≥60 % room coverage → whole-room ACM | attribute on room face(s) or sub-region child of a room | hatched polygon (red diagonal) / room fill attribute |

**Where I disagree with the suggested strategy (and why):**

1. **Rooms should NOT be primarily boxes.** A box cannot represent an L-shaped lounge, a corridor, or the notched loft in your own test sketches. But a room *mask* shouldn't be primary either — at mask mAP50-95 ≈ 0.48 the boundaries wobble, and two adjacent room masks never share an exact wall line, which destroys topology. The correct source of room geometry is the **wall graph**: rooms are the *enclosed faces* of the planar graph formed by wall centerlines. That is how Raster-to-Vector, FloorSP, MonteFloor, HEAT and CubiCasa-style systems all work. Room *boxes* are still used — as high-precision (0.90) evidence for "a room exists roughly here", to seed face labeling, and to catch faces the wall graph fails to close. Room *masks* are the fallback geometry when the wall graph is locally broken.
2. **ACM should be MASK, not BOX.** ACM in survey sketches is a free-form hatched wash that often covers part of a room or an odd L-shaped loft. A box over-claims asbestos area — commercially dangerous in an asbestos survey product. Use the mask, then regularize it against room faces (below).
3. **Doors as BOX — agreed**, with the mandatory constraint that a door is only *valid* when it lands on a wall centerline. A door that cannot be projected onto a wall within `k × wall_thickness` is rejected (no floating doors).
4. **Wall: MASK → skeleton → centerline.** Boxes are useless for walls (long, thin, sometimes diagonal; a box loses everything). Raw mask polygons give blobby double-edged outlines. Skeletonization + vectorization gives 1-px centerlines with thickness recovered from the distance transform — exactly the representation CAD needs.

---

## 2. Full Pipeline (stage by stage)

```
S0 Preprocess ─ S1 Inference ─ S2 Class routing ─ S3 Wall extraction ─ S4 Constraint solve
   ─ S5 Topology graph ─ S6 Room faces ─ S7 Attachments ─ S8 Semantics ─ S9 QA gate ─ S10 Export
```

### S0 — Preprocessing (before the model sees anything)
Hand-drawn input on graph paper, photographed at an angle, with shadows. Fix as much as possible *before* inference:
- **Page detection + perspective rectification:** detect the paper quad (largest light quadrilateral; cv2 contour + `getPerspectiveTransform`). Removes camera keystone.
- **Deskew / dominant-angle estimate:** Hough on long edges (or the graph-paper grid itself — it is a free calibration target); rotate so the dominant axis is horizontal. Store `page_rotation_deg` in metadata; all later "snap to 90°" happens in this rectified frame.
- **Illumination flattening:** divide by a large-kernel Gaussian blur (or CLAHE) to kill shadows.
- **Resolution policy:** letterbox to the training imgsz (keep 1280 — inference imgsz must equal training imgsz; do not follow generic "960–1024" advice, mismatch silently costs accuracy).
- Keep the *unrectified→rectified homography* so OCR/labels read from the original photo can be mapped into plan space.

### S1 — Inference (single pass, both heads)
- One YOLO11-seg forward pass: `conf` per-class thresholds (structural classes can afford 0.30+ because box precision is 0.90; stairs/loft_hatch lower, e.g. 0.20, they are rarer), `iou=0.5` NMS, **class-agnostic NMS OFF** (a door legitimately overlaps a wall and a room).
- Emit a `Detections` object: per instance `{class, conf, box, mask(optional), mask_quality}`. `mask_quality` = compactness + solidity score; used later to decide mask-vs-box fallbacks per instance, not per class.

### S2 — Class routing (the hybrid dispatcher)
Route every instance to its geometry extractor per the matrix in §1. This is a small, explicit, testable module — not an implicit if-chain inside one god-function.

### S3 — Wall extraction (the heart of the redesign)
1. **Union all wall masks** into one binary raster (walls are annotated as many instances; the union is the wall ink).
2. **Image-guided refinement:** dilate the union by ~1.5× the estimated stroke width and re-threshold the *source image* inside that band (adaptive threshold). The model's mask says "wall is roughly here"; the original ink says exactly where. This recovers crisp edges the 0.48 mask mAP50-95 can't give you.
3. **Morphological close** (kernel ≈ wall thickness) to seal pen gaps, then **open** to remove specks.
4. **Thickness field:** distance transform of the refined raster; median inside the mask ×2 = `wall_thickness` (global) plus a per-segment local value.
5. **Skeletonize:** Zhang–Suen thinning (`cv2.ximgproc.thinning`, fallback: scikit-image `skeletonize`, fallback: iterative morphological thinning in pure OpenCV) → 1-px centerline raster.
6. **Vectorize:** trace the skeleton pixel graph (8-connectivity): nodes = pixels with ≠2 neighbours (endpoints & junctions), edges = pixel chains between nodes. Douglas–Peucker each chain (ε ≈ 1.5 px).
7. **Line merging:** merge chains that are near-collinear (angle < 7°, gap < 2× thickness, lateral offset < thickness). Weighted least-squares refit per merged segment.
8. **Corner detection:** junction nodes from the skeleton graph are candidate corners; refine each by intersecting the two (or three) fitted segment lines analytically — corners become exact line intersections, not blurry pixel positions.

### S4 — Constraint solving (regularization)
Do this on the *vector* segments, in the rectified frame:
1. **Angle clustering:** histogram segment angles; snap every segment within ±12° of a dominant axis to exactly 0°/90° (or to the secondary dominant angle if the building genuinely has a diagonal wing — never force diagonals orthogonal).
2. **Coordinate clustering (snap):** cluster the x-coords of vertical segments and y-coords of horizontal segments (1-D agglomerative, tolerance ≈ wall thickness). Collinear walls across the whole plan land on shared coordinates → shared walls, clean corridors.
3. **Gap closing:** extend segment endpoints along their direction up to `3 × wall_thickness`; if the extension hits another segment, create the junction. Pen lifts and doorway gaps in the ink stop breaking room loops (real door gaps are re-opened in S7 from door instances — geometrically closed, semantically a door).
4. **Intersection fixing:** compute all pairwise segment intersections (sweep line if large); split segments at intersections so the graph is *planar* (edges meet only at nodes). Remove danglers shorter than `2 × thickness` unless a door/box supports them.
5. **Polygon simplification** happens implicitly: the wall graph is already minimal; faces extracted from it need no further Douglas-Peucker.

### S5 — Topology graph
Build a **half-edge structure (DCEL)** on the planar wall graph: nodes (corners), edges (wall runs, carrying thickness + supporting-instance ids), faces.

### S6 — Room polygon generation
1. Extract all bounded faces of the DCEL → candidate room polygons (closed, shared walls by construction).
2. **Face ↔ detection assignment:** score each face against room boxes (IoU box∩face / box area) and room masks. Faces matched to a room detection become rooms; the face polygon is the geometry (not the mask).
3. **Unmatched faces:** large ones inside the floor envelope become "unlabeled room" (review flag); slivers get merged into their largest neighbour.
4. **Unmatched room detections** (wall graph failed to enclose them): fallback order is **mask/contour first, box last** — room boxes alone produce the "big rectangles / wrong layout" failure mode; the mask polygon (orthogonalized + snapped to nearby wall lines) preserves L-shapes. Emit a `topology_incomplete` review flag either way.
5. **Floor envelope check:** every room face must lie inside the floor-mask envelope; clip protrusions.

### S7 — Attachments
- **Door–wall attachment:** project each door box centre onto the nearest DCEL edge (reject > `3 × thickness` away). Door span = box extent projected along the edge. Split the edge at the span → the gap; swing direction: into the room face whose label ranks lower in privacy heuristics (hall → room) or by mask asymmetry if available; record `connects=(face_a, face_b)` → this *is* the room-adjacency edge.
- **Stairs:** min-area rect of mask → orientation; flight direction = long axis, sign disambiguated by an arrow if the sketch has one (small line-detector inside the stair box) else default "up". Attach to the face containing the rect centre.
- **Loft hatch:** box centroid → point symbol; attach to containing face; if the plan has a loft floor, register a vertical connection loft ↔ landing.
- **ACM:** clip acm mask polygons to room faces. Coverage ≥ 60 % of a face → mark the whole face `acm=true` (survey convention); otherwise keep the clipped sub-polygon as a child region of that face. Both carry `confidence`.

### S8 — Semantics
- **Room labeling:** OCR inside each face (crop, mask out neighbours; PaddleOCR → EasyOCR → pytesseract fallback) and/or the existing GPT-4o label pass, merged via `apply_labels`-style hook (nearest-centroid matching, number/name split "005 Kitchen"). Auto-number unlabeled rooms per floor (`001…`), flagged for review.
- **Floor partitioning:** separate connected components of the wall graph (disjoint blocks on the page = separate floors, e.g. loft drawn beside the house) + label keywords; assign `floor_idx`, floor names.
- **Samples/annotations** (red-pen `S001 …`): carried as first-class annotation objects with resolved target room and ACM polarity.

### S9 — Quality gate
Machine-checkable validators, each → boolean + details: all room polygons simple & closed; doors on walls; adjacency graph connected per floor (or explained by no-door evidence); ACM regions inside rooms; face area ≈ mask area (|Δ|<25 %); OCR/label coverage. Aggregate score → below threshold: route to review / active-learning queue instead of export. **Gate reconstruction on geometry metrics, not on detection mAP** — mAP measures the model, not the floor plan. Task-level metrics (reviewer-agreed):

- **Room IoU / Wall IoU / Hausdorff** vs. manual ground truth.
- **Door placement rate:** % of ground-truth doors whose reconstructed gap lands on the correct wall within `k × thickness` — a door detection that floats or lands on the wrong wall counts as wrong even if its detection score is high.
- **ACM room-assignment accuracy:** % of ACM regions assigned to the correct room — mask-vs-sketch pixel overlap is NOT the success criterion; landing in the right room is.
- **Closed-loop rate:** % of rooms produced as closed wall-graph faces (vs. fallbacks).

### S10 — Export
All exporters consume one immutable `PlanDocument` (single source of truth). JSON (full graph + geometry + provenance), DXF (layers per class; walls as closed polyline pairs or centerline+thickness; doors as blocks with swing arcs; ACM as HATCH), VSDX (template-page-sized shapes, one tab per floor), plus PNG/SVG/PDF in the survey house style.

---

## 3. Why hybrid beats masks-only

1. **Masks don't share boundaries.** Two adjacent room masks overlap or leave slivers; every pair of rooms needs manual wall reconciliation. Faces of one wall graph share walls *by construction* — topology is correct before any cleanup.
2. **Your masks are the weak head.** mask mAP50-95 ≈ 0.48 means boundary jitter of several pixels everywhere. Boxes (0.68) and ink-refined centerlines are simply more accurate signals; use masks only where nothing else encodes shape (floor, stairs, acm).
3. **CAD wants primitives, not blobs.** DXF/Visio need lines, arcs, closed polylines, thickness. A skeleton centerline + thickness *is* the CAD wall; a 200-vertex mask contour is noise that still has to be converted.
4. **Constraints are only solvable on vectors.** Snap-to-90°, coordinate sharing, gap closing, and intersection fixing are well-posed on line segments, ill-posed on rasters. Regularization is where "hand-drawn" becomes "professional".
5. **Failure isolation.** When a mask is bad, the box still places the object; when the wall graph fails locally, the mask fallback still produces a room. Masks-only has a single point of failure; the hybrid degrades gracefully with review flags.
6. **Auditability (commercial requirement).** Every exported polygon carries provenance (`source: wall_graph | mask_fallback | box`), confidences and validator results — an asbestos-survey deliverable must be reviewable.

---

## 4. Repository architecture (production)

Evolves the current repo (keeps `RECONSTRUCTION_ENGINE` flag and the legacy path untouched; the existing `reconstruction/`, `exporters/`, `mlops/`, `api/`, `ui/`, `active_learning/`, `evaluation/` remain, restructured as below).

```
acorn/                                  # importable top-level package
├── config/
│   ├── settings.py                     # env-backed (keeps MODEL_IMGSZ=1280, MODEL_CONF_THRESHOLD, MODEL_IOU_NMS, RECONSTRUCTION_ENGINE)
│   ├── classes.yaml                    # id ↔ name ↔ routing (box/mask/skeleton) ↔ per-class conf
│   ├── pipeline.yaml                   # per-stage tolerances (angle_tol, snap_tol, gap_close_mult…)
│   └── style.yaml                      # export style (colors, hatches, fonts) — current acorn_style constants
├── perception/
│   ├── preprocess.py                   # rectify_page(), deskew(), flatten_illumination() → Homography
│   ├── detector.py                     # YoloSegDetector(imgsz, conf_per_class, iou) → Detections
│   ├── onnx_detector.py                # torch-free serving (existing onnx_infer.py)
│   └── types.py                        # Detections, Instance, MaskQuality
├── geometry/
│   ├── walls/
│   │   ├── raster.py                   # union, image-guided refine, morphology, thickness field
│   │   ├── skeleton.py                 # thin(), trace_pixel_graph() → PixelGraph
│   │   ├── vectorize.py                # chains → segments (DP), fit_segments()
│   │   └── merge.py                    # collinear merge, junction refit
│   ├── constraints.py                  # angle clustering, snap axes, coordinate clustering
│   ├── gaps.py                         # endpoint extension, junction creation
│   ├── intersections.py                # planarize (split at crossings), dangler pruning
│   └── primitives.py                   # Segment, Corner, OrientedRect, polygon utils
├── topology/
│   ├── dcel.py                         # half-edge structure: Node/Edge/Face
│   ├── room_faces.py                   # face extraction, face↔detection assignment, fallbacks
│   ├── attachments.py                  # attach_door(), attach_stairs(), attach_hatch(), clip_acm()
│   └── connectivity.py                 # adjacency graph, connected_to JSON
├── semantics/
│   ├── labeling.py                     # OCR-in-face + GPT merge hook (apply_labels)
│   ├── numbering.py                    # auto-number per floor + duplicate validation
│   ├── floors.py                       # component split, floor_idx, loft detection
│   └── samples.py                      # S00X annotations, material expansion
├── model/
│   └── plan_document.py                # PlanDocument (versioned schema) = current PlanModel + provenance
├── export/
│   ├── base.py                         # Exporter interface: export(doc, path) -> Path
│   ├── json_export.py  dxf_export.py  vsdx_export.py
│   ├── svg_export.py   png_export.py  pdf_export.py
│   └── layout.py                       # multi-floor layout + annotation placement (current acorn_style)
├── quality/
│   ├── validators.py                   # per-check functions → ValidationReport
│   ├── geometry_metrics.py             # Room/Wall IoU, Hausdorff, door alignment (existing benchmark)
│   └── gate.py                         # aggregate score, review routing
├── pipeline.py                         # Pipeline = [Stage, …]; Stage protocol: run(ctx) -> ctx
└── cli.py                              # python -m acorn --image X --out Y --formats json,dxf,vsdx

mlops/          # keep: registry, evaluate, per_class_metrics, deploy (regression + per-class gates)
api/  ui/  active_learning/  tests/     # keep, updated imports
```

**Key interfaces**

```python
class Stage(Protocol):
    name: str
    def run(self, ctx: PipelineContext) -> PipelineContext: ...

@dataclass
class PipelineContext:
    image: np.ndarray
    homography: np.ndarray | None
    detections: Detections | None
    wall_graph: DCEL | None
    doc: PlanDocument | None
    flags: list[ReviewFlag]

class Exporter(Protocol):
    format: str
    def export(self, doc: PlanDocument, out_path: Path) -> Path: ...
```

Every stage is pure (`ctx in → ctx out`), independently unit-testable with synthetic fixtures (no torch/GPU needed — the existing test approach), and configured from `pipeline.yaml`. The pipeline definition itself is data: reorder or disable stages per deployment without code changes.

**Config note:** the class list here is 7 classes (`loft_hatch` added, `wall` singular). `classes.yaml` / `NUM_CLASSES` must match the *trained* model's order exactly — update when the 7-class weights are promoted through the regression + per-class gates (box mAP50 ≥ 0.85 structural; mask mAP50-95 ≥ 0.50 area classes, stairs relaxed).

---

## 5. Reviewer adjustments incorporated (S. Kambala)

- Rooms: mask/contour fallback ranks above box fallback (box-only caused the "big rectangles / wrong layout" regression); primary remains wall-graph faces.
- Walls: confirmed as *topology evidence* (line extraction + snapping), not an object class rendered directly.
- Doors: evaluated by placement on wall lines (door placement rate in S9), not detection score alone.
- Stairs: detection only *locates* the stair area; rendering always uses the Visio/template stair symbol.
- ACM: evaluated by correct-room assignment (S9), not raw mask overlap with the sketch.
- Division of labour confirmed: YOLO detects evidence → geometry engine reconstructs (rooms, shared walls, right angles, closed loops, floor grouping, door-wall connections) → GPT-4o reads text (names, numbers, sample IDs, floors, ambiguous handwriting) → Visio renderer draws (clean walls, template doors/stairs, ACM shading, sample markers, Loft tab only when a loft exists). More YOLO training alone will not fix output quality; reconstruction is the leverage point.

## 6. Migration order (lowest risk first)

1. `geometry/walls/*` (raster→skeleton→vectorize→merge) + unit tests on synthetic wall rasters.
2. `constraints/gaps/intersections` + DCEL + `room_faces` — behind `RECONSTRUCTION_ENGINE=v3`, benchmarked with `quality/geometry_metrics` against the plans-zip ground truth before flipping.
3. Attachments (door/stairs/hatch/acm) — door-on-wall logic already exists (`postprocess.attach_doors_to_walls`); generalize to DCEL edges.
4. Exporters already consume a single document — only the wall representation changes (centerline+thickness instead of room-outline-only).
5. Gate v3 on Room IoU / Wall IoU / door alignment vs. v2 on real sketches; keep v2 as fallback exactly as v2 keeps legacy.
```
