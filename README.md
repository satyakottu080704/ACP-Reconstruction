# ACP-Reconstruction

Hand-drawn asbestos-survey sketch → professional digital floor plan.

Pipeline: **YOLOv11-seg** detects evidence (rooms, walls, doors, floor, stairs,
ACM) → a deterministic **geometry engine** reconstructs clean plans (wall
centerlines, closed room polygons from wall topology, shared walls, snapped
90° angles, gap-free coverage, doors attached to walls) → **GPT-4o** reads
handwriting (room names/numbers, sample IDs, floors) → export to
**JSON / SVG / PNG / PDF / DXF / Visio (.vsdx)** in the Acorn survey style
(ACM red hatching, sample leader arrows, loft hatches, multi-floor layout).

## Repository layout

| Path | Purpose |
|---|---|
| `pipeline.py`, `main.py` | Legacy production pipeline (default, untouched) |
| `reconstruction/` | v2 mask-first geometry engine (walls, topology, coverage, postprocess, quality) |
| `exporters/` | PlanModel → json / png / svg / dxf / pdf / vsdx (Acorn style) |
| `mlops/` | Model registry, evaluation, per-class metric gates, gated deploy |
| `evaluation/` | Geometry benchmark (Room/Wall IoU, Hausdorff, door alignment) |
| `api/`, `ui/` | FastAPI service + Streamlit UI |
| `automation/` | n8n workflow (review-only), Windows render service |
| `tests/` | Unit tests (run without torch/GPU/Visio) |
| `RECONSTRUCTION_PIPELINE_DESIGN.md` | Full pipeline design document |

## Setup

```bash
git clone https://github.com/satyakottu080704/ACP-Reconstruction-.git
cd ACP-Reconstruction-
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

Create your `.env` from the template (never commit it):

```bash
copy .env.example .env         # then edit:
# OPENAI_API_KEY=sk-...
# OPENAI_VISION_MODEL=gpt-4o
```

Place the trained weights at `models/weights/best.pt` (not in git — box mAP50
0.883 / mask mAP50 0.803, trained at imgsz 1280). Keep `MODEL_IMGSZ=1280`:
inference size must equal training size.

## How to run

**1. Full pipeline on a sketch (legacy production path):**

```bash
python main.py --image path/to/sketch.jpg
```

**2. v2 reconstruction engine (feature-flagged; mask-first geometry):**

```bash
set RECONSTRUCTION_ENGINE=v2   # Linux/macOS: export RECONSTRUCTION_ENGINE=v2
python -m reconstruction.cli --image path/to/sketch.jpg --out output/ --formats json,svg,png,pdf,vsdx
```

Or from Python:

```python
from ultralytics import YOLO
from reconstruction import from_yolo_result, apply_labels
from exporters import export_all
import cv2

img = cv2.imread("sketch.jpg")
result = YOLO("models/weights/best.pt")(img, imgsz=1280, conf=0.15, iou=0.5)[0]
plan = from_yolo_result(result, source_image=img)
# merge GPT-4o labels/samples: apply_labels(plan, rooms_info, samples, floor_names)
export_all(plan, "output/", stem="plan", formats=["json", "svg", "png", "vsdx"])
```

**3. API service:**  `uvicorn api.app:app --port 8000`
(endpoints: `/health /predict /reconstruct /export /batch_predict /ocr`)

**4. Streamlit UI:**  `streamlit run ui/streamlit_app.py`

**5. Tests (no GPU/torch/Visio needed):**  `pytest tests/ -q`

**6. Train / evaluate / deploy (GPU host):**

```bash
python train_floorplans.py                       # imgsz 1280, yolo11m-seg
python mlops/deploy.py --weights runs/.../best.pt --data datasets/new_final_yolo/data.yaml --per-class-gate
```

The regression gate only promotes weights that don't regress box/mask mAP;
`--per-class-gate` additionally requires box mAP50 ≥ 0.85 for
room/door/walls/acm and mask mAP50-95 ≥ 0.50 for floor (0.40 stairs).

## Notes

- VSDX rendering with a client template requires Windows + Visio (COM); the
  zip/XML fallback works cross-platform against `utils/visio/template.vsdx`.
- Production publishing stays in review mode: plans upload to SharePoint
  `Manual_Review` until surveyor sign-off.
- Confidential data (sketches, datasets, weights, `.env`) is git-ignored by
  design — code only in this repository.
