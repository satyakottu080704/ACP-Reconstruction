# ============================================
# ACORN ATLAS FLOOR PLAN - CONFIGURATION
# ============================================

import os

# Project root resolves from this file's location, so the project folder can
# be renamed or moved without breaking these paths.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================
# MODEL — YOLOv11m-seg, trained on "New Final YOLO" (7,294 imgs, 6 classes)
# ============================================
# Current production weights: best.pt
# box mAP50 = 0.883, mask mAP50 = 0.803, trained at imgsz 1280.
# A 50-epoch re-run scored 0.855/0.769 (worse) — regression gate blocks it.
# The old ResNet+UNet checkpoint (acorn_atlas_v5_NEW.pth) is kept on disk
# but no longer referenced.
USE_MODEL = os.environ.get("USE_MODEL", "true").strip().lower() in ("true", "1", "yes")
DEBUG_MODE = os.environ.get("DEBUG_MODE", "true").strip().lower() in ("true", "1", "yes")
_DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "weights", "best.pt")
if not os.path.exists(_DEFAULT_MODEL_PATH):
    _DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "training", "Training", "weights", "best.pt")
MODEL_PATH = os.environ.get("ACORN_MODEL_PATH", _DEFAULT_MODEL_PATH)
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_PATH)
MODEL_IMGSZ = int(os.environ.get("MODEL_IMGSZ", "1280"))  # match current training imgsz
MODEL_CONF_THRESHOLD = float(os.environ.get("MODEL_CONF_THRESHOLD", "0.15"))
                                # Tuned 2026-05-13 from 0.25 -> 0.15 to recover
                                # rooms / stairs the model predicts in the
                                # 0.15-0.24 range (missed rooms on prod sketches)
MODEL_IOU_NMS = float(os.environ.get("MODEL_IOU_NMS", "0.5"))
                                # NMS IoU threshold (recommended 0.45-0.55).
                                # Pass as `iou=` to model.predict()/val().
MODEL_ENSEMBLE = os.environ.get("ACORN_MODEL_ENSEMBLE", "false").strip().lower() in ("true", "1", "yes", "on")
EXTRA_MODEL_PATHS = [
    os.path.join(PROJECT_ROOT, p.strip()) if p.strip() and not os.path.isabs(p.strip()) else p.strip()
    for p in os.environ.get("ACORN_EXTRA_MODEL_PATHS", "").replace(";", ",").split(",")
    if p.strip()
]
# ============================================
# FOLDERS
# ============================================
SKETCHES_FOLDER = os.environ.get("SKETCHES_FOLDER") or os.path.join(PROJECT_ROOT, "Input")
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER") or os.path.join(PROJECT_ROOT, "output")

# ============================================
# RENDER PAGE + COORDINATE SPACE (env-overridable)
# ============================================
# A3 landscape in inches; the 0-1000 grid the layout/exporters work in.
PAGE_WIDTH_IN = float(os.environ.get("PAGE_WIDTH_IN", "16.54"))
PAGE_HEIGHT_IN = float(os.environ.get("PAGE_HEIGHT_IN", "11.69"))
COORD_MAX = float(os.environ.get("COORD_MAX", "1000"))

# ============================================
# PIXEL-SCALE FALLBACK (env-overridable)
# ============================================
# Estimated real-world building WIDTH (metres) by room count, used ONLY when the
# surveyor wrote no measurements. Override per deployment without touching code.
EST_WIDTH_M_SMALL = float(os.environ.get("EST_WIDTH_M_SMALL", "10"))    # <= 5 rooms
EST_WIDTH_M_MEDIUM = float(os.environ.get("EST_WIDTH_M_MEDIUM", "15"))  # <= 10 rooms
EST_WIDTH_M_LARGE = float(os.environ.get("EST_WIDTH_M_LARGE", "25"))    # <= 20 rooms
EST_WIDTH_M_XLARGE = float(os.environ.get("EST_WIDTH_M_XLARGE", "40"))  # > 20 rooms

# ============================================
# GPT-4o — Set in .env file:
#   OPENAI_API_KEY=sk-...
#   OPENAI_VISION_MODEL=gpt-4o
# ============================================

# ============================================
# 6 SEGMENTATION CLASSES
# ============================================
NUM_CLASSES = 6
CLASSES = ['acm', 'door', 'floor', 'room', 'stairs', 'walls']


# ============================================
# V2 RECONSTRUCTION ENGINE (feature flag)
# ============================================
# "legacy"  — existing pipeline.py path (default, production-safe)
# "v2"      — new mask-first geometry platform (reconstruction/ package)
#             Only enable after v2 passes the geometry benchmark on real
#             ground truth (evaluation/geometry_benchmark.py gate_check).
RECONSTRUCTION_ENGINE = os.environ.get("RECONSTRUCTION_ENGINE", "legacy").strip().lower()

# Comma-separated list of export formats produced by the v2 engine.
# Supported: json, svg, dxf, png, pdf, vsdx
RECON_EXPORT_FORMATS = os.environ.get("RECON_EXPORT_FORMATS", "json,svg,dxf,png,pdf,vsdx")

# Directory where low-quality predictions are auto-saved for re-annotation
# (active learning queue).  Human annotators review and correct these.
ACTIVE_LEARNING_DIR = os.environ.get(
    "ACTIVE_LEARNING_DIR",
    os.path.join(PROJECT_ROOT, "active_learning", "queue"),
)

# ============================================
# DATASET (used by train_floorplans.py default)
# ============================================
# New Final YOLO: 9,482 images, 6 classes, imgsz 1280
# Roboflow: satyas-workspace-r0oul/modelyolo-by34i v2
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    os.path.join(PROJECT_ROOT, "datasets", "new_final_yolo"),
)


def ensure_folders():
    """Create required output and working directories."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "visio"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "cache"), exist_ok=True)
    os.makedirs(ACTIVE_LEARNING_DIR, exist_ok=True)
    os.makedirs(DATASET_PATH, exist_ok=True)
