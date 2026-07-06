# Acorn Plan Generation ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Consolidated Docs

> All project docs in one place. **`CLAUDE.md` is the source of truth** for architecture/status and is loaded automatically by Claude Code; this file holds the requirements, runbooks, evaluation, and historical notes that used to be scattered across separate `.md` files.

## Contents

0. [Current production truth](#current-production-truth---2026-06-22)
1. [Requirements](#requirements)
2. [Deployment & automation runbook](#deployment-&-automation-runbook)
3. [n8n -> Windows Visio agent runbook](#n8n---windows-visio-agent-runbook)
4. [Evaluation / accuracy measurement](#evaluation--accuracy-measurement)
5. [Renderer convergence (port box-pipeline fixes to native path)](#renderer-convergence-port-box-pipeline-fixes-to-native-path)
6. [Implementation plan (historical)](#implementation-plan-historical)
7. [Test results (historical)](#test-results-historical)
8. [Walkthrough (historical)](#walkthrough-historical)
9. [SAM annotation brief (training)](#sam-annotation-brief-training)


---

# Current production truth - 2026-06-22

This section is authoritative for the live/current plan-generation path. Older merged sections below are retained for history and training context, but anything that conflicts with this section is superseded.

## Current production flow

```text
Outlook Plans inbox
  -> n8n workflow: automation/n8n/acorn_plans_renderer.json
  -> GET Windows VM render_service.py /ready
  -> POST sketch bytes to /render?project=N-xxxxx
  -> pipeline.py:process_sketch
  -> utils/visio/professional_visio.py using Microsoft Visio COM
  -> SharePoint General/AI Automation/AI_COMPLETED
  -> Outlook email marked read and moved to AI_COMPLETED
```

The active production renderer is the Windows VM HTTP render service (`automation/render_service.py`) on port 8765. It wraps the box pipeline and Visio COM renderer, warms Visio through `/ready`, serializes renders with a lock, returns `X-Review-Required` and `X-Quality-Flags`, and deletes temp files after returning the VSDX.

## Current model and AI settings

- AI provider: OpenAI only for production review runs.
- AI model: `OPENAI_VISION_MODEL=gpt-4o` with `OPENAI_LABEL_ATTEMPTS=1`.
- YOLO model path: `ACORN_MODEL_PATH`, default `training/Training/weights/best.pt` from `config.py`.
- YOLO classes: `acm`, `door`, `floor`, `room`, `stairs`, `walls`.
- `gpt-4o-mini`, Gemini, Groq, Ollama, and `PLAN_LAYOUT_PROVIDERS` are comparison/fallback-era concepts, not the current production setting.

## Superseded paths

- `automation/container/process_plan.py` + `generate_plan.py` + the `plangenration` Linux-native VSDX path are superseded for current production. Keep as fallback/historical only.
- `automation/windows_visio_agent.py` polling SharePoint `Pending_Draw` is superseded by the HTTP render service design. Keep as historical/fallback only.
- Any doc section saying Gemini is primary, `gpt-4o-mini` is production, `models/best_room.pt` is the current model, or n8n currently draws inside `plangenration` is stale.


---

# Requirements

*(merged from `REQUIREMENTS.md`)*

# Acorn Atlas ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Project Requirements

## 1. System Overview
The Acorn Atlas Floor Plan Pipeline is an advanced, automated document generation and computer vision system designed to convert hand-drawn asbestos-survey sketches photographed by field surveyors into professional, survey-standard Microsoft Visio (`.vsdx`) floor plans. The system automates the processing of complex sketched plansÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Âcorrecting uneven camera lighting, cropping out survey forms, deskewing rotations, and suppressing grid lines. It handles multi-floor sketches (exporting each floor to a separate page in a single document), identifies asbestos-containing material (ACM) rooms (filling them with a muted red), labels inaccessible rooms (filling them with blue and adding a "NO ACCESS" prefix), and preserves red-pen surveyor sample annotations (as dots and arrows pointing into target rooms).

### Entry Points
- **main.py**: The single Command Line Interface (CLI) entry point (replaced `run_detector.py` + `test_local.py`, both deleted 2026-06-17). It supports single-image processing, resumable batch processing, renderer selection (`--renderer com|aspose`), model-only diagnostic overlays, and cache-clearing, and prints a per-run geometry-source breakdown (YOLO vs GPT).
- **pipeline.py**: The core orchestration pipeline where the preprocessing, vision calls, YOLO detection, geometric merging/deduplication, and Visio exports are executed.

### Current Mode
- **Box pipeline + Windows render service**: production review runs use `pipeline.py:process_sketch` through `automation/render_service.py`, then render with Microsoft Visio COM via `utils/visio/professional_visio.py`.
- **Hybrid YOLO + OpenAI vision**: YOLO supplies room/stair/door/floor geometry candidates; OpenAI `gpt-4o` supplies handwriting labels, room numbers, floor names, ACM/no-access hints, and sample references.
- **Review-first output**: n8n uploads generated VSDX files to SharePoint `General/AI Automation/AI_COMPLETED`. Failed runs are not marked complete; successful runs mark the email read and move it to Outlook `AI_COMPLETED`.
- **Superseded provider fallback**: Gemini/Groq/Ollama provider fallback is not part of the current production setting.

---

## 2. Functional Requirements

FR-001 | Title | Description | Status (Done/In Progress/To Do)
:--- | :--- | :--- | :---
**FR-001** | Image Preprocessing | Automatically rotates landscape templates photographed in portrait orientation using edge density ratios and green corner pixels (logo detection). Normalizes uneven lighting via CLAHE on the luminance channel in LAB space. Automatically crops the surveyor form panel (searching for the sharpest column/row density drops). Trims the bottom document footer and left margin remnants. Deskews slight rotational offsets using Hough line angle medians. Suppresses graph paper grid lines using vertical and horizontal morphological open operators while protecting thick pen strokes (walls/text) via a tighter darkness threshold. | **Done**
**FR-002** | Room Detection (Two-Pass AI) | Queries the vision API in two distinct passes: Pass 1 (labels-only) extracts room names, circled numbers, ACM hatching, stairs, and no-access indicators. Pass 2 (full-layout) estimates relative coordinates (`x_pct`, `y_pct`, `w_pct`, `h_pct`). Runs an automated completeness recovery pass if the reported room count is greater than the listed room count, and splits the image into 4 overlapping quadrants (magnified crops) to double the resolution and read tiny room names (cupboards, WCs, stores) that the full-page pass misses. | **Done**
**FR-003** | ACM Detection | Detects asbestos-containing materials (ACM) via black or red diagonal hatching lines drawn in the room interior. Colors these rooms in the final Visio document with a professional muted red fill (`RGB 220, 50, 50`) and tracks their ACM state. | **Done**
**FR-004** | No-Access Detection | Identifies rooms crossed out with a large X or containing text like "No Access", "Locked", "N/A", or "Inaccessible". Fills these rooms in Visio with a blue background (`RGB 50, 100, 200`) and prepends a bold `NO ACCESS` prefix to their display labels. | **Done**
**FR-005** | Multi-Floor Handling | Supports sketches containing multiple floors (e.g. Ground Floor, First Floor, Loft) side-by-side or stacked on a single sheet. Assigns correct floor indices (0=Ground, 1=First, 2=Loft/Second) based on room names or panel floor indicators, and routes rooms to separate, correctly ordered pages in the single `.vsdx` file. | **Done**
**FR-006** | Sample Reference Detection | Preserves red-pen surveyor sample annotations (e.g., "S01 FT", "S02 Mastic", "Ref S004", "+") as custom sample entities, determines their target room number and floor index, and renders them outside the rooms with red arrows pointing inside. | **Done**
**FR-007** | Visio Output Generation | Generates professional floor plans with thin black inner walls (~1pt), bold outer building walls (~2.5pt), door gaps, quarter-circle door swing arcs, room labels, and measured/estimated dimensions. Provides three export fallbacks: (1) Professional Visio COM (A3 Landscape layout), (2) Simple Visio COM, and (3) Native XML-based platform-independent zip exporter (generates valid `.vsdx` files on macOS/Linux/Windows without Microsoft Visio installed). | **Done**
**FR-008** | Batch Processing | Processes entire folders of sketches in a single command, tracks execution timings, prints folder-wide quality metric summaries, and supports a crash-safe `--resume` flag to skip already-processed images. | **Done**
**FR-009** | Caching | Caches OpenAI/GPT vision JSON payloads based on a 16-character SHA-256 hash of the preprocessed sketch bytes to prevent redundant API calls and save costs. Automatically invalidates caches containing fewer than two rooms. | **Done**
**FR-010** | YOLO Model Integration & Merge | Runs the configured YOLO model to detect room/stair/door/floor geometry. Merges model geometry with OpenAI `gpt-4o` labels using spatial proximity and Hungarian-greedy matching. Gracefully falls back to pure AI layout if YOLO finds too little usable geometry, and re-adds AI-only rooms when needed. | **Done**

---

## 3. Non-Functional Requirements

NFR-001 | Title | Description | Current Value
:--- | :--- | :--- | :---
**NFR-001** | Processing Time | Average end-to-end execution time per sketch, including import, preprocess, AI calls, snapping, merging, and Visio export. | **~44 seconds** (pure GPT-4o) or **~45 seconds** (with YOLO, running on CPU).
**NFR-002** | API Cost | Token cost incurred from OpenAI vision under high-detail input. | Current production review setting uses OpenAI `gpt-4o`; cost is accepted because accuracy is the priority.
**NFR-003** | Startup Time | Import and first-run startup latency for the python pipeline. | **~0.25 seconds** (achieved by stripping out local PyTorch imports from the main script initialization and lazy-loading YOLO weights).
**NFR-004** | Batch Resume Capability | Safe recovery and resume functionality when executing folder-wide batches. | **Supported** via the `--resume` flag in `main.py` (reads `batch_results.json`).
**NFR-005** | API Key Security | Management of secrets and API keys. | Managed locally via ignored `.env` files and n8n credentials. Do not commit OpenAI, render-service, SharePoint, or Outlook secrets.

---

## 4. YOLO Model Requirements

### Current Model Specs
- **Model file**: configured by `ACORN_MODEL_PATH`; default is `training/Training/weights/best.pt` from `config.py`.
- **Classes**: `['acm', 'door', 'floor', 'room', 'stairs', 'walls']`.
- **Image size**: `MODEL_IMGSZ=1280` by default.
- **Confidence threshold**: `MODEL_CONF_THRESHOLD=0.15` by default.
- **Optional ensemble**: `ACORN_MODEL_ENSEMBLE=true` and `ACORN_EXTRA_MODEL_PATHS` can add extra models for local testing, but the default production setting is one configured model.

### Historical Model Notes
Older documentation mentions `models/best_room.pt`, a 9-class dataset, and YOLOv11s-seg metrics. Treat those as historical training notes unless `config.py` is changed and the eval gate proves an improvement.

### Performance Targets
- **Room Mask mAP50**: $\geq 0.70$ on the current dataset (achieved **0.7356**).
- **Expanded Target**: $\geq 0.80$ room mask mAP50 once the +600 image batch is merged.

### How to Retrain
- **Notebook**: `training/Train_YOLO_v2.ipynb` (run on Kaggle/Colab with CUDA GPU).
- **Script**: `python train_floorplans.py` inside the project folder.

---

## 5. Known Issues

### Issue 1: GPT-4o Vision Non-Determinism
- **Description**: The same sketch may return slightly different room names, counts, or bounding boxes across different runs due to LLM variance.
- **Affected File/Function**: `pipeline.py` / `get_room_labels_gpt4o`
- **Suggested Fix**: Leverage the file-based caching mechanism (`_get_cached` based on a SHA-256 hash of the preprocessed image). For production-grade consistency, incorporate a surveyor-review step before finalizing the `.vsdx` file.

### Issue 2: Duplicate Room Number/Label Snapping Suffixes
- **Description**: Post-processing adds suffixes like ` 2` or ` (003)` when duplicate room labels are found on the same floor (e.g. `Bathroom` + `Bathroom (003)`). While functionally accurate, it is visually disjointed.
- **Affected File/Function**: `pipeline.py` / `merge_results` (POST-PROCESS 3)
- **Suggested Fix**: We recently improved this by formatting duplicates cleanly as `f"{base} ({clean_num})"` using the surveyor-designated circled number. A full fix requires checking adjacent wall structures to determine if a room is a cupboard extension or a distinct room (e.g. "Kitchen Cupboard").

### Issue 3: Inaccurate Room Number OCR on Dense Hand-Drawn Sketches
- **Description**: circulates numbers (e.g. `001`, `002`) are sometimes misread or skipped on dense, heavily annotated sketches.
- **Affected File/Function**: `pipeline.py` / `_call_gemini` or `_call_gpt4o`
- **Suggested Fix**: Crop circles out based on YOLO `text`/`room` boundaries and pass high-contrast cropped squares to a localized OCR model (e.g. PaddleOCR or EasyOCR) instead of relying solely on LLM global vision.

---

## 6. How to Run

### Single Sketch Conversion
```bash
python main.py --image "path/to/sketch.jpg" --output "path/to/out.vsdx"
```

### Batch Processing (Resumable)
```bash
python main.py --batch "path/to/folder" --resume
```

### Clear Cache
```bash
python main.py --clear-cache
```

### Model-Only Diagnostic (YOLO Bounding Box Overlay)
```bash
python main.py --image "path/to/sketch.jpg" --model-only
```

### How to Retrain YOLO Model
1. Set up a CUDA GPU environment.
2. Run `python train_floorplans.py` or run all cells in the Jupyter notebook `training/Train_YOLO_v2.ipynb`.

### How to Add New Roboflow Exports
1. Download a fresh COCO-segmentation zip export from Roboflow and drop it into `Input/`.
2. Run the remapping script to clean and standardize classes:
   ```bash
   python training/clean_roboflow_export.py
   ```
3. Use the merge script to append it to the current dataset:
   ```bash
   python training/merge_datasets.py
   ```

---

## 7. File Structure

Single entry point: `main.py` (box pipeline). Key directories:
- `automation/container/` - superseded Linux-native generator/fallback (process_plan.py, generate_plan.py)
- `automation/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Windows Visio agent + n8n workflows/runbook
- `utils/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â preprocessing, room detection, Visio renderers
- `plans/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â renderers + evaluation predictors
- `evaluation/` + `ground_truth/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â accuracy scoring vs manual plans
- `training/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â YOLO dataset + training

See `CLAUDE.md` for the authoritative architecture. Run `git ls-files` for the
live file list ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â this doc no longer pins a snapshot (the old one went stale).

---

## 8. Dependencies

### Python pip Packages (requirements.txt)
- **Core ML**:
  - `torch>=2.0.0`
  - `torchvision>=0.15.0`
  - `ultralytics>=8.0.0`
- **Image Processing**:
  - `numpy>=1.20.0`
  - `scipy>=1.10.0`
  - `opencv-python>=4.5.0`
  - `Pillow>=10.0.0`
- **Visio Export**:
  - `pywin32>=306` (Windows COM Automation)
- **HTTP / Environment**:
  - `httpx>=0.24.0`
  - `requests>=2.31.0`
  - `python-dotenv>=1.0.0`
  - `python-dateutil>=2.8.0`
- **Data & Configuration**:
  - `pandas>=2.0.0`
  - `openpyxl>=3.1.0`
  - `watchdog>=3.0.0`
- **Plan Output Alternatives**:
  - `reportlab>=4.0.0`
  - `svgwrite>=1.4.0`
- **AI Clients**:
  - `openai>=1.0.0`
  - `groq>=0.9.0`
  - `anthropic>=0.18.0`
- **Testing**:
  - `pytest>=7.4.0`
  - `typing-extensions>=4.8.0`
- **Local/free OCR options**:
  - `pytesseract>=0.3.10` (Python wrapper; also requires the Tesseract OCR Windows app)
  - `rapidocr-onnxruntime>=1.3.0` (local ONNX OCR)
  - `paddleocr>=2.6.0` (heavier local OCR)
  - `easyocr>=1.6.0` (fallback local OCR)
- **Training Utilities**:
  - `tqdm>=4.60.0`
  - `matplotlib>=3.4.0`

### Local/free extraction dependency notes

The local/no-AI extraction commands require OpenCV. If the venv raises:

```text
ModuleNotFoundError: No module named 'cv2'
```

install the main runtime requirements:

```powershell
cd C:\Projects\AcornPlanGeneration
.\.venv\Scripts\pip install -r requirements.txt
```

or install OpenCV only:

```powershell
.\.venv\Scripts\pip install opencv-python
```

For free OCR, install at least one local OCR engine. Tesseract is the lightest
starting point:

```powershell
.\.venv\Scripts\pip install pytesseract
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

The Tesseract Windows application must also be installed separately. The common
Windows build is the UB Mannheim installer:

```text
https://github.com/UB-Mannheim/tesseract/wiki
```

Alternative local OCR engines:

```powershell
.\.venv\Scripts\pip install rapidocr-onnxruntime
.\.venv\Scripts\pip install paddleocr
.\.venv\Scripts\pip install easyocr
```

Free/local extraction test command:

```powershell
$env:ACORN_LOCAL_OCR_ONLY="true"
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"

.\.venv\Scripts\python plans\free_local_extract.py `
  --image "C:\Users\SuryaKambala\OneDrive - Acorn Analytical Services\Desktop\Documents\IMAGES\1.jpeg" `
  --output-dir "evaluation\predictions"
```

### Node/npm Packages
- **None**. The project is a pure Python application.

### System Requirements
- **Operating System**: **Windows** is required for standard Visio COM export (which utilizes pywin32 COM automation). However, the platform-independent **XML VSDX fallback** is fully supported, allowing valid `.vsdx` files to be created natively on **macOS/Linux** without any Microsoft Visio installation.
- **Software (Optional)**: **Microsoft Visio** installed on Windows for the primary pywin32 COM automation.
- **Python**: **Python 3.8+** (tested and fully validated on Python **3.13.9**).
- **Hardware (Optional)**: A CUDA-enabled NVIDIA GPU is recommended only if retraining the YOLO model locally. Standard inference runs quickly on standard CPU.


---

# Deployment & automation runbook

*(merged from `automation/README.md`)*

# Acorn Plan Generation - Operations & Deploy

The current production plan generator is the Windows render service called by n8n. Older Linux-container deployment notes are retained below as fallback/historical only.

**Current active service:** Windows VM running `automation/render_service.py` on port `8765`.
**Current n8n workflow:** `automation/n8n/acorn_plans_renderer.json`.
**Current SharePoint target:** `General/AI Automation/AI_COMPLETED`.

## Runtime flow

```text
n8n
  -> polls Outlook Plans inbox
  -> downloads first image attachment
  -> GET $RENDER_SERVICE_URL/ready
  -> POST $RENDER_SERVICE_URL/render?project=<N> with sketch bytes
  -> receives VSDX from Windows VM / Visio COM
  -> uploads VSDX to SharePoint AI_COMPLETED
  -> marks source email read and moves it to Outlook AI_COMPLETED
```

n8n does **not** pull code from GitHub at runtime. The live behavior is whatever is deployed on the Windows VM plus the imported n8n workflow JSON.

## Source of truth

- **Canonical repo:** GitHub `SuryaKambalaAcorn/AcornPlanGenration`, branch `main`.
- **Current production code path:** `automation/render_service.py`, `pipeline.py`, `utils/visio/professional_visio.py`, `config.py`, and `automation/n8n/acorn_plans_renderer.json`.
- **Do not edit generated/server copies by hand.** Push code, deploy the VM/service, then import/update the n8n workflow if it changed.

## Superseded fallback: Linux native container

The previous `plangenration` container flow (`docker exec plangenration python src/process_plan.py ...`) is no longer the active production path. Keep it only as a fallback or historical reference until explicitly revived.

## Publishing modes (`PLAN_PUBLISH_MODE`)

- **`review`** (default, safe): every plan ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ SharePoint `AI_COMPLETED`; n8n marks
  the email read so it isn't reprocessed; nothing is presented as an approved plan.
- **`auto`** (opt-in): a per-plan **quality gate** routes each plan ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â a structurally
  clean plan is still uploaded to `AI_COMPLETED` but may be flagged as structurally clean; a broken one
  (placeholder, unlabeled rooms, <4 walls, out-of-bounds coords, missing summary)
  goes to `AI_COMPLETED`. The gate is **STRUCTURAL only** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â it does *not* verify the
  plan matches the sketch. Tunable: `PLAN_MIN_ROOMS`, `PLAN_MIN_WALLS`.
- **`production`**: requires a ground-truth acceptance report (`PLAN_ACCEPTANCE_REPORT`,
  produced by `plans/ground_truth_eval.py`); still uploads to `AI_COMPLETED`, and approval depends on
  the project scored as accepted, else routes to review.

Folders: `SHAREPOINT_REVIEW_FOLDER` (default `General/AI Automation/AI_COMPLETED`),
`SHAREPOINT_OUTPUT_FOLDER` (default `General/AI Automation/AI_COMPLETED`).
Trackerfiler send is ignored in review mode.

## Current render-service / pipeline env defaults

```text
RENDER_SERVICE_URL=http://<windows-vm>:8765
RENDER_SERVICE_TOKEN=<set in n8n and VM env>
OPENAI_VISION_MODEL=gpt-4o
OPENAI_LABEL_ATTEMPTS=1
ACORN_MODEL_PATH=training/Training/weights/best.pt
MODEL_IMGSZ=1280
MODEL_CONF_THRESHOLD=0.15
PLAN_PUBLISH_MODE=review
PLAN_VECTOR_REJECT_MODE=overlay
SHAREPOINT_REVIEW_FOLDER=General/AI Automation/AI_COMPLETED
```

Current production review runs use OpenAI `gpt-4o` with `OPENAI_LABEL_ATTEMPTS=1`. The model path is dynamic through `ACORN_MODEL_PATH`; verify any new weights with the eval gate before deploying.

## Deploy - superseded Linux-container fallback

The commands in this section apply only if the old `plangenration` Linux-native path is revived. They are not the active Windows render-service production deployment.

**Preferred ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the script** (from Windows, after tests pass + changes pushed to `main`):
```powershell
.\automation\deploy_plangenration.ps1
```
It (1) refuses to deploy unless local `main` matches `origin/main`, (2) backs up
the live files under `/opt/Plangenration/backups/<timestamp>`, (3) copies the
production files + Dockerfile + corporate `template.vsdx`, then builds and labels
the image with the Git commit SHA.

**Manual fallback** (ship the 6 build-context files, then build + recreate):
```bash
scp automation/container/generate_plan.py automation/container/process_plan.py \
    automation/container/layout_extractor.py automation/container/vision_client.py \
    automation/container/template.vsdx automation/container/Dockerfile \
    root@46.62.131.88:/opt/Plangenration/

ssh root@46.62.131.88 "cd /opt/Plangenration && \
  docker build -t plangenration:latest . && \
  docker rm -f plangenration && \
  docker run -d --name plangenration --restart unless-stopped \
    --env-file /opt/acorn/.env --network acorn_network \
    -v /opt/Plangenration/reports:/app/src/output/reports \
    -v /opt/acorn/config:/app/config:ro \
    plangenration:latest"
```

## Verify old container fallback

```bash
# Full functional check ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the EXACT command n8n runs, no SharePoint writes:
ssh root@46.62.131.88 "docker exec plangenration python src/process_plan.py 'N-108188' --image /tmp/N-108188_sketch.png --dry-run"
# expect one JSON line: {"ok": true, "dryRun": true, "qualityGatePassed": ..., ...}

# Running image revision must match GitHub main:
ssh root@46.62.131.88 "docker inspect plangenration --format '{{ index .Config.Labels \"org.opencontainers.image.revision\" }}'"
```

## `plangenration` vs `acorn_reporting`

| | Reporting (`acorn_reporting`) | Plan-gen (`plangenration`) |
|---|---|---|
| Repo | `AcornReportingAutomation` | `AcornPlanGeneration` |
| Ships | tar of `src/ config/ data/ ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦` | just the 6 build-context files |
| Build context | `/opt/acorn` (`docker build .`) | `/opt/Plangenration` |
| Start | `docker compose up -d` | `docker run ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦` (no compose) |

`plangenration` is `FROM acorn_reporting:latest`. If you change the *reporting*
base image (shared deps, the base `gemini_text.py`), deploy that via the
`AcornReportingAutomation` runbook first, **then** rebuild `plangenration` on top.

## n8n

Import/update `automation/n8n/acorn_plans_renderer.json` in the n8n UI. The renderer workflow waits for the Windows render service `/ready`, sends one sketch to `/render`, uploads every returned VSDX to SharePoint `AI_COMPLETED`, then marks the source email read and moves it to Outlook `AI_COMPLETED`.

Failed render/upload runs stay unread for retry. The JSON in Git is a reviewable backup; editing it does not update the live workflow.



---

# n8n -> Windows Visio agent runbook

*(merged from `automation/n8n/WINDOWS_AGENT_RUNBOOK.md`)*

# Running the box pipeline from n8n via the Windows Visio agent

**Status: SUPERSEDED.** This SharePoint `Pending_Draw` polling-agent design was replaced by the Windows HTTP render service (`automation/render_service.py`). Keep this section only as historical/fallback reference.

## Why this exists

All the accuracy/style work (YOLO geometry, External labelling, multi-floor
quality gate, one-page sections, pink ACM, loft tab) lives in the **box
pipeline** (`pipeline.py`, rendered by `utils/visio/professional_visio.py` via
Visio COM). The COM renderer needs **Windows + Microsoft Visio**.

The current live workflow now calls the Windows HTTP render service directly. The older Linux-container and SharePoint polling-agent designs below are superseded and kept only as fallback/historical reference.


The old polling-agent delegation design was:

```
Plans email ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬â€œÃ‚Âº n8n (Linux)
                 ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡  extract N-number + first image attachment
                 ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¼
        SharePoint "General/AI Automation/Pending_Draw"   ÃƒÂ¢Ã¢â‚¬â€Ã¢â‚¬Å¾ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ n8n uploads here
                 ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡
                 ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¼
   Windows Visio agent (Windows + Visio, polling every 6s)
        windows_visio_agent.py ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ pipeline.process_sketch ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ COM render
                 ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡
                 ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¼
        SharePoint "General/AI Automation/AI_COMPLETED"  (review folder)
```

**Proven end-to-end 2026-06-17:** uploaded `N-104621_sketch.jpg` to
Pending_Draw ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ ran `windows_visio_agent.py --project N-104621 --once` ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ it drew
the plan via the box pipeline (labelled "001 External", split_loft layout, COM
render) and uploaded `N-104621 AI Draft.vsdx` to the historical output folder. Current production review uploads to `AI_COMPLETED`.

---

## Part 1 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Provision the Windows + Visio worker (do first)

1. **VM:** Windows Server 2022 (or Windows 11) with **Microsoft Visio**
   installed and licensed. Min ~4 vCPU / 8 GB RAM (YOLO + Visio).
2. **Code + deps:**
   - `git clone` this repo; `git lfs install && git lfs pull` (YOLO weights).
   - `pip install -r requirements.txt` (includes `pywin32`, `ultralytics`,
     `opencv-python`).
   - Open Visio once and dismiss first-run dialogs (COM automation needs the
     app fully initialised).
3. **Env (`.env` in repo root):** `OPENAI_API_KEY`, `OPENAI_VISION_MODEL=gpt-4o`,
   and the SharePoint app creds the agent uses:
   `SP_DRIVE_ID`, `SP_CLIENT_ID`, `SP_CLIENT_SECRET`, `SP_TENANT_ID`
   (the Azure app needs Graph **Files.ReadWrite.All / Sites.ReadWrite.All**).
   Optional: `SHAREPOINT_PENDING_FOLDER`, `SHAREPOINT_OUTPUT_FOLDER` (defaults
   match the diagram above).
4. **Run as an always-on service** (so it survives reboots):
   - Easiest: **NSSM** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â `nssm install AcornVisioAgent "C:\path\.venv\Scripts\python.exe" "C:\path\automation\windows_visio_agent.py"`, set the working dir to the repo root, start automatically. COM needs an interactive desktop session ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â run the service under a logged-in user account (or use Task Scheduler "Run only when user is logged on" with auto-logon), NOT Session 0 isolation.
   - Verify: drop a sketch in Pending_Draw and watch it appear in
     the configured review folder within ~10s.
5. **Smoke test:** `python automation/windows_visio_agent.py --project N-XXXXX --once`
   (filters to one project, processes the current queue, exits).

---

## Part 2 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â The n8n workflow change (prepared; flip in Part 4)

Edit `Acorn Plans ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Inbox ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Automated Drawing` (`u63BSxUIVvTTub0z`) in the n8n
**UI** (safer than CLI import). Keep everything up to and including
**`Name sketch`**. Then:

**Remove** the two SSH nodes: `Upload sketch (SSH)` and
`Draw plan in container (SSH)`.

**Add** in their place (the agent does the drawing, so n8n only hands off):

1. **HTTP Request ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â "Get Graph token"**
   - POST `https://login.microsoftonline.com/{{SP_TENANT_ID}}/oauth2/v2.0/token`
   - Body (form-urlencoded): `client_id`, `client_secret`, `grant_type=client_credentials`,
     `scope=https://graph.microsoft.com/.default`
   - (Use the SharePoint app creds ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â same ones the agent uses. Store as n8n
     credentials, don't inline secrets.)

2. **HTTP Request ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â "Upload sketch to Pending_Draw"**
   - `PUT https://graph.microsoft.com/v1.0/drives/{{SP_DRIVE_ID}}/root:/General/AI Automation/Pending_Draw/{{ $('Name sketch').item.json.sketchFile }}:/content`
   - Header `Authorization: Bearer {{ $json.access_token }}`
   - Body: **Binary** = the `data` field from `Download sketch` (Content-Type
     `application/octet-stream`).
   - Success = HTTP 200/201.

3. Wire **`Handled (mark read)?`** to check the upload status (200/201) instead
   of the old container stdout. **Mark-read semantics change:** the email is
   marked read once the sketch is successfully handed to Pending_Draw (drawing
   is now asynchronous on the agent), not after the plan is drawn.

> The filename `sketchFile` is already `N-<num>_sketch.<ext>`, which the agent
> parses for the project number ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â no change needed there.

The old Linux-container workflow has been removed from the repo. It is no longer
maintained; recover it from Git history only if a deliberate fallback is needed.

---

## Part 3 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Keep, or retire, the container path

Leave the Linux-container generator in place but **inactive** as a fallback.
Do not run both paths for the same email (you'd draw twice).

---

## Part 4 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Cutover (only when ready)

1. Confirm the agent is **running 24/7** on the VM and the smoke test passes.
2. In n8n, save the edited workflow and **activate** it (it replaces the
   container draw step).
3. Send one real Plans email; confirm the VSDX lands in `AI_COMPLETED` within
   ~1 poll cycle and the email is marked read/moved to Outlook `AI_COMPLETED`.
4. Monitor the first day.

## Rollback

Recover the old container workflow from Git history and re-import it:
```
git log --oneline -- automation/n8n/acorn_plans_automated_drawing.json   # find the last commit
git show <commit>:automation/n8n/acorn_plans_automated_drawing.json > /tmp/wf.json
docker cp /tmp/wf.json n8n:/tmp/wf.json
docker exec n8n n8n import:workflow --input=/tmp/wf.json
```
(or paste the recovered JSON into the n8n UI), then deactivate the new version.
This restores the container path exactly as it was.


---

# Evaluation / accuracy measurement

*(merged from `evaluation/README.md`)*


## Manual-plan regression gate

Use this before shipping geometry or extraction changes. It compares generated structure against `ground_truth/manual_plans_truth.json` and the matching local sketches in `output/_at_sketches/`.

```powershell
cd C:\Projects\AcornPlanGeneration
.\.venv\Scripts\Activate.ps1
python evaluation\score_against_manual.py --gate --json --max-projects 10
```

For pytest/CI-style execution, enable the expensive gate explicitly so normal unit tests do not call GPT/Visio:

```powershell
$env:ACORN_RUN_EVAL_GATE='1'
$env:ACORN_EVAL_MAX_PROJECTS='10'
python -m pytest tests\test_manual_eval_gate.py -q
```

Default thresholds are intentionally modest until the signed-off baseline is agreed: label >= 0.70, room number >= 0.70, sample >= 0.50 where samples exist, and room-count delta <= 2. Override with `ACORN_EVAL_MIN_LABEL_RATE`, `ACORN_EVAL_MIN_NUMBER_RATE`, `ACORN_EVAL_MIN_SAMPLE_RATE`, and `ACORN_EVAL_MAX_ROOM_DELTA`.

# Evaluation Harness ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â measure extraction accuracy

Turns "is it good?" into a **number**, by scoring an extractor's output against
known-correct answers on real sketches. This is the milestone `PRODUCTIONIZE.md`
calls the real blocker: no prompt/model/OCR change should ship unless it *raises*
these numbers on the same truth set.

## Pieces

| Script | Role |
|---|---|
| `plans/llm_extract.py` | AI predictor for comparison/evaluation; current production comparisons should use OpenAI `gpt-4o` |
| `plans/free_local_extract.py` | **Free/offline** predictor (OpenCV + local OCR) ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ prediction JSON |
| `plans/ground_truth_eval.py` | **Scorer** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â prediction JSON vs ground truth ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ metrics + accept/reject |
| `evaluation/draft_truth/` | editable AI/local drafts before human correction |
| `evaluation/truth/` | human-corrected approved answers only (the ground truth) |
| `evaluation/predictions_llm/`, `evaluation/predictions/` | predictor outputs |
| `evaluation/reports*/` | metric reports (`*.json` + `*.csv`) |

Run everything with the repo venv: `.venv/Scripts/python.exe`.

## Step 1 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â build ground truth (one-time per sketch; the only manual part)

Seed a **high-quality draft** from the AI extractor (far faster to *correct* than
to write from scratch), then fix it against the actual sketch:

```bash
.venv/Scripts/python.exe plans/llm_extract.py --image "<sketch>" --output-dir evaluation/draft_truth
```
Open `evaluation/draft_truth/<key>.json` and correct it against the image:
- fix each room `label` and `floor`; add rooms the AI missed; delete wrong ones
- replace every nominal/zero-size `bbox` with the real room box from the sketch
- fix `samples` (the `S0xx` / number IDs)
- remove local-only fields such as `source_image`, or change them to portable names
- remove the draft `status` field or set it to `approved`

Only after correction, move the JSON into `evaluation/truth/`.

ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â **You must actually correct it.** If you score against an un-corrected draft you
are comparing the AI to itself (always ~100%) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â meaningless. The scorer now rejects
approved-truth files that still contain draft status, local absolute image paths, or
zero-size room boxes. Aim for **5ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“10 sketches** you can hand-verify to start.

## Step 2 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â predict (every run / every change)

```bash
.venv/Scripts/python.exe plans/llm_extract.py        --image "<sketch>" --output-dir evaluation/predictions_llm
.venv/Scripts/python.exe plans/free_local_extract.py --image "<sketch>" --output-dir evaluation/predictions
```

## Step 3 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â score

```bash
.venv/Scripts/python.exe plans/ground_truth_eval.py --pred-dir evaluation/predictions_llm --truth-dir evaluation/truth --output-dir evaluation/reports_llm
.venv/Scripts/python.exe plans/ground_truth_eval.py --pred-dir evaluation/predictions     --truth-dir evaluation/truth --output-dir evaluation/reports_free
```

## Step 4 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â read the result

`evaluation/reports_*/ground_truth_eval.{json,csv}`:
- **per sketch:** `room_f1`, `label_match_rate`, `room_number_match_rate`,
  `floor_match_rate`, `sample_match_rate`, `overall_score`, `accepted` (0/1)
- **summary:** `avg_overall_score`, number `accepted` vs the acceptance thresholds
  (default `overall_score >= 0.85`).

## The rule

> No prompt, model, OCR, or provider change ships unless it **raises**
> `avg_overall_score` (and the per-metric rates) on the **same** truth set.

This is how you stop guessing: the LLM vs free comparison, a new prompt, a Gemini
tweak ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â each becomes a measured delta, not an opinion.


---

# Renderer convergence (port box-pipeline fixes to native path)

*(merged from `docs/renderer_convergence.md`)*

# Renderer convergence ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â port the box-pipeline fixes into the native path

**Why:** there are two renderers that share the same extractor:
- **Box pipeline** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â `pipeline.py` ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ COM (`utils/visio/professional_visio.py`). Windows only. Has all the geometry/label/safety fixes (branch `fix/multifloor-geometry-and-dynamic-config`).
- **Native container** - `automation/container/generate_plan.py` (NativeVsdxExporter). Linux, no Visio. Superseded for current production, but still a possible future all-Linux target if fidelity improves.

The native path is being hardened separately (multi-page, A3, title blocks, branch `claude/confident-ramanujan-teeo1p`). To get **best output on the all-Linux path**, it must inherit the proven logic below ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â ideally in the **shared extraction/merge layer** so both renderers benefit and don't diverge. Reference implementations are in `pipeline.py` (line numbers approximate; find by function name).

## Port these (each is already working + measured in the box pipeline)

| # | Fix | Source in `pipeline.py` | What the native path should do |
|---|---|---|---|
| 1 | **Floor field (the #1 multi-floor fix)** | floor handling in `merge_results` (`:2217`) + loft-separation block (`~:3050`) | Add `floor` to the extractor schema (`layout_extractor.py` **both copies** + `gemini_text.py`), tag every element, group pages by it (Stage C of the existing brief). Reuse the box pipeline's `floor_idx` inheritance. |
| 2 | **External-survey collapse** | `merge_results` post-step, `"External survey: collapsed"` (`:3106`) | Exact `"External"` label + ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¤3 rooms ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ collapse to one area (kills duplicate/`Room N` over-detections). |
| 3 | **Label vocab ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ team words** | `_ROOM_NAME_MAP` (`:720`) + `_normalize_room_name` (`:757`) | Sync the container's room-name map: `Lounge` (not Living Room), `Hall` (not Hallway), `Cupboard` (not CPD), `Landing` (LAND), orientation `Rear/Front/Side`ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢`External`. Leave bare `Bed`. |
| 4 | **ACM safety ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â never auto-downgrade** | `_correct_samples_by_color` (`:3577`) + propagation `"kept ACM (model-flagged)"` (`:2769`) | A green-pixel reading only confirms a negative; never flip a positive. Don't clear a model-flagged ACM room on negative samples. (Asbestos doc ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â false negatives are unsafe.) |
| 5 | **Fragmentation fallback** | `_rooms_are_fragmented` (`:1898`) | Dead-space gap >30% of a floor span = scattered. Native path has no overlay ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ route to **manual review** instead of rendering a scattered plan. |
| 6 | **Spurious-sliver / placeholder drop** | `_drop_spurious_isolated_rooms` (`:2180`) | Drop isolated tiny boxes (detections over sample arrows etc.). |
| 7 | **Wall-endpoint snapping** (brief Stage D) | `_snap_shared_walls` (`:3202`) | Reuse ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â don't re-derive. |
| 8 | **Geometry gate** (brief Gate B) | `_vector_plan_geometry_is_usable` (`:1936`) | Per-floor normalisation (validate each floor against its OWN region, not the whole sketch ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â else multi-floor plans are wrongly rejected). |
| 9 | **Uniform scaling** | n/a (bug to avoid) | Scale X and Y by the SAME factor + letterbox; never stretch 1000ÃƒÆ’Ã¢â‚¬â€1000 onto A3's 1.41:1 (squashes right angles). |
| 10 | **ACM fill = light pink** | COM uses `RGB(245,208,212)` | Match in the native STYLE block so both renderers agree. |

## Non-negotiable: measure, don't just run unit tests
Use `evaluation/score_against_manual.py` + `ground_truth/manual_plans_truth.json` (18 projects) + AlphaTracker sketch pairs (sketch = largest image in each project's `plans@` submission email). Report label/sample/room deltas **before and after**. Box-pipeline baseline: **label 0.89, sample 0.89**. "71 tests green" ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â  "plans improved".

## Decision for the owner
If the end-state is **all-Linux** (no Windows/Visio), the native renderer becomes the single production path and the shared extractor must hold #1ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“#6. The COM renderer then stays as the high-fidelity/optional path. Land geometry/label logic in the **shared extractor**, not renderer-specific code, to stop the two tracks drifting.


---

# Implementation plan (historical)

*(merged from `docs/implementation_plan.md`)*

# Implementation Plan ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Clean Room Labels and Renaming AI Files

This plan details the changes required to ensure that room label normalization, abbreviation mapping (e.g., `BSP`/`BSD` -> `Bed`), candidate room filtering (dropping `G01 CANDIDATES`), and room number extraction are applied across **both** pipelines, followed by an optional renaming pass to eliminate confusing legacy AI brand names.

---

## Proposed Changes

### Phase 1: Clean Room Labels and Numbers (Implemented & Ready)

#### 1. Production Container Pipeline
- **Files**:
  - [utils/gemini_text.py](file:///c:/Projects/AcornPlanGeneration/utils/gemini_text.py)
  - [automation/container/gemini_text.py](file:///c:/Projects/AcornPlanGeneration/automation/container/gemini_text.py)
- **Changes**:
  - Implemented the room abbreviation map `_ROOM_NAME_MAP` and standard `_normalize_room_name`.
  - Added `_parse_room_name_and_number` to split room numbers (e.g., `"008 BSP"` -> `"008"` and `"Bed"`).
  - Modified `_normalize_layout` to clean all rooms, drop candidates, and return `"room_number"` so `generate_plan.py` draws correct numbers.

#### 2. Local Box Pipeline
- **File**:
  - [pipeline.py](file:///c:/Projects/AcornPlanGeneration/pipeline.py)
- **Changes**:
  - Added a call to `_dedup_room_list(ai_data_full)` right after `_apply_panel_floor(ai_data_full)` in `process_sketch`.

---

### Phase 2: Rename AI Files to Sensible Names (Proposed)

To avoid confusion since the pipeline has migrated to ChatGPT (OpenAI), we propose renaming the legacy files to provider-agnostic names.

#### 1. File Renames
- `utils/gemini_text.py` -> `utils/layout_extractor.py`
- `utils/gemini_vision.py` -> `utils/vision_client.py`
- `automation/container/gemini_text.py` -> `automation/container/layout_extractor.py`
- `automation/container/gemini_vision.py` -> `automation/container/vision_client.py`

#### 2. Dependency Updates
If approved, we will update all references in:
- `automation/container/Dockerfile` (the COPY commands)
- `automation/container/generate_plan.py` (the import statements)
- `automation/deploy_plangenration.ps1` (the SCP and SSH deploy commands)
- `plans/llm_extract.py` (the import statement)
- `utils/ai_provider.py` (imports and functions)
- `automation/check_environment.py` and `tests/` (environment check tools and unit tests)

---

## Verification Plan

### Automated Tests
- Run all 180 unit tests to check for regressions:
  ```powershell
  .\.venv\Scripts\pytest
  ```

### Manual Verification
- Re-run the local test command to confirm that `G01 CANDIDATES` is dropped, `BSP` room names are normalized to `Bed`, and the numbers are extracted/drawn correctly in the resulting Visio plan:
  ```powershell
  python main.py --image "C:\Users\SuryaKambala\OneDrive - Acorn Analytical Services\Desktop\Documents\IMAGES\1.jpeg" --output "output\reports\1_gpt4o_yolo_test.vsdx" --renderer com --vector
  ```


---

# Test results (historical)

*(merged from `docs/test_results.md`)*

# Floor Plan Layout Verification Results

This document presents the visual outputs from the consolidated floor plan generation. We tested the **`split_loft`** mode, which merges all main floor plans onto Tab 1 ("Floor Plans") and isolates any Loft floor plans onto Tab 2 ("Loft").

Below is the visual comparison of the hand-drawn surveyor sketches (Input) and the resulting generated Visio page exports (Output) for the 6 tested floor plans.

---

## 1. N-101780 (Reference Case)
This plan contains Ground, First, Second, and Loft floors. Ground, First, and Second floors are consolidated onto Tab 1; Loft is separated onto Tab 2.

````carousel
### Sketch Input
![N-101780 Input](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/N-101780_input.jpg)
<!-- slide -->
### Tab 1 - Floor Plans
![N-101780 Tab 1](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/N-101780_new_page_1_Floor_Plans.png)
<!-- slide -->
### Tab 2 - Loft
![N-101780 Tab 2](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/N-101780_new_page_2_Loft.png)
````

---

## 2. Sketch 1000000050
This plan only contains main floors (no Loft detected), so they are consolidated onto a single page.

````carousel
### Sketch Input
![1000000050 Input](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/1000000050_input.jpg)
<!-- slide -->
### Tab 1 - Floor Plans
![1000000050 Tab 1](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/1000000050_page1_Floor_Plans.png)
````

---

## 3. Sketch 1000000051
This plan only contains main floors, consolidated onto a single page.

````carousel
### Sketch Input
![1000000051 Input](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/1000000051_input.jpg)
<!-- slide -->
### Tab 1 - Floor Plans
![1000000051 Tab 1](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/1000000051_page1_Floor_Plans.png)
````

---

## 4. Sketch 1000000052
This plan only contains main floors, consolidated onto a single page.

````carousel
### Sketch Input
![1000000052 Input](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/1000000052_input.jpg)
<!-- slide -->
### Tab 1 - Floor Plans
![1000000052 Tab 1](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/1000000052_page1_Floor_Plans.png)
````

---

## 5. Sketch 1000000057
This plan contains both main floors (consolidated on Tab 1) and a Loft plan (isolated on Tab 2).

````carousel
### Sketch Input
![1000000057 Input](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/1000000057_input.jpg)
<!-- slide -->
### Tab 1 - Floor Plans
![1000000057 Tab 1](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/1000000057_page1_Floor_Plans.png)
<!-- slide -->
### Tab 2 - Loft
![1000000057 Tab 2](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/1000000057_page2_Loft.png)
````

---

## 6. Sketch 1000000065
This plan only contains main floors, consolidated onto a single page.

````carousel
### Sketch Input
![1000000065 Input](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/1000000065_input.jpg)
<!-- slide -->
### Tab 1 - Floor Plans
![1000000065 Tab 1](/C:/Users/SuryaKambala/.gemini/antigravity/brain/89bf260e-9a89-423e-ab76-baae77e9e6cb/1000000065_page1_Floor_Plans.png)
````

---

## Conclusion
The **`split_loft`** default behavior operates successfully:
1. When no loft is present, all plans reside beautifully on Page 1 (e.g. `1000000050`, `1000000051`, `1000000052`, `1000000065`).
2. When a loft is present, main floors are grouped together on Page 1, and the Loft is placed on Page 2 (e.g. `N-101780`, `1000000057`).


---

# Walkthrough (historical)

*(merged from `docs/walkthrough.md`)*

# Walkthrough ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Room Labels Cleanup & File Renaming

We have completed the room label cleanup and file renaming tasks across both the local box pipeline and the production container pipeline. Additionally, the unit test suite has been removed per your request to keep the workspace clean.

---

## Changes Implemented

### 1. Room Label & Number Cleanup (Phase 1)
- **Production Container Pipeline (`layout_extractor.py`)**:
  - Implemented the `_ROOM_NAME_MAP` and standard `_normalize_room_name` functions to clean abbreviations (`BSP`/`BSD` -> `Bed`, `ESU Suite` -> `En-Suite`, `Bath` -> `Bathroom`).
  - Added `_parse_room_name_and_number` to split room numbers from the name (e.g. `"008 BSP"` -> name: `"Bed"`, number: `"008"`) and write them under `"room_number"`.
  - Added candidate room filtering (drops any rooms whose names contain `candidate`).
- **Local Box Pipeline (`pipeline.py`)**:
  - Called `_dedup_room_list(ai_data_full)` right after `_apply_panel_floor(ai_data_full)` in `process_sketch`.

### 2. File Renaming (Phase 2)
Renamed the legacy Gemini files to sensible, provider-agnostic names to reflect that we migrated to ChatGPT (OpenAI):
- `utils/gemini_text.py` ÃƒÂ¢Ã…Â¾Ã¢â‚¬Â `utils/layout_extractor.py`
- `utils/gemini_vision.py` ÃƒÂ¢Ã…Â¾Ã¢â‚¬Â `utils/vision_client.py`
- `automation/container/gemini_text.py` ÃƒÂ¢Ã…Â¾Ã¢â‚¬Â `automation/container/layout_extractor.py`
- `automation/container/gemini_vision.py` ÃƒÂ¢Ã…Â¾Ã¢â‚¬Â `automation/container/vision_client.py`
- Updated all references in `Dockerfile`, `generate_plan.py`, `deploy_plangenration.ps1`, `llm_extract.py`, `ai_provider.py`, `check_environment.py`, and documentation files.

### 3. Cleanup
- Removed the `tests/` directory from the repository to clean up the workspace, committing and pushing the change to `main`.

---

## Verification Results

### 1. Unit Tests
All **180 tests passed** successfully before the `tests/` folder was deleted:
```text
tests\test_layout_provider_fallback.py .....                             [ 28%]
tests\test_room_label_cleanup.py ...                                     [ 90%]
============================= 180 passed in 6.53s =============================
```

### 2. Local Manual Verification
Running the professional plan generator locally:
```bash
python automation/container/generate_plan.py N-101780 --image "C:\Users\SuryaKambala\OneDrive - Acorn Analytical Services\Desktop\Documents\IMAGES\1.jpeg" --output "output\reports\test_output.svg"
```
Produced the following results in the quality sidecar:
- **Expected Rooms**: `BED` (normalized from `BSP`), `En-Suite`, `OFFICE`, and `Bathroom`.
- **Candidates**: `0` (successfully filtered out and dropped).
- **Coordinate Integrity**: Checked and fully in bounds.
- **Output File**: [test_output.vsdx](file:///c:/Projects/AcornPlanGeneration/output/reports/test_output.vsdx).


---

# SAM annotation brief (training)

*(merged from `SAM_ANNOTATION_BRIEF.md`)*

# Annotation brief ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â next batch of Acorn floor plan sketches

For Sam (and any future annotator). Read once, keep handy while annotating in Roboflow.

## Goal

We have a trained model that finds rooms, ACM areas, doors, etc. in your hand-drawn sketches and produces Visio (.vsdx) output. It works, but **needs more training data to get reliably accurate**. Aim for **200ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“500 newly annotated sketches per batch.**

## What "new" means

- **New site surveys**, not re-annotating sketches you've already done.
- Filenames don't matter ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the pipeline handles duplicates. But the *content* of the photo/scan must be a sketch the dataset hasn't seen before.

## The 9 classes to use ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â exact spelling, case-sensitive

| Class name | What it is |
|---|---|
| `room` | Any room rectangle (Kitchen, Bedroom, Lounge, Hall, Loft, etc. ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â all collapse to `room`) |
| `acm` | ACM-containing area (the diagonal-hatched zones you mark) |
| `background` | Grid paper / page background |
| `door` | Door symbol (arc + line) |
| `CupBoard` | Cupboards |
| `Loft Hatch` | The X marks indicating loft access |
| `stairs` | Stair hatching |
| `text` | Sample IDs (S001, S002ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦), material codes (Felt, TC, Mastic), room labels, header text |
| `wall` | Drawn wall lines |

**Watch the spelling**: `CupBoard` (capital B), `Loft Hatch` (with a space). If you use any new class name, the pipeline will drop those annotations until we add a mapping ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â so stick to these 9.

## Where to focus your effort

Some classes are very thin in the current training set. Extra care on these gives the biggest accuracy lift:

| Class | Current count | Priority |
|---|---:|---|
| **wall** | 197 | **Highest** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â model currently can't predict walls at all |
| **CupBoard** | 325 | High |
| **stairs** | 437 | High |
| **acm** | 951 | Medium |
| **door** | 3,216 | Already plenty ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â but please **draw tight polygons** (see below) |
| room, text, background | 4,000+ each | Already strong |

If you're short on time, prioritise sketches that have **walls, cupboards, ACM areas, and stairs** drawn on them.

## Quality tips ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â what matters most

### 1. Use polygons, not bounding boxes

In Roboflow, select the **polygon** annotation tool, not the rectangle tool. The model is a *segmentation* model ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â it needs the actual shape outlines, not just rectangles around features. (For room rectangles, the polygon is just 4 corners ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â quick to do.)

### 2. Tight polygons, especially for doors

The current door polygons in training are *too loose* ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â they include too much grid paper around the door arc. This is the #1 reason the model struggles with doors.

When you annotate a door:
- Trace **just the arc + the door line** (the visible drawn parts).
- Don't include the surrounding empty grid squares.
- A door polygon should be small and tight, not a big square around the door symbol.

### 3. Walls ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â only the lines, not the rooms next to them

When annotating walls, trace only the **wall line itself** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â don't extend the polygon into the rooms on either side.

### 4. ACM areas ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â full hatched region

ACM polygons should cover the **entire hatched area** you've drawn, edge to edge. If the hatch is in a corner of a room, the ACM polygon covers just that corner, not the whole room.

### 5. Text ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â boxes around the text, not letter-shaped

For text (S001, S002, material codes, room labels) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â use the polygon as a tight rectangle around the text region. The model only needs to know WHERE text is; the actual reading of the characters happens in a separate step.

## Roboflow workflow

1. **Project**: same Roboflow project you've been using (`acorn-floor-plans` in workspace `softwares-workspace-z2kih`). Don't create a new project.

2. **Upload new sketches** to the project's image queue.

3. **Annotate** using the polygon tool with the 9 class names above.

4. **When the batch is done**:
   - Generate a new version in Roboflow
   - Export as **YOLOv12 segmentation** OR **COCO segmentation** (both work)
   - Download the zip and send it to Surya
   - Surya will run the cleanup ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ merge ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ retrain pipeline (see `training/DATASET_WORKFLOW.md`)

## What to expect after the next batch

Realistic projections with 200ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“500 new well-annotated sketches:

| Metric | Now | After next batch |
|---|---:|---:|
| Room detection (box AP) | ~0.71 | likely hits 0.78ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“0.82 (milestone) |
| Door detection | improving with cleaner polygons | meaningful jump |
| Wall detection | unusable | becomes usable IF 200+ new wall annotations |

## Questions for Surya

If you hit a sketch type that doesn't fit the 9 classes (e.g. a boiler symbol, a chimney detail, a new annotation convention), **don't force it into an existing class**. Ask first ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â we may add a new class name to the schema.
