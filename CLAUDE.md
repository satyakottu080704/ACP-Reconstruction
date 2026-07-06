# CLAUDE.md

This file is the working memory for Claude Code / future agents in this repository.

## Current Truth - 2026-06-23

This repo converts hand-drawn asbestos-survey sketches into professional Visio (`.vsdx`) floor plans: walls, doors, numbered rooms, ACM shading, stair symbols, sample markers, and multi-floor pages.

Current production path is the Windows render service. The old Linux-native container path and the old SharePoint polling Windows agent are superseded/fallback paths.

```text
Outlook Plans inbox
  -> n8n workflow: automation/n8n/acorn_plans_renderer.json
  -> GET /ready on Windows VM render_service.py
  -> POST sketch bytes to /render?project=N-xxxxx
  -> pipeline.py:process_sketch using YOLO + OpenAI labels
  -> utils/visio/professional_visio.py via Microsoft Visio COM
  -> n8n uploads the returned VSDX to SharePoint AI_COMPLETED
  -> n8n marks the email read and moves it to Outlook AI_COMPLETED
```

Production remains REVIEW-ONLY. Uploads go to `AI_COMPLETED`, but do not treat them as approved/unattended publishing until the eval gate and surveyor sign-off are complete.

## Current Runtime Facts

| Area | Current value |
|---|---|
| Production entry | `automation/render_service.py` (`GET /ready`, `POST /render`) |
| n8n workflow | `automation/n8n/acorn_plans_renderer.json` |
| Renderer | Windows + Microsoft Visio COM through `utils/visio/professional_visio.py` |
| Pipeline | `pipeline.py:process_sketch` box pipeline |
| AI provider | OpenAI only for production review runs |
| AI model | `OPENAI_VISION_MODEL=gpt-4o`, `OPENAI_LABEL_ATTEMPTS=1` |
| OpenAI timeout | `OPENAI_REQUEST_TIMEOUT_SECONDS`, default 60 seconds; use 45 seconds for interactive testing |
| YOLO model | `ACORN_MODEL_PATH`, default `models/weights/best.pt`, fallback `training/Training/weights/best.pt` |
| YOLO classes | `acm`, `door`, `floor`, `room`, `stairs`, `walls` from `config.py` |
| SharePoint target | `General/AI Automation/AI_COMPLETED` |
| Email handling | Mark read, find/create Outlook `AI_COMPLETED`, move processed email there |

## Superseded / Fallback Paths

- `automation/container/process_plan.py`, `generate_plan.py`, and the `plangenration` container are the old Linux-native path. Keep as fallback/historical only.
- `automation/windows_visio_agent.py` polling `Pending_Draw` is the old SharePoint-mediated Windows agent design. The active design is HTTP render service called by n8n.
- `PLAN_LAYOUT_PROVIDERS`, Gemini, Groq, and Ollama are not part of the current production path. Do not re-enable provider fallback without a measured improvement on the same ground-truth set.
- `gpt-4o-mini` is useful for comparisons only; it is not the current production-quality review setting.

## Commands

```powershell
pip install -r requirements.txt

# Main local/VM box pipeline.
python main.py --image "sketch.jpg" --output out.vsdx
python main.py --image "sketch.jpg" --renderer com
python main.py --image "sketch.jpg" --overlay
python main.py --image "sketch.jpg" --no-model
python main.py --image "sketch.jpg" --model-only
python main.py --batch "folder/" --resume
python main.py --clear-cache

# Current review-mode local test command.
cd C:\Projects\AcornPlanGeneration
.\.venv\Scripts\Activate.ps1
$env:OPENAI_VISION_MODEL='gpt-4o'
$env:OPENAI_LABEL_ATTEMPTS='1'
$env:OPENAI_REQUEST_TIMEOUT_SECONDS='45'
$env:DRAW_OUTPUT_MODE='vector'
$env:PLAN_PUBLISH_MODE='review'
$env:VISIO_COM_TIMEOUT_SECONDS='120'
$env:PLAN_ALLOW_UNCERTAIN_VECTOR='false'
Remove-Item Env:\VISIO_DRAW_MODEL_CONTOURS -ErrorAction SilentlyContinue
.\.venv\Scripts\python main.py --image "C:\Projects\AcornPlanGeneration\input\N-108889.jpg" --output "C:\Projects\AcornPlanGeneration\output\reports\N-108889_review_test.vsdx" --renderer com --vector

# Render service on the Windows VM.
python automation/render_service.py

# Tests and eval gates.
pytest -q
pytest tests/test_manual_eval_gate.py -q
$env:ACORN_RUN_EVAL_GATE='1'; python -m pytest tests/test_manual_eval_gate.py -q
python evaluation/score_against_manual.py --gate --json --max-projects 10

# YOLO training.
python train_floorplans.py
```

Tests that need `scipy`, `Flask`, or Visio will fail in a bare environment; install requirements first and run Visio tests only on the Windows VM/local Windows desktop.

## Architecture

The box pipeline (`pipeline.py:process_sketch`) preprocesses the survey image, runs YOLO geometry, reads labels with OpenAI `gpt-4o`, merges geometry and labels in `merge_results`, writes a `.quality.json` sidecar, and renders through Visio COM for production-quality output.

YOLO class IDs are mapped dynamically from `model.names`. Do not hardcode class indexes.

The render service (`automation/render_service.py`) exposes `/health`, `/ready`, and `/render`, warms Visio COM at boot, serializes render calls with a lock because Visio COM is single-render safe, returns `X-Review-Required` and `X-Quality-Flags`, and cleans temp input/output files after each request.

The n8n Windows renderer workflow (`automation/n8n/acorn_plans_renderer.json`) polls the Plans inbox, waits for `/ready`, downloads the first image attachment, calls `/render`, uploads the returned VSDX to SharePoint `AI_COMPLETED`, then marks the source email read, finds or creates the Outlook `AI_COMPLETED` folder, and moves the processed email there. Failed render/upload paths loop back to the batch or leave the item unread for retry.

## Renderer Rules

- `utils/visio/professional_visio.py` is the production renderer.
- Always draw the clean editable wall/door renderer. YOLO mask contours are evidence only.
- Production contour overlay is disabled by default. Enable only for diagnostics with `VISIO_DRAW_MODEL_CONTOURS=true`.
- Stairs must use the central Acorn-style editable stair symbol. Do not draw ad-hoc text boxes or duplicate stair implementations.
- Doors should use the professional swing/arc renderer and stay attached to walls.
- Keep non-loft sections together on the Floor Plans tab. Split Loft to a separate tab only when the pipeline identifies a loft.

## Gotchas

- Plan AI is fixed to OpenAI `gpt-4o` for production review. Gemini was dropped because it produced rate-limit failures and malformed/truncated JSON on real sketches.
- `config.py` is the current model source of truth: `MODEL_PATH`, `MODEL_IMGSZ`, `MODEL_CONF_THRESHOLD`, and `CLASSES`.
- To test new weights, set `ACORN_MODEL_PATH`; do not hardcode a new path.
- Visio COM needs an interactive Windows desktop session. Do not run the render service as a Session-0 Windows service.
- Review-first is intentional. A VSDX opening in Visio does not prove geometry correctness.
- The root `README.md` may be stale. Trust `CLAUDE.md`, `DOCS.md`, `config.py`, `automation/render_service.py`, and the current n8n workflow JSON.

## Production Status And Decisions

Status: deployed but REVIEW-ONLY, not approved for unattended publishing.

The blocker is measured accuracy, not only renderer polish. A good-looking VSDX can still be geometrically wrong.

### The accuracy problem has two jobs

- Geometry: room count and room placement. The YOLO model provides the best current geometry signal, but it still needs measured validation.
- Text: room names, room numbers, floors, samples, and handwriting. OpenAI `gpt-4o` is currently the most reliable option.

Free OCR and local OCR are useful baselines only; they are not production-ready for handwriting on these survey sketches.

### Evaluation harness

- `plans/llm_extract.py`: AI predictor.
- `plans/free_local_extract.py`: free/OCR baseline.
- `plans/ground_truth_eval.py`: scorer.
- `evaluation/score_against_manual.py`: regression/eval gate.

Every model, prompt, or renderer change should be judged against the same ground-truth set. Do not ship a change based only on one screenshot.

### Current open issues

- Geometry accuracy still needs a signed-off benchmark set.
- Some sketches are still too ambiguous for unattended output.
- n8n should remain review-first until surveyor sign-off on a larger batch.
- Long-term debt: `pipeline.py` is large and regression-prone; do not refactor mid-delivery without a golden-sketch regression gate.

## Current Production Notes - 2026-06-23

- Latest local tests are about 90 percent acceptable visually, but still not approved for unattended publishing.
- `models/weights/best.pt` is now the preferred default model path. It is the extracted current model. `training/Training/weights/best.pt` is fallback only.
- `main.py --model-only` prints the active YOLO model path and SHA so model drift is visible.
- `OPENAI_REQUEST_TIMEOUT_SECONDS` was added so OpenAI calls cannot look silently frozen during poor API/network conditions.
- OpenAI 429 quota/billing failures now fail fast and can reuse stale cached labels for AI_COMPLETED review output instead of wasting retries.
- `PLAN_RETRY_INVALID_AI_LAYOUT=false` is the review-mode default. It prevents extra GPT geometry retry calls when full-layout boxes fail validation; the invalid layout is rejected and the pipeline continues to YOLO/labels or overlay fallback.
- `VISIO_DRAW_MODEL_CONTOURS` is off by default because raw mask contours made borders/doors look broken. Leave it off for production/review output.
- Generated local VSDX files get a `.quality.json` sidecar with `requires_review`, `quality_flags`, room count, sample count, floors, and input metrics.
- If reconstructed vector geometry fails safety checks, `PLAN_VECTOR_REJECT_MODE=overlay` exports a source-faithful review VSDX instead of failing the batch item. The sidecar gets `vector_geometry_rejected` and `source_overlay_fallback`.
- n8n workflows should upload drafts to SharePoint `General/AI Automation/AI_COMPLETED` by default.
- Keep `AI_COMPLETED` as the only destination folder, but do not treat those drafts as unattended production until acceptance gates and surveyor sign-off are complete.
- n8n workflow JSON must not contain real tokens. Use environment variables for `RENDER_SERVICE_URL` and `RENDER_SERVICE_TOKEN`.