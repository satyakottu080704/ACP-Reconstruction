"""
streamlit_app.py — Acorn v2 floor-plan reconstruction UI.

Upload a sketch → YOLO segmentation → OCR → reconstruct → download.

Run:
  streamlit run ui/streamlit_app.py

Requires: streamlit (pip install streamlit)
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import streamlit as st
except ImportError:
    print("streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

import numpy as np
import cv2


def main():
    st.set_page_config(
        page_title="Acorn Floor Plan Reconstruction",
        page_icon="🏠",
        layout="wide",
    )

    st.title("🏠 Acorn Floor Plan Reconstruction (v2)")
    st.caption("Upload a survey sketch → segment → reconstruct → download")

    engine = os.environ.get("RECONSTRUCTION_ENGINE", "legacy")
    if engine != "v2":
        st.warning(
            "⚠️ RECONSTRUCTION_ENGINE is not set to 'v2'. "
            "Set `RECONSTRUCTION_ENGINE=v2` in your `.env` to enable the v2 path."
        )

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Confidence threshold", 0.05, 0.5, 0.15, 0.01,
                         help="Lower = more detections; recommended 0.15–0.25")
        run_ocr = st.checkbox("Run OCR", value=True, help="Detect room labels via OCR")
        export_formats = st.multiselect(
            "Export formats",
            ["json", "png", "svg", "dxf", "pdf"],
            default=["json", "png", "svg"],
        )
        st.divider()
        st.caption("Model info")
        try:
            import config as _cfg
            st.text(f"Classes: {_cfg.CLASSES}")
            st.text(f"imgsz: {_cfg.MODEL_IMGSZ}")
            st.text(f"conf: {_cfg.MODEL_CONF_THRESHOLD}")
        except Exception:
            st.text("config.py not loaded")

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a floor-plan sketch (JPG/PNG)",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded is None:
        st.info("Upload a sketch image to get started.")
        return

    file_bytes = uploaded.read()
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not decode the uploaded image.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input sketch")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ── Run inference ─────────────────────────────────────────────────────────
    with st.spinner("Running segmentation..."):
        try:
            from ultralytics import YOLO
            import config as _cfg
            model = YOLO(_cfg.MODEL_PATH)
            results = model(img, imgsz=_cfg.MODEL_IMGSZ, conf=conf, verbose=False)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return

    with st.spinner("Reconstructing floor plan..."):
        try:
            from reconstruction.engine import from_yolo_result
            plan = from_yolo_result(results[0], source_image=img, run_ocr=run_ocr)
        except Exception as e:
            st.error(f"Reconstruction failed: {e}")
            return

    # ── Results summary ───────────────────────────────────────────────────────
    st.success(
        f"✅ Found {len(plan.rooms)} rooms, {len(plan.doors)} doors, "
        f"{len(plan.stairs)} stairs, loft={'yes' if plan.has_loft else 'no'}"
    )

    # Quality report
    with st.expander("Quality report"):
        from reconstruction.quality import quality_report_dict
        qr = quality_report_dict(plan)
        st.json(qr)
        if qr.get("needs_review"):
            st.warning(f"⚠️ Needs review: {qr.get('review_flags', [])}")

    # Room list
    with st.expander("Room list"):
        for i, r in enumerate(plan.rooms):
            tag = "🔴 ACM" if r.is_acm else ("🏠 Loft" if r.is_loft else "🟢")
            st.write(f"{tag} Room {i}: **{r.label or 'unlabelled'}** "
                     f"(floor {r.floor_idx}, conf={r.confidence:.2f})")

    # Connectivity
    with st.expander("Room connectivity"):
        st.json(plan.connectivity)

    # ── Render preview ────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # PNG preview
        try:
            from exporters.png_export import export_png
            png_path = tmp_path / "plan.png"
            export_png(plan, png_path)
            with col2:
                st.subheader("Reconstructed plan")
                preview = cv2.imread(str(png_path))
                if preview is not None:
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
        except Exception as e:
            with col2:
                st.warning(f"PNG preview failed: {e}")

        # ── Downloads ─────────────────────────────────────────────────────────
        st.subheader("Downloads")
        dl_cols = st.columns(len(export_formats))

        from exporters import export_all
        exported = export_all(plan, tmp_path, stem="plan", formats=export_formats)

        for col_el, fmt in zip(dl_cols, export_formats):
            path_str = exported.get(fmt, "")
            if path_str.startswith("ERROR"):
                col_el.error(f"{fmt}: {path_str}")
            else:
                ep = Path(path_str)
                if ep.exists():
                    col_el.download_button(
                        label=f"⬇ {fmt.upper()}",
                        data=ep.read_bytes(),
                        file_name=ep.name,
                        mime="application/octet-stream",
                    )


if __name__ == "__main__":
    main()
