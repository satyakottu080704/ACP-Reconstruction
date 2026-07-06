"""
exporters — v2 plan exporters (all consume PlanModel).

Available formats: json, png, svg, dxf, pdf, vsdx
"""
from .json_export import export_json
from .png_export import export_png
from .svg_export import export_svg
from .dxf_export import export_dxf
from .pdf_export import export_pdf
from .vsdx_export import export_vsdx

from pathlib import Path
from typing import Dict, List, Optional, Any

DEFAULT_FORMATS = ["json", "svg", "dxf", "png", "pdf"]

_EXPORTERS = {
    "json": export_json,
    "png":  export_png,
    "svg":  export_svg,
    "dxf":  export_dxf,
    "pdf":  export_pdf,
    "vsdx": export_vsdx,
}


def export_all(
    plan,
    out_dir,
    stem: str = "plan",
    formats: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Export a PlanModel to all requested formats.

    Returns:
        Dict mapping format name → output file path (or error message).
    """
    if formats is None:
        formats = DEFAULT_FORMATS

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for fmt in formats:
        fn = _EXPORTERS.get(fmt.lower())
        if fn is None:
            results[fmt] = f"ERROR: unknown format '{fmt}'"
            continue
        try:
            out_path = fn(plan, out_dir / f"{stem}.{fmt.lower()}")
            results[fmt] = str(out_path)
        except Exception as e:
            results[fmt] = f"ERROR: {e}"

    return results


__all__ = [
    "export_json", "export_png", "export_svg", "export_dxf",
    "export_pdf", "export_vsdx", "export_all", "DEFAULT_FORMATS",
]
