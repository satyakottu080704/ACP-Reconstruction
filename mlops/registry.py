"""
registry.py — versioned model weight registry.

Tracks every trained model with:
  - version number (auto-incremented)
  - file path + SHA256
  - training metrics (box_map50, mask_map50, epoch, dataset)
  - "best" pointer (only promoted by deploy.py regression gate)
  - publish timestamp

Registry is a single JSON file at REGISTRY_PATH (default: mlops/registry.json).

Usage:
  from mlops.registry import ModelRegistry
  reg = ModelRegistry()
  ver = reg.register("runs/.../best.pt", box_map50=0.883, mask_map50=0.803, epoch=200)
  reg.set_best(ver)   # only call after regression gate passes
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

_DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "mlops" / "registry.json"
_DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "models" / "versions"


class ModelRegistry:
    def __init__(
        self,
        registry_path: Optional[str] = None,
        weights_dir: Optional[str] = None,
    ):
        self.registry_path = Path(registry_path or _DEFAULT_REGISTRY)
        self.weights_dir = Path(weights_dir or _DEFAULT_WEIGHTS_DIR)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"versions": [], "best": None}

    def _save(self):
        self.registry_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── register ──────────────────────────────────────────────────────────────

    def register(
        self,
        weights_path: str,
        box_map50: float = 0.0,
        mask_map50: float = 0.0,
        epoch: int = 0,
        dataset: str = "",
        notes: str = "",
        copy_weights: bool = True,
    ) -> int:
        """
        Register a new model version.

        Args:
            weights_path: path to the .pt file.
            box_map50:    box mAP@50 from validation.
            mask_map50:   mask mAP@50 from validation.
            epoch:        training epoch at which weights were saved.
            dataset:      dataset name/description.
            notes:        free-form notes.
            copy_weights: if True, copy weights into the versions directory.

        Returns:
            version number (int).
        """
        wp = Path(weights_path)
        if not wp.exists():
            raise FileNotFoundError(f"Weights not found: {wp}")

        sha = _sha256(wp)
        version = len(self._data["versions"]) + 1

        dest = wp
        if copy_weights:
            dest = self.weights_dir / f"v{version}_{wp.name}"
            shutil.copy2(wp, dest)

        entry = {
            "version": version,
            "path": str(dest),
            "sha256": sha,
            "box_map50": box_map50,
            "mask_map50": mask_map50,
            "epoch": epoch,
            "dataset": dataset,
            "notes": notes,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "promoted": False,
        }
        self._data["versions"].append(entry)
        self._save()
        print(f"[registry] registered v{version}: box={box_map50:.4f} mask={mask_map50:.4f} sha={sha[:8]}")
        return version

    # ── best pointer ──────────────────────────────────────────────────────────

    def set_best(self, version: int):
        """Mark a version as the current best (production model)."""
        entry = self.get_version(version)
        if entry is None:
            raise ValueError(f"Version {version} not found in registry")
        entry["promoted"] = True
        self._data["best"] = version
        self._save()
        # Update the canonical best.pt symlink / copy
        best_dest = Path(__file__).resolve().parents[1] / "models" / "weights" / "best.pt"
        best_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry["path"], best_dest)
        print(f"[registry] set best → v{version}: {entry['path']}")

    def get_best(self) -> Optional[Dict[str, Any]]:
        """Return the current best version entry, or None."""
        best_v = self._data.get("best")
        if best_v is None:
            return None
        return self.get_version(best_v)

    def get_version(self, version: int) -> Optional[Dict[str, Any]]:
        for entry in self._data["versions"]:
            if entry["version"] == version:
                return entry
        return None

    def list_versions(self) -> List[Dict[str, Any]]:
        return list(self._data["versions"])

    def publish(self) -> Dict[str, Any]:
        """Return a summary dict suitable for logging / reporting."""
        best = self.get_best()
        return {
            "total_versions": len(self._data["versions"]),
            "best_version": self._data.get("best"),
            "best_box_map50": best["box_map50"] if best else None,
            "best_mask_map50": best["mask_map50"] if best else None,
            "registry_path": str(self.registry_path),
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
