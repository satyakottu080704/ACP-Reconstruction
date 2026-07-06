"""Room Detection Module - preprocessing, model inference, color analysis.

Legacy room detection is loaded lazily so lightweight imports such as
``utils.room_detection.preprocessing`` do not pull the full detector stack.
"""

__all__ = ["detect_rooms_multi_strategy", "detect_geometry_only"]


def __getattr__(name):
    if name in __all__:
        from .detector_legacy import detect_geometry_only, detect_rooms_multi_strategy
        return {
            "detect_rooms_multi_strategy": detect_rooms_multi_strategy,
            "detect_geometry_only": detect_geometry_only,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
