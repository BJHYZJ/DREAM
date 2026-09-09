from .encoders import get_encoder

__all__ = ["get_encoder", "OvmmPerception", "create_semantic_sensor"]


def __getattr__(name):
    # An encoder does not need the real-robot semantic sensor or its optional
    # dependencies. Keep the public imports compatible while loading those
    # dependencies only when the caller actually requests the wrapper.
    if name in {"OvmmPerception", "create_semantic_sensor"}:
        from . import wrapper

        return getattr(wrapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
