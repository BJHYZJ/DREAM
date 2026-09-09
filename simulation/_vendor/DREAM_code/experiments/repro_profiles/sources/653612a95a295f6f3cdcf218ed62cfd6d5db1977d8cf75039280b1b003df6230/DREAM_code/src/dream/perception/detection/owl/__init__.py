from .owl_perception import OwlPerception

__all__ = ["OwlPerception", "OWLSAMProcessor"]


def __getattr__(name):
    if name == "OWLSAMProcessor":
        from .owlsam_perception import OWLSAMProcessor

        return OWLSAMProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
