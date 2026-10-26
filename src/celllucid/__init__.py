"""CellLucid's frozen-backbone topic modeling package."""

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["TopicModel", "FrozenTranscriptFormer", "TopicBank"]


def __getattr__(name):
    locations = {
        "TopicModel": ".models.topic",
        "FrozenTranscriptFormer": ".models.backbone",
        "TopicBank": ".tasks.topic_bank",
    }
    if name not in locations:
        raise AttributeError(name)
    return getattr(import_module(locations[name], __name__), name)
