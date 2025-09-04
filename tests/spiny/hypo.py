import os

from hypothesis import settings


def configure_hypo() -> None:
    settings.register_profile("fast", max_examples=10)
    if os.environ.get("HYPO_SLOW") != "1":
        settings.load_profile("fast")
