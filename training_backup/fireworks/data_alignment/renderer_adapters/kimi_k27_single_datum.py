"""Kimi K2.7 Code renderer with an explicit preserve-history extension contract."""

from __future__ import annotations

from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.tokenizer_utils import Tokenizer
from training.renderer.kimi_k27_code import KimiK27CodeRenderer


class KimiK27CodeSingleDatumRenderer(KimiK27CodeRenderer):
    """Preserve historical thinking so every conversation prefix is stable."""

    @property
    def has_extension_property(self) -> bool:
        return True


def _factory(
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
) -> KimiK27CodeSingleDatumRenderer:
    return KimiK27CodeSingleDatumRenderer(
        tokenizer,
        image_processor=image_processor,
    )


register_renderer("kimi_k27_code_single_datum", _factory)
