"""Core logic and base definitions for PEFT (Parameter-Efficient Fine-Tuning)."""

from .applicator import inject_peft_and_freeze, merge_peft
from .base import PeftInjectionResult, PeftMethod

__all__ = [
    "PeftInjectionResult",
    "PeftMethod",
    "inject_peft_and_freeze",
    "merge_peft",
]
