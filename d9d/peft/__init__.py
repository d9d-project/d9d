"""Core logic and base definitions for parameter-efficient fine-tuning (PEFT)."""

from .applicator import inject_peft_and_freeze, merge_peft
from .base import PeftInjectionResult, PeftMethod

__all__ = [
    "PeftInjectionResult",
    "PeftMethod",
    "inject_peft_and_freeze",
    "merge_peft",
]
