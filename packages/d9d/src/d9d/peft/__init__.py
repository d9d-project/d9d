"""Core logic and base definitions for parameter-efficient fine-tuning (PEFT)."""

from .applicator import inject_peft_and_freeze, merge_peft
from .base import PeftInjectionResult, PeftMethod
from .factory import AnyPeftConfig, peft_method_from_config

__all__ = [
    "AnyPeftConfig",
    "PeftInjectionResult",
    "PeftMethod",
    "inject_peft_and_freeze",
    "merge_peft",
    "peft_method_from_config",
]
