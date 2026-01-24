"""
Provides core logic for PEFT (Parameter-Efficient Fine-Tuning) application and base definitions.
"""

from .applicator import inject_peft_and_freeze, merge_peft
from .base import PeftInjectionResult, PeftMethod

__all__ = [
    "PeftInjectionResult",
    "PeftMethod",
    "inject_peft_and_freeze",
    "merge_peft"
]
