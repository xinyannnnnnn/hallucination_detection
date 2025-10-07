"""
Self-evaluation implementation for low-resource languages.

This package implements the self-evaluation baseline from the HaloScope paper
for Armenian, Basque, and Tigrinya languages.
"""

from .main import (
    LowResourceDatasetLoader,
    SelfEvaluationTemplate,
    SelfEvaluationExperiment,
    HF_NAMES
)

__version__ = "1.0.0"
__author__ = "HaloScope Project"

__all__ = [
    "LowResourceDatasetLoader",
    "SelfEvaluationTemplate", 
    "SelfEvaluationExperiment",
    "HF_NAMES"
]