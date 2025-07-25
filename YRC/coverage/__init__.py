"""
Coverage algorithms for efficient sampling of monotonic curves.
"""

from .binary_search import (
    BinarySearchSampler,
    SamplePoint,
    create_threshold_sampler,
)

__all__ = [
    'BinarySearchSampler',
    'SamplePoint', 
    'create_threshold_sampler',
]