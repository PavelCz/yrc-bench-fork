"""
YRC-specific coverage algorithms for efficient sampling of monotonic curves.

This module provides YRC-specific wrappers around the ABCS library.
"""

# Import generic ABCS classes
from abcs import BinarySearchSampler, SamplePoint

# Import YRC-specific wrapper functions
from .binary_search import create_threshold_sampler

__all__ = [
    'BinarySearchSampler',
    'SamplePoint', 
    'create_threshold_sampler',
]