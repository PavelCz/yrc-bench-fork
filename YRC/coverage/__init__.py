"""
YRC-specific coverage algorithms for efficient sampling of monotonic curves.

This module provides YRC-specific wrappers around the ABCS library.
"""

# Re-export the ABCS joint sampler types for convenience
from abcs import JointCoverageSampler, CurvePoint, SamplingResult

# Import YRC-specific wrapper functions
from .binary_search import create_threshold_sampler

__all__ = [
    'JointCoverageSampler',
    'CurvePoint',
    'SamplingResult',
    'create_threshold_sampler',
]