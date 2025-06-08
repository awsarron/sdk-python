"""Multi-agent orchestration.

This module provides multi-agent patterns as first-class
SDK concepts with full Agent customization and tool-based coordination.
"""

from .base import MultiAgentBase, MultiAgentResult
from .graph import GraphBuilder, GraphResult

# TODO: add swarm imports

__all__ = [
    "GraphBuilder",
    "GraphResult",
    "MultiAgentBase",
    "MultiAgentResult",
]
