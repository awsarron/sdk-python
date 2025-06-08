"""Multi-agent orchestration.

This module provides multi-agent patterns as first-class
SDK concepts with full Agent customization and tool-based coordination.
"""

from .base import AgentResult, MultiAgentBase
from .swarm import Swarm, SwarmExecutionConfig

__all__ = [
    "MultiAgentBase",
    "AgentResult",
    "Swarm",
    "SwarmExecutionConfig",
]
