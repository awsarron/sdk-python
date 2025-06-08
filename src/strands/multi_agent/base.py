"""Multi-Agent Base Class for Strands SDK.

Provides minimal foundation for multi-agent patterns (Swarm, AgentGraph, Workflow)
with essential Agent management capabilities.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import existing Strands types
from ..agent import Agent

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from agent execution."""

    agent_name: str
    task_id: str
    status: str
    result: Any
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiAgentBase(ABC):
    """Base class for multi-agent helpers.

    This class integrates with existing Strands Agent instances and provides
    multi-agent orchestration capabilities.
    """

    def __init__(self, agents: List[Agent]):
        """Initialize multi-agent base class.

        Args:
            agents: List of pre-configured Agent instances
        """
        self.agents = agents

        self._validate_setup()

    def _validate_setup(self) -> None:
        """Validate configuration."""
        if not self.agents:
            raise ValueError("Must provide agents list")

        # Validate agents have names
        names = []
        for i, agent in enumerate(self.agents):
            if not hasattr(agent, "name") or not agent.name:
                raise ValueError(f"Agent {i} must have a name attribute")
            names.append(agent.name)

        # Agent names must be unique
        if len(names) != len(set(names)):
            raise ValueError("Agent names must be unique")

    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get agent by name.

        Args:
            agent_name: Name of the agent to find

        Returns:
            Agent instance if found, None otherwise
        """
        for agent in self.agents:
            if getattr(agent, "name", None) == agent_name:
                return agent
        return None

    @abstractmethod
    async def execute(self, task: str) -> List[AgentResult]:
        """Execute task with multi-agent pattern (implemented by subclasses)."""
        raise NotImplementedError("execute not implemented")

    @abstractmethod
    async def resume_from_user_input(self, user_response: str) -> List[AgentResult]:
        """Resume task with multi-agent pattern after user provides input (implemented by subclasses)."""
        raise NotImplementedError("resume_from_user_input not implemented")

    def __repr__(self) -> str:
        """Return string representation of the multi-agent system."""
        return f"{self.__class__.__name__}(agents={len(self.agents)})"
