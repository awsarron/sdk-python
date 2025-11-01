"""Agent Interface.

Defines the minimal interface that all agent types must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Type

from pydantic import BaseModel

from ..types.agent import AgentInput
from .agent_result import AgentResult
from .state import AgentState


class AgentBase(ABC):
    """Abstract interface for all agent types in Strands.

    This interface defines the minimal contract that all agent implementations
    must satisfy.
    """

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for the agent.

        Returns:
            Unique string identifier for this agent instance.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the agent.

        Returns:
            Display name for the agent.
        """
        pass

    @property
    @abstractmethod
    def state(self) -> AgentState:
        """Current state of the agent.

        Returns:
            AgentState object containing stateful information.
        """
        pass

    @abstractmethod
    async def invoke_async(
        self,
        prompt: AgentInput = None,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Asynchronously invoke the agent with the given prompt.

        Args:
            prompt: Input to the agent.
            invocation_state: Optional state to pass to the agent invocation.
            structured_output_model: Optional Pydantic model for structured output.
            **kwargs: Additional provider-specific arguments.

        Returns:
            AgentResult containing the agent's response.
        """
        pass

    @abstractmethod
    def __call__(
        self,
        prompt: AgentInput = None,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Synchronously invoke the agent with the given prompt.

        Args:
            prompt: Input to the agent.
            invocation_state: Optional state to pass to the agent invocation.
            structured_output_model: Optional Pydantic model for structured output.
            **kwargs: Additional provider-specific arguments.

        Returns:
            AgentResult containing the agent's response.
        """
        pass

    @abstractmethod
    def stream_async(
        self,
        prompt: AgentInput = None,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream agent execution asynchronously.

        Args:
            prompt: Input to the agent.
            invocation_state: Optional state to pass to the agent invocation.
            structured_output_model: Optional Pydantic model for structured output.
            **kwargs: Additional provider-specific arguments.

        Yields:
            Events representing the streaming execution.
        """
        pass
