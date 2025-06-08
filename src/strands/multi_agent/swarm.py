"""Swarm Multi-Agent Pattern for Strands SDK.

Tool-based coordination system that works universally across all model providers.
Agents coordinate through tools rather than prompt engineering.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from strands import tool

from ..agent import Agent
from .base import AgentResult, MultiAgentBase

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in swarm communication."""

    HANDOFF = "handoff"
    RESPONSE = "response"
    COMPLETION = "completion"
    USER_HANDOFF = "user_handoff"


class SwarmStatus(Enum):
    """Status of swarm execution."""

    NOT_STARTED = "not_started"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_FOR_USER = "waiting_for_user"


@dataclass
class SwarmMessage:
    """Message passed between agents in swarm."""

    # TODO: Use Strands Message type that is multi-modal

    from_agent: str
    to_agent: str
    content: str
    context: Dict[str, Any]
    message_type: MessageType
    timestamp: float = field(default_factory=time.time)


@dataclass
class SwarmExecutionConfig:
    """Configuration for swarm execution safety."""

    max_handoffs: int = 10
    max_iterations: int = 20
    execution_timeout: float = 900.0  # Total execution timeout (seconds)
    agent_timeout: float = 900.0  # Individual agent timeout (seconds)
    ping_pong_check_agents: int = 8  # Number of recent agents to check for ping-pong
    ping_pong_min_unique_agents: int = 3  # Minimum unique agents required in recent sequence


class SharedContext:
    """Shared context accessible via tools."""

    def __init__(self) -> None:
        """Initialize shared context for swarm agents."""
        self.facts: Dict[str, Dict[str, Any]] = {}
        self.artifacts: Dict[str, Any] = {}
        self.agent_history: List[str] = []
        self.current_task: Optional[str] = None
        self.available_agents: List[str] = []

    def set_task(self, task: str) -> None:
        """Set the current task."""
        self.current_task = task

    def set_available_agents(self, agent_names: List[str]) -> None:
        """Set list of available agents."""
        self.available_agents = agent_names

    def add_fact(self, agent_name: str, key: str, value: Any) -> None:
        """Tool-accessible method to add facts."""
        if agent_name not in self.facts:
            self.facts[agent_name] = {}
        self.facts[agent_name][key] = value

    def add_artifact(self, name: str, artifact: Any) -> None:
        """Add an artifact (file, data, etc.)."""
        self.artifacts[name] = artifact

    def get_relevant_context(self, agent_name: str) -> Dict[str, Any]:  # TODO: better types
        """Get context relevant to specific agent."""
        return {
            "task": self.current_task,
            "agent_history": self.agent_history,
            "shared_facts": {k: v for k, v in self.facts.items() if v},
            "shared_artifacts": list(self.artifacts.keys()),
            "available_agents": [name for name in self.available_agents if name != agent_name],
        }


class Swarm(MultiAgentBase):
    """Tool-based swarm coordination that works with any model provider."""

    def __init__(
        self,
        agents: List[Agent],
        swarm_config: Optional[SwarmExecutionConfig] = None,
    ):
        """Initialize swarm with agents and configuration.

        Args:
            agents: List of agents to include in the swarm
            swarm_config: Optional swarm execution configuration
        """
        super().__init__(agents)

        self.swarm_config = swarm_config or SwarmExecutionConfig()
        self._request_state: Dict[str, Any] = {}  # TODO: better types
        self.shared_context = SharedContext()
        # Initialize with a dummy state that will be replaced in execute()
        # This avoids Optional typing issues while maintaining clean initialization
        self.swarm_state: SwarmState = SwarmState(
            current_agent="",
            task="",
            swarm_ref=self,
            completion_status=SwarmStatus.NOT_STARTED,
        )

        self._setup_swarm()
        self._inject_swarm_tools()

    def _setup_swarm(self) -> None:
        """Initialize swarm configuration."""
        # After validation in base class, all agents are guaranteed to have names
        agent_names = [str(agent.name) for agent in self.agents]

        self.shared_context.set_available_agents(agent_names)
        logger.info("agents=<%s> | initialized swarm with agents", agent_names)

    def _inject_swarm_tools(self) -> None:
        """Add swarm coordination tools to each agent."""
        # Create tool functions with proper closures
        swarm_tools = [
            self._create_handoff_tool(),
            self._create_complete_tool(),
            self._create_context_tool(),
            self._create_handoff_to_user_tool(),
        ]

        for agent in self.agents:
            # Use the agent's tool registry to process and register the tools
            agent.tool_registry.process_tools(swarm_tools)

        logger.info(
            "tool_count=<%d>, agent_count=<%d> | injected coordination tools into agents",
            len(swarm_tools),
            len(self.agents),
        )

    def _create_handoff_tool(self) -> Callable[..., Any]:  # TODO: better types
        """Create handoff tool using @tool decorator."""
        swarm_ref = self  # Capture swarm reference

        # TODO: better types
        @tool
        def handoff_to_agent(agent_name: str, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Transfer control to another agent in the swarm for specialized help.

            Args:
                agent_name: Name of the agent to hand off to
                message: Message explaining what needs to be done and why you're handing off
                context: Additional context to share with the next agent

            Returns:
                Confirmation of handoff initiation
            """
            try:
                context = context or {}

                # Validate target agent exists
                if not swarm_ref.get_agent(agent_name):
                    return {"status": "error", "content": [{"text": f"Error: Agent '{agent_name}' not found in swarm"}]}

                # Execute handoff
                swarm_ref._handle_handoff(agent_name, message, context)

                # TODO: Force stop agent event loop

                return {"status": "success", "content": [{"text": f"Handed off to {agent_name}: {message}"}]}

            except Exception as e:
                return {"status": "error", "content": [{"text": f"Error in handoff: {str(e)}"}]}

        return handoff_to_agent

    def _create_complete_tool(self) -> Callable[..., Any]:  # TODO: better types
        """Create completion tool using @tool decorator."""
        swarm_ref = self  # Capture swarm reference

        @tool
        def complete_swarm_task(result: str, summary: Optional[str] = None) -> Dict[str, Any]:  # TODO: better types
            """Mark the task as complete with final result. No more agents will be called.

            Args:
                result: The final result/answer
                summary: Optional summary of how the task was completed

            Returns:
                Task completion confirmation
            """
            try:
                # Mark swarm as complete
                swarm_ref._handle_completion(result, summary or "")

                # TODO: Force stop agent event loop

                return {"status": "success", "content": [{"text": f"Task completed: {result}"}]}

            except Exception as e:
                return {"status": "error", "content": [{"text": f"Error completing task: {str(e)}"}]}

        return complete_swarm_task

    def _create_context_tool(self) -> Callable[..., Any]:  # TODO: better types
        """Create context tool using @tool decorator."""
        swarm_ref = self  # Capture swarm reference

        @tool
        def get_swarm_context() -> Dict[str, Any]:  # TODO: better types
            """Get the current shared context and agent history.

            Returns:
                Current swarm state including shared facts, agent history, etc.
            """
            try:
                # Get context for current agent
                current_agent = swarm_ref.swarm_state.current_agent
                context = swarm_ref.shared_context.get_relevant_context(current_agent)

                context_text = swarm_ref._format_context(context)

                return {"status": "success", "content": [{"text": context_text}]}

            except Exception as e:
                return {"status": "error", "content": [{"text": f"Error getting context: {str(e)}"}]}

        return get_swarm_context

    def _create_handoff_to_user_tool(self) -> Callable[..., Any]:  # TODO: better types
        """Create handoff to user tool that stops the event loop."""
        swarm_ref = self  # Capture swarm reference

        # TODO: better types
        @tool
        def handoff_to_user(
            question: str, context: Optional[Dict[str, Any]] = None, options: Optional[List[Any]] = None
        ) -> Dict[str, Any]:
            """Hand off control back to the human user for input or decision.

            This tool pauses the current task and saves the user question for the orchestrator to handle.
            After the user has given input, the task may resume.

            Args:
                question: The question or request to present to the user
                context: Additional context to preserve for when user responds
                options: Optional list of choices to present to user

            Returns:
                Handoff confirmation with user input request
            """
            try:
                context = context or {}

                # Execute user handoff (saves state)
                swarm_ref._handle_user_handoff(question, context, options)

                # Stop the event loop to prevent further agent execution
                swarm_ref._request_state["stop_event_loop"] = True

                return {
                    "status": "success",
                    "content": [{"text": f"Question for user: {question}\nPausing execution for user response."}],
                }

            except Exception as e:
                return {"status": "error", "content": [{"text": f"Error in user handoff: {str(e)}"}]}

        return handoff_to_user

    def _select_initial_agent(self, task: str) -> str:
        """Choose which agent should start working on the task."""
        # Simple heuristic: use first agent
        # TODO: Could be enhanced with an LLM invocation to select one
        if self.agents:
            return str(self.agents[0].name)  # name is guaranteed to exist after validation
        raise ValueError("No agents available in swarm")

    # TODO: better types
    def _handle_handoff(self, target_agent: str, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle handoff to another agent."""
        # If task is already completed, don't allow further handoffs
        if self.swarm_state.completion_status != SwarmStatus.ACTIVE:
            logger.info(
                "task_status=<%s> | ignoring handoff request - task already completed",
                self.swarm_state.completion_status.value,
            )
            return {"status": "ignored", "reason": f"task_already_{self.swarm_state.completion_status.value}"}

        # Update swarm state
        previous_agent = self.swarm_state.current_agent
        self.swarm_state.current_agent = target_agent

        # Add handoff message
        handoff_msg = SwarmMessage(
            from_agent=previous_agent,
            to_agent=target_agent,
            content=message,
            context=context,
            message_type=MessageType.HANDOFF,
        )
        self.swarm_state.message_history.append(handoff_msg)

        # Store handoff context as shared facts
        if context:
            for key, value in context.items():
                self.shared_context.add_fact(previous_agent, key, value)

        logger.info(
            "from_agent=<%s>, to_agent=<%s>, message=<%s> | handed off from agent to agent",
            previous_agent,
            target_agent,
            message,
        )
        return {"status": "success", "target_agent": target_agent}

    def _handle_completion(self, result: str, summary: str = "") -> None:
        """Handle task completion."""
        self.swarm_state.completion_status = SwarmStatus.COMPLETED
        self.swarm_state.final_result = result

        # Add completion message
        completion_msg = SwarmMessage(
            from_agent=self.swarm_state.current_agent,
            to_agent="system",
            content=result,
            context={"summary": summary},
            message_type=MessageType.COMPLETION,
        )
        self.swarm_state.message_history.append(completion_msg)

        logger.info("result=<%s> | swarm task completed", result)

    # TODO: better types
    def _handle_user_handoff(
        self, question: str, context: Dict[str, Any], options: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Handle handoff to user and stop execution."""
        # Update swarm state to waiting for user
        self.swarm_state.completion_status = SwarmStatus.WAITING_FOR_USER

        # Store user handoff info
        user_handoff_data = {
            "question": question,
            "context": context,
            "options": options,
            "requesting_agent": self.swarm_state.current_agent,
            "timestamp": time.time(),
        }

        # Add to shared context for persistence
        self.shared_context.add_artifact("pending_user_input", user_handoff_data)

        # Add user handoff message to history
        user_msg = SwarmMessage(
            from_agent=self.swarm_state.current_agent,
            to_agent="user",
            content=question,
            context=context,
            message_type=MessageType.USER_HANDOFF,
        )
        self.swarm_state.message_history.append(user_msg)

        logger.info("agent=<%s>, question=<%s> | user handoff from agent", self.swarm_state.current_agent, question)
        return {"status": "waiting_for_user", "question": question, "options": options}

    async def _continue_execution(self) -> List[AgentResult]:
        """Shared execution logic used by both execute() and resume_from_user_input()."""
        results = []

        try:
            # Main execution loop
            while True:
                should_continue, reason = self.swarm_state.should_continue(self.swarm_config)
                if not should_continue:
                    logger.info("reason=<%s> | stopping execution", reason)
                    break

                self.swarm_state.increment_iteration(self.swarm_state.current_agent)

                # Get current agent
                current_agent = self.get_agent(self.swarm_state.current_agent)
                if not current_agent:
                    logger.error("agent=<%s> | agent not found", self.swarm_state.current_agent)
                    self.swarm_state.completion_status = SwarmStatus.FAILED
                    break

                logger.info(
                    "current_agent=<%s>, iteration=<%d> | executing agent",
                    self.swarm_state.current_agent,
                    self.swarm_state.iteration_count,
                )

                # Execute agent with timeout protection
                try:
                    result = await asyncio.wait_for(
                        self._execute_agent_with_swarm_tools(current_agent, self.swarm_state.task),
                        timeout=self.swarm_config.agent_timeout,
                    )
                    results.append(result)

                    self.swarm_state.agent_history.append(str(current_agent.name))
                    self.shared_context.agent_history.append(str(current_agent.name))

                    logger.info("status=<%s> | agent execution completed", result.status)

                    # Immediate check for completion after agent execution
                    if self.swarm_state.completion_status != SwarmStatus.ACTIVE:
                        logger.info(
                            "status=<%s> | task completed with status", self.swarm_state.completion_status.value
                        )
                        break

                except asyncio.TimeoutError:
                    logger.error(
                        "agent=<%s>, timeout=<%s>s | agent execution timed out after timeout",
                        self.swarm_state.current_agent,
                        self.swarm_config.agent_timeout,
                    )
                    self.swarm_state.completion_status = SwarmStatus.FAILED
                    break

                except Exception as e:
                    logger.error(
                        "agent=<%s>, error=<%s> | agent execution failed", self.swarm_state.current_agent, str(e)
                    )
                    self.swarm_state.completion_status = SwarmStatus.FAILED
                    break

        except Exception as e:
            logger.error("error=<%s> | swarm execution failed", str(e))
            self.swarm_state.completion_status = SwarmStatus.FAILED

        elapsed_time = time.time() - self.swarm_state.start_time
        logger.info("status=<%s> | swarm execution completed", self.swarm_state.completion_status.value)
        logger.info(
            "iterations=<%d>, handoffs=<%d>, time=<%s>s | metrics",
            self.swarm_state.iteration_count,
            len(self.swarm_state.agent_history),
            f"{elapsed_time:.2f}",
        )

        return results

    async def _execute_agent_with_swarm_tools(self, agent: Agent, task: str) -> AgentResult:
        """Execute agent with swarm tools available."""
        start_time = time.time()
        agent_name = getattr(agent, "name", "unknown")
        task_id = f"swarm_{uuid.uuid4().hex[:8]}"

        try:
            # Prepare context for agent
            context_info = self.shared_context.get_relevant_context(agent_name)

            # Create task message with context
            task_with_context = f"Task: {task}\n\n"
            task_with_context += self._format_context(context_info)

            # Agent uses normal Strands execution - tools handle coordination
            result = agent(task_with_context)
            execution_time = time.time() - start_time

            return AgentResult(
                agent_name=agent_name, task_id=task_id, status="success", result=result, execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                agent_name=agent_name,
                task_id=task_id,
                status="error",
                result=None,
                execution_time=execution_time,
                error=str(e),
            )

    def _format_context(self, context_info: Dict[str, Any]) -> str:  # TODO: better types
        """Format task message with relevant context."""
        context_text = ""

        # Include detailed agent history
        if context_info.get("agent_history"):
            context_text += f"Previous agents who worked on this: {' → '.join(context_info['agent_history'])}\n\n"

        # Include actual shared facts, not just a mention
        shared_facts = context_info.get("shared_facts", {})
        if shared_facts:
            context_text += "🧠 SHARED KNOWLEDGE FROM PREVIOUS AGENTS:\n"
            for agent_name, facts in shared_facts.items():
                if facts:  # Only include if agent has contributed facts
                    context_text += f"• {agent_name}: {facts}\n"
            context_text += "\n"

        # Include shared artifacts
        artifacts = context_info.get("shared_artifacts", [])
        if artifacts:
            context_text += f"📁 Shared artifacts available: {', '.join(artifacts)}\n\n"

        # Include available agents
        if context_info.get("available_agents"):
            context_text += (
                f"Other agents available for collaboration: {', '.join(context_info['available_agents'])}\n\n"
            )

        context_text += (
            "You have access to swarm coordination tools if you need help from other agents "
            "or want to complete the task."
        )

        return context_text

    def get_swarm_summary(self) -> Dict[str, Any]:
        """Get summary of swarm execution with enhanced metrics."""
        if self.swarm_state.completion_status == SwarmStatus.NOT_STARTED:
            return {"status": "not_started"}

        elapsed_time = time.time() - self.swarm_state.start_time

        return {
            "status": self.swarm_state.completion_status.value,
            "current_agent": self.swarm_state.current_agent,
            "agent_history": self.swarm_state.agent_history,
            "handoff_count": len(self.swarm_state.agent_history) - 1,
            "iteration_count": self.swarm_state.iteration_count,
            "elapsed_time": elapsed_time,
            "final_result": self.swarm_state.final_result,
            "message_count": len(self.swarm_state.message_history),
            "safety_metrics": {
                "max_handoffs": self.swarm_config.max_handoffs,
                "max_iterations": self.swarm_config.max_iterations,
                "execution_timeout": self.swarm_config.execution_timeout,
                "agent_timeout": self.swarm_config.agent_timeout,
                "agent_sequence": self.swarm_state.last_agent_sequence.copy(),
            },
            "messages": [
                {
                    "from": msg.from_agent,
                    "to": msg.to_agent,
                    "type": msg.message_type.value,
                    "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                }
                for msg in self.swarm_state.message_history
            ],
        }

    def get_user_question(self) -> Optional[Dict]:
        """Get the pending user question if any."""
        if self.swarm_state.completion_status != SwarmStatus.WAITING_FOR_USER:
            return None

        return self.shared_context.artifacts.get("pending_user_input")

    async def execute(self, task: str) -> List[AgentResult]:
        """Execute swarm task using tool-based coordination with comprehensive safety mechanisms."""
        # Initialize swarm state with safety configuration
        initial_agent = self._select_initial_agent(task)
        self.swarm_state = SwarmState(
            current_agent=initial_agent,
            task=task,
            max_handoffs=self.swarm_config.max_handoffs,
            max_iterations=self.swarm_config.max_iterations,
            swarm_ref=self,
            completion_status=SwarmStatus.ACTIVE,  # Set to ACTIVE when execution starts
        )
        self.shared_context.set_task(task)

        logger.info("current_agent=<%s> | starting safe swarm execution with agent", self.swarm_state.current_agent)
        logger.info(
            "max_handoffs=<%d>, max_iterations=<%d>, timeout=<%s>s | safety config",
            self.swarm_config.max_handoffs,
            self.swarm_config.max_iterations,
            self.swarm_config.execution_timeout,
        )

        # Delegate to the shared execution logic
        return await self._continue_execution()

    async def resume_from_user_input(self, user_response: str) -> List[AgentResult]:
        """Resume swarm execution after user provides input."""
        if self.swarm_state.completion_status != SwarmStatus.WAITING_FOR_USER:
            raise ValueError("Swarm is not waiting for user input")

        # TODO: make it simpler to pass in SwarmState and SharedContext for resuming

        # Get the pending user input data
        pending_input = self.shared_context.artifacts.get("pending_user_input")
        if not pending_input:
            raise ValueError("No pending user input found")

        # Store user response in context
        self.shared_context.add_fact("user", "original_question", pending_input["question"])
        self.shared_context.add_fact("user", "response", user_response)

        # Resume with the agent that requested user input
        self.swarm_state.current_agent = pending_input["requesting_agent"]
        self.swarm_state.completion_status = SwarmStatus.ACTIVE

        # Reset stop flag to allow execution to continue
        self._request_state["stop_event_loop"] = False

        # Create user response message
        user_response_msg = SwarmMessage(
            from_agent="user",
            to_agent=pending_input["requesting_agent"],
            content=user_response,
            context={"original_question": pending_input["question"]},
            message_type=MessageType.RESPONSE,
        )
        self.swarm_state.message_history.append(user_response_msg)

        # Clean up pending input
        if "pending_user_input" in self.shared_context.artifacts:
            del self.shared_context.artifacts["pending_user_input"]

        logger.info("user_response=<%s> | resuming swarm with user response", user_response)

        # Create enhanced task including user response
        enhanced_task = (
            f"User responded: '{user_response}' to the question: '{pending_input['question']}'. Continue with the task."
        )

        # Update the task in shared context for continued execution
        original_task = self.swarm_state.task
        self.swarm_state.task = enhanced_task
        self.shared_context.set_task(enhanced_task)

        try:
            # Delegate to execute method to handle the actual execution logic
            # This eliminates code duplication and ensures consistent behavior
            return await self._continue_execution()
        finally:
            # Restore original task
            self.swarm_state.task = original_task
            self.shared_context.set_task(original_task)


@dataclass
class SwarmState:
    """Current state of swarm execution with safety mechanisms."""

    current_agent: str
    task: str
    swarm_ref: Swarm
    shared_context: Dict[str, Any] = field(default_factory=dict)
    agent_history: List[str] = field(default_factory=list)
    message_history: List[SwarmMessage] = field(default_factory=list)
    max_handoffs: int = 10
    max_iterations: int = 20
    completion_status: SwarmStatus = SwarmStatus.NOT_STARTED
    final_result: Any = None
    iteration_count: int = 0
    start_time: float = field(default_factory=time.time)
    last_agent_sequence: List[str] = field(default_factory=list)

    def should_continue(self, config: SwarmExecutionConfig) -> Tuple[bool, str]:
        """Comprehensive check for continuation with detailed reason.

        Returns: (should_continue, reason)
        """
        elapsed = time.time() - self.start_time

        # 1. Check completion status
        if self.completion_status != SwarmStatus.ACTIVE:
            return False, f"completion_status_changed_to_{self.completion_status.value}"

        # 2. Check if event loop should stop (user handoff or other reasons)
        if hasattr(self, "swarm_ref") and self.swarm_ref and self.swarm_ref._request_state.get("stop_event_loop"):
            return False, "stop_event_loop_requested"

        # 3. Check handoff limit
        if len(self.agent_history) >= self.max_handoffs:
            self.completion_status = SwarmStatus.FAILED
            return False, f"max_handoffs_reached_{self.max_handoffs}"

        # 4. Check iteration limit
        if self.iteration_count >= self.max_iterations:
            self.completion_status = SwarmStatus.FAILED
            return False, f"max_iterations_reached_{self.max_iterations}"

        # 5. Check timeout
        if elapsed > config.execution_timeout:
            self.completion_status = SwarmStatus.FAILED
            return False, f"execution_timeout_{config.execution_timeout}s"

        # 6. Check for agent ping-pong (agents passing back and forth)
        if len(self.last_agent_sequence) >= config.ping_pong_check_agents:
            recent = self.last_agent_sequence[-config.ping_pong_check_agents :]
            unique_agents = len(set(recent))
            if unique_agents < config.ping_pong_min_unique_agents:
                self.completion_status = SwarmStatus.FAILED
                return (
                    False,
                    f"agent_ping_pong_detected_{unique_agents}_unique_in_{config.ping_pong_check_agents}_recent",
                )

        return True, "continuing"

    def increment_iteration(self, current_agent: str) -> None:
        """Safely increment iteration and track agent usage."""
        self.iteration_count += 1
        self.last_agent_sequence.append(current_agent)

        # Keep only the required number of agents for ping-pong detection plus some buffer
        max_sequence_length = self.swarm_ref.swarm_config.ping_pong_check_agents + 2

        if len(self.last_agent_sequence) > max_sequence_length:
            self.last_agent_sequence = self.last_agent_sequence[-max_sequence_length:]
