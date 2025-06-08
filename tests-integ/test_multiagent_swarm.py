"""
Integration test for Swarm multi-agent pattern.
"""

import asyncio
import logging
import uuid

import pytest

from strands import Agent, tool
from strands.multi_agent.swarm import MessageType, Swarm, SwarmExecutionConfig, SwarmMessage, SwarmState, SwarmStatus

logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()])


# Mock tools for test agents
@tool
def research_web(query: str) -> dict:
    """Research information on the web."""
    return {
        "status": "success",
        "content": [
            {
                "text": (
                    "🔍 Research results for 'hello world': A basic phrase used in the context of programming. "
                    "Hello world is a simple program that displays the text hello world on the screen. "
                    "It serves as a basic test to ensure that a programming environment is set up correctly "
                    "and that a new language is understood."
                )
            },
            {"text": "Data: 99.5% usage, 100% adoption, hundreds of languages"},
            {"text": "Recent trends: +400% usage YoY, learning resources increased 42%"},
        ],
    }


@pytest.fixture
def test_agents():
    """Create test agents for integration testing."""
    session_id = uuid.uuid4()

    researcher = Agent(
        name="researcher",
        tools=[research_web],
        system_prompt=(
            "You are a research specialist. Research the topic and hand off to analyst "
            "when you have enough information."
        ),
        trace_attributes={"session.id": session_id},
    )

    analyst = Agent(
        name="analyst",
        system_prompt="You are a data analyst. Analyze the research data and hand off to writer for report creation.",
        trace_attributes={"session.id": session_id},
    )

    writer = Agent(
        name="writer",
        system_prompt="You are a report writer. Create reports based on analysis and complete the task.",
        trace_attributes={"session.id": session_id},
    )

    return [researcher, analyst, writer]


@pytest.fixture
def swarm_config():
    """Create swarm configuration."""
    return SwarmExecutionConfig(max_handoffs=10, max_iterations=20, execution_timeout=120.0, agent_timeout=60.0)


@pytest.fixture
def swarm(test_agents, swarm_config):
    """Create swarm with test agents and configuration."""
    return Swarm(
        agents=test_agents,
        swarm_config=swarm_config,
    )


@pytest.mark.asyncio
async def test_swarm_execute(swarm):
    """Test swarm execution."""
    task = "Analyze recent data trends for 'hello world' and provide a short summary"

    results = await asyncio.wait_for(
        swarm.execute(task),
        timeout=130,
    )

    # Basic validation - should have some results
    assert results is not None
    assert isinstance(results, list)

    # Check swarm completed
    assert swarm.swarm_state is not None
    assert swarm.swarm_state.completion_status is SwarmStatus.COMPLETED


@pytest.mark.asyncio
async def test_swarm_resume_from_user_input(swarm):
    """Test swarm resume from user input with simulated user handoff state."""

    # Set up a realistic swarm state as if execution was paused for user input
    # Create a simple, completable task
    task = "Create a greeting message for the user with their name"

    # Modify the writer agent to be more likely to complete the task
    writer_agent = swarm.get_agent("writer")
    if writer_agent:
        writer_agent.system_prompt = (
            "You are a report writer. When you receive a user's name, immediately create a simple "
            "greeting message and complete the task using complete_swarm_task. Do not ask additional questions."
        )

    # Initialize swarm state (similar to what execute() would do)
    swarm.swarm_state = SwarmState(
        current_agent="writer",
        task=task,
        swarm_ref=swarm,
        max_handoffs=swarm.swarm_config.max_handoffs,
        max_iterations=swarm.swarm_config.max_iterations,
        completion_status=SwarmStatus.WAITING_FOR_USER,
        agent_history=["researcher", "analyst"],
    )
    swarm.swarm_state._swarm_ref = swarm
    swarm.shared_context.set_task(task)
    swarm.shared_context.agent_history = ["researcher", "analyst"]
    swarm.shared_context.add_fact("analyst", "task_analysis", "User needs a personalized greeting message")
    swarm.shared_context.add_fact("researcher", "requirements", "Simple greeting with user's name")

    # Simulate pending user input (as if handoff_to_user tool was called)
    dummy_question = "What is your name for the greeting?"
    pending_user_input = {
        "question": dummy_question,
        "context": {"greeting_context": "personalization"},
        "options": None,
        "requesting_agent": "writer",
        "timestamp": 1234567890.0,
    }
    swarm.shared_context.add_artifact("pending_user_input", pending_user_input)

    # Add handoff message to history
    handoff_msg = SwarmMessage(
        from_agent="writer",
        to_agent="user",
        content=dummy_question,
        context={"greeting_context": "personalization"},
        message_type=MessageType.USER_HANDOFF,
    )
    swarm.swarm_state.message_history.append(handoff_msg)

    # Validate initial state
    assert swarm.swarm_state.completion_status == SwarmStatus.WAITING_FOR_USER

    user_question = swarm.get_user_question()
    assert user_question is not None
    assert user_question["question"] == dummy_question
    assert user_question["requesting_agent"] == "writer"

    # Resume with user input - provide a clear, simple answer
    user_response = "Alice"
    results = await asyncio.wait_for(
        swarm.resume_from_user_input(user_response),
        timeout=130,
    )

    # Validate resume completed successfully
    assert results is not None
    assert isinstance(results, list)
    assert len(results) > 0  # Should have at least one result from resumed execution

    # Check swarm completed successfully
    assert swarm.swarm_state is not None
    assert swarm.swarm_state.completion_status == SwarmStatus.COMPLETED

    # Validate user response was stored in context
    user_facts = swarm.shared_context.facts.get("user", {})
    assert "response" in user_facts
    assert user_facts["response"] == user_response
    assert "original_question" in user_facts
    assert user_facts["original_question"] == dummy_question

    # Should have user response message in history
    user_messages = [msg for msg in swarm.swarm_state.message_history if msg.from_agent == "user"]
    assert len(user_messages) > 0
    assert user_messages[0].content == user_response
    assert user_messages[0].to_agent == "writer"

    # Validate that execution progressed (iteration count should have increased)
    assert swarm.swarm_state.iteration_count > 0

    # Should have a final result
    assert swarm.swarm_state.final_result is not None
