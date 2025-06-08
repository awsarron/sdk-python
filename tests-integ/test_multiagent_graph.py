"""
Integration test for AgentGraph multi-agent pattern.
"""

import asyncio
import logging
import uuid

import pytest

from strands import Agent, tool
from strands.multi_agent.graph import GraphBuilder, GraphStatus
from strands.multi_agent.swarm import Swarm

logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()])


@tool
def analyze_request(request: str) -> dict:
    """Analyze a request and determine its approval status."""
    # Simulate different approval statuses based on request content
    if "critical" in request.lower() or "emergency" in request.lower():
        status = "REJECTED"
        reason = "Contains critical issues requiring immediate attention"
    elif "complex" in request.lower() or "review" in request.lower():
        status = "REVIEW_NEEDED"
        reason = "Requires additional review before approval"
    else:
        status = "APPROVED"
        reason = "The request meets all standard requirements"

    return {
        "status": "success",
        "content": [
            {
                "text": (
                    f"📋 Request Analysis Complete:\n"
                    f"- Status: {status}\n"
                    f"- Reason: {reason}\n"
                    f"- Confidence: 92%\n"
                    f"- Request: '{request}'"
                )
            }
        ],
    }


@tool
def standard_processing(request: str) -> dict:
    """Process a standard approved request."""
    return {
        "status": "success",
        "content": [
            {
                "text": (
                    "✅ Standard Processing Complete:\n"
                    "- Automated approval workflow executed\n"
                    "- All compliance checks passed\n"
                    "- Processing time: 5 minutes\n"
                    "- Status: COMPLETED\n"
                    "- Ready for deployment"
                )
            }
        ],
    }


@tool
def manual_review_processing(request: str) -> dict:
    """Process a request that needs manual review."""
    return {
        "status": "success",
        "content": [
            {
                "text": (
                    "🔍 Manual Review Initiated:\n"
                    "- Escalated to human reviewers\n"
                    "- Additional documentation requested\n"
                    "- Expected timeline: 2-3 business days\n"
                    "- Status: PENDING_REVIEW\n"
                    "- Stakeholders notified"
                )
            }
        ],
    }


@tool
def emergency_handling(request: str) -> dict:
    """Handle emergency/critical requests."""
    return {
        "status": "success",
        "content": [
            {
                "text": (
                    "🚨 Emergency Protocol Activated:\n"
                    "- Emergency response team notified\n"
                    "- Escalated to senior management\n"
                    "- 24/7 monitoring activated\n"
                    "- Status: CRITICAL_ATTENTION_REQUIRED\n"
                    "- Immediate action required"
                )
            }
        ],
    }


@tool
def final_reporting(results: str) -> dict:
    """Generate final report from processing results."""
    return {
        "status": "success",
        "content": [
            {
                "text": (
                    "📊 Final Processing Report:\n"
                    "- All routing decisions executed successfully\n"
                    "- Appropriate handlers engaged\n"
                    "- Status tracking completed\n"
                    "- Report timestamp: 2024-12-08 20:00:00\n"
                    "- Process completed"
                )
            }
        ],
    }


@pytest.fixture
def test_agents():
    """Create test agents for conditional edge testing."""
    session_id = uuid.uuid4()

    analyzer = Agent(
        name="analyzer",
        tools=[analyze_request],
        system_prompt=(
            "You are a request analyzer. ALWAYS use the analyze_request tool to analyze every request you receive. "
            "Use the analyze_request tool with the request text as the parameter to determine the approval status: "
            "APPROVED, REVIEW_NEEDED, or REJECTED."
            "\n\n"
            "Once analyze_request has been used, repeat the approval status."
        ),
        trace_attributes={"session.id": session_id},
    )

    standard_processor = Agent(
        name="standard_processor",
        tools=[standard_processing],
        system_prompt=(
            "You are a standard processor. ALWAYS use the standard_processing tool to handle approved "
            "requests through automated workflows."
        ),
        trace_attributes={"session.id": session_id},
    )

    manual_reviewer = Agent(
        name="manual_reviewer",
        tools=[manual_review_processing],
        system_prompt=(
            "You are a manual reviewer. ALWAYS use the manual_review_processing tool to handle requests "
            "that need human review and additional evaluation."
        ),
        trace_attributes={"session.id": session_id},
    )

    emergency_assessor = Agent(
        name="emergency_assessor",
        tools=[emergency_handling],
        system_prompt=(
            "You are an emergency assessor. ALWAYS use the emergency_handling tool to assess "
            "and classify the severity of emergency requests."
        ),
        trace_attributes={"session.id": session_id},
    )

    emergency_responder = Agent(
        name="emergency_responder",
        tools=[emergency_handling],
        system_prompt=(
            "You are an emergency responder. ALWAYS use the emergency_handling tool to execute "
            "the appropriate emergency response based on the assessment."
        ),
        trace_attributes={"session.id": session_id},
    )

    reporter = Agent(
        name="reporter",
        tools=[final_reporting],
        system_prompt=(
            "You are a final reporter. ALWAYS use the final_reporting tool to generate comprehensive "
            "reports from processing results."
        ),
        trace_attributes={"session.id": session_id},
    )

    swarm_coordinator = Agent(
        name="swarm_coordinator",
        tools=[analyze_request],
        system_prompt=(
            "You are a swarm coordinator. Analyze tasks and delegate to team members as needed. "
            "Use the analyze_request tool to assess incoming tasks and coordinate with your team."
        ),
        trace_attributes={"session.id": session_id},
    )

    swarm_specialist = Agent(
        name="swarm_specialist",
        tools=[standard_processing],
        system_prompt=(
            "You are a specialist in the swarm. Handle specialized processing tasks assigned by the coordinator. "
            "Use the standard_processing tool to complete assigned work."
        ),
        trace_attributes={"session.id": session_id},
    )

    return {
        "analyzer": analyzer,
        "standard": standard_processor,
        "manual": manual_reviewer,
        "emergency_assessor": emergency_assessor,
        "emergency_responder": emergency_responder,
        "reporter": reporter,
        "swarm_coordinator": swarm_coordinator,
        "swarm_specialist": swarm_specialist,
    }


# Conditional edge functions
def _is_approved(results):
    """Check if the analysis result shows APPROVED status."""
    if not results:
        return False

    latest_result = results[-1]
    result_text = latest_result.result

    return "APPROVED" in result_text


def _needs_review(results):
    """Check if the analysis result shows REVIEW_NEEDED status."""
    if not results:
        return False

    latest_result = results[-1]
    result_text = latest_result.result

    return "REVIEW_NEEDED" in result_text


def _is_critical(results):
    """Check if the analysis result shows REJECTED (critical) status."""
    if not results:
        return False

    latest_result = results[-1]
    result_text = latest_result.result

    return "REJECTED" in result_text


@pytest.fixture
def test_graph(test_agents):
    """Create a conditional routing graph with multiple processing paths."""
    builder = GraphBuilder()

    # Create nested AgentGraph for emergency handling
    emergency_graph_builder = GraphBuilder()

    # Build the nested emergency graph
    emergency_graph_builder.add_node(test_agents["emergency_assessor"], "assess")
    emergency_graph_builder.add_node(test_agents["emergency_responder"], "respond")
    emergency_graph_builder.add_edge("assess", "respond")
    emergency_graph_builder.set_entry_point("assess")

    emergency_graph = emergency_graph_builder.build()

    # Create a swarm with 2 agents for emergency processing
    swarm_agents = [test_agents["swarm_coordinator"], test_agents["swarm_specialist"]]
    emergency_swarm = Swarm(agents=swarm_agents)

    # Add all nodes to main graph
    builder.add_node(test_agents["analyzer"], "analyze")
    builder.add_node(test_agents["standard"], "standard")
    builder.add_node(test_agents["manual"], "manual")
    builder.add_node(emergency_graph, "emergency")  # AgentGraph as a node
    builder.add_node(emergency_swarm, "emergency_swarm")  # Swarm as a node
    builder.add_node(test_agents["reporter"], "report")

    # Add conditional edges from analyzer
    builder.add_edge("analyze", "standard", condition=_is_approved)
    builder.add_edge("analyze", "manual", condition=_needs_review)
    builder.add_edge("analyze", "emergency", condition=_is_critical)

    # Add edges from processing nodes to reporter
    builder.add_edge("standard", "report")
    builder.add_edge("manual", "report")
    # Emergency goes through swarm first, then to reporter
    builder.add_edge("emergency", "emergency_swarm")
    builder.add_edge("emergency_swarm", "report")

    # Set entry point
    builder.set_entry_point("analyze")

    return builder.build()


@pytest.mark.asyncio
async def test_graph_resume_from_user_input(test_graph):
    """Test graph resume functionality with a simulated running state."""

    # Manually set up a graph in running state to test resume functionality
    test_graph.state.status = GraphStatus.RUNNING
    test_graph.state.task = "Process user feedback data"

    # Test resume with user input
    user_response = "Please include customer demographics in the analysis"
    resume_results = await test_graph.resume_from_user_input(user_response)

    # Should return empty list since there are no MultiAgentBase nodes to resume
    # but the method should execute without error
    assert isinstance(resume_results, list)
    assert len(resume_results) == 0  # No MultiAgentBase nodes to resume

    # Graph should still be in running state since we simulated it
    assert test_graph.state.status == GraphStatus.RUNNING


@pytest.mark.asyncio
async def test_graph_execute_standard_path(test_graph):
    """Test standard approved request routing through analyze -> standard -> report."""

    task = "Process this standard routine maintenance request"

    results = await asyncio.wait_for(
        test_graph.execute(task),
        timeout=120,
    )

    # Should route: analyze -> standard -> report
    assert len(results) == 3
    assert test_graph.state.status == GraphStatus.COMPLETED
    assert test_graph.state.execution_order == ["analyze", "standard", "report"]

    # Verify the standard processor was used
    standard_result = next(r for r in results if r.agent_name == "standard_processor")
    assert standard_result is not None

    # Verify reporter was the final step
    reporter_result = next(r for r in results if r.agent_name == "reporter")
    assert reporter_result is not None


@pytest.mark.asyncio
async def test_graph_execute_manual_review_path(test_graph):
    """Test complex request routing through analyze -> manual -> report."""

    task = "Evaluate this complex proposal that may need additional review"

    results = await asyncio.wait_for(
        test_graph.execute(task),
        timeout=120,
    )

    # Should route: analyze -> manual -> report
    assert len(results) == 3
    assert test_graph.state.status == GraphStatus.COMPLETED
    assert test_graph.state.execution_order == ["analyze", "manual", "report"]

    # Verify the manual reviewer was used
    manual_result = next(r for r in results if r.agent_name == "manual_reviewer")
    assert manual_result is not None

    # Verify reporter was the final step
    reporter_result = next(r for r in results if r.agent_name == "reporter")
    assert reporter_result is not None


@pytest.mark.asyncio
async def test_graph_execute_emergency_path(test_graph):
    """Test critical emergency request routing through analyze -> emergency -> emergency_swarm -> report."""

    task = "Handle this critical security breach immediately"

    results = await asyncio.wait_for(
        test_graph.execute(task),
        timeout=120,
    )

    # Should route: analyze -> emergency (nested graph with assess->respond) -> emergency_swarm -> report
    assert len(results) >= 5  # analyze + assess + respond + swarm agents + report (minimum)
    assert test_graph.state.status == GraphStatus.COMPLETED
    assert test_graph.state.execution_order == ["analyze", "emergency", "emergency_swarm", "report"]

    # Verify the emergency graph was used (should have results from nested agents)
    emergency_results = [r for r in results if r.agent_name in ["emergency_assessor", "emergency_responder"]]
    assert len(emergency_results) >= 1  # At least one emergency sub-agent should have executed

    # Verify the swarm was used (should have results from swarm agents)
    swarm_results = [r for r in results if r.agent_name in ["swarm_coordinator", "swarm_specialist"]]
    assert len(swarm_results) >= 1  # At least one swarm agent should have executed

    # Verify reporter was the final step
    reporter_result = next(r for r in results if r.agent_name == "reporter")
    assert reporter_result is not None


@pytest.mark.asyncio
async def test_graph___str__(test_graph):
    """Test string representation of conditional edge graph."""

    dag_viz = str(test_graph)

    expected_dag_viz = """Agent Graph Structure:
Nodes (6):
  analyze (Agent)
  standard (Agent)
  manual (Agent)
  emergency (AgentGraph)
    └─ Agent Graph Structure:
    └─ Nodes (2):
    └─   assess (Agent)
    └─   respond (Agent)
    └─ Entry Points: ['assess']
    └─ Edges:
    └─   assess -> respond
  emergency_swarm (Swarm)
    └─ swarm_coordinator (Agent)
    └─ swarm_specialist (Agent)
  report (Agent)
Entry Points: ['analyze']
Edges:
  analyze -> emergency [conditional]
  analyze -> manual [conditional]
  analyze -> standard [conditional]
  emergency -> emergency_swarm
  emergency_swarm -> report
  manual -> report
  standard -> report"""

    assert dag_viz == expected_dag_viz


@pytest.mark.asyncio
async def test_graph_execution_summary(test_graph):
    """Test execution summary functionality."""

    # Execute the graph to generate summary data
    await test_graph.execute("Test request for summary")

    # Test execution summary structure
    summary = test_graph.get_execution_summary()
    assert summary["total_nodes"] == 6  # analyze + standard + manual + emergency + emergency_swarm + report
    assert len(summary["entry_points"]) == 1
    assert summary["entry_points"][0] == "analyze"

    # Should have executed only one path
    # Note: With nested AgentGraph, the completed_nodes count represents the main graph nodes
    assert summary["completed_nodes"] == 3  # analyze + standard + report (for standard approval path)
    assert summary["failed_nodes"] == 0

    # Verify edge information includes conditional edges
    edges = summary["edges"]
    # Should have 7 edges total (3 conditional from analyze + 4 regular to/from report and swarm)
    assert len(edges) == 7

    # Verify all conditional edges are present
    conditional_edges = [("analyze", "standard"), ("analyze", "manual"), ("analyze", "emergency")]

    for edge in conditional_edges:
        assert edge in edges

    # Verify regular edges
    report_edges = [
        ("standard", "report"),
        ("manual", "report"),
        ("emergency", "emergency_swarm"),
        ("emergency_swarm", "report"),
    ]

    for edge in report_edges:
        assert edge in edges


@pytest.mark.asyncio
async def test_graph_reset_state(test_graph):
    """Test reset_state functionality to clear execution history and status."""

    print("\n🧪 Testing reset_state functionality...")

    # Manually set graph state to simulate an executed state (no actual execution)
    test_graph.state.status = GraphStatus.COMPLETED
    test_graph.state.task = "Mock executed task"
    test_graph.state.execution_order = ["analyze", "standard", "report"]

    # Get pre-reset execution summary for comparison
    pre_reset_summary = test_graph.get_execution_summary()

    # Now reset the graph state
    test_graph.reset_state()

    # Verify state has been reset to initial values
    assert test_graph.state.status == GraphStatus.PENDING
    assert test_graph.state.execution_order == []
    assert test_graph.state.task == ""

    # Verify reset execution summary shows no completed nodes
    post_reset_summary = test_graph.get_execution_summary()
    assert post_reset_summary["completed_nodes"] == 0
    assert post_reset_summary["failed_nodes"] == 0

    # Verify total nodes count remains the same (graph structure unchanged)
    assert post_reset_summary["total_nodes"] == pre_reset_summary["total_nodes"]
    assert post_reset_summary["entry_points"] == pre_reset_summary["entry_points"]
    assert post_reset_summary["edges"] == pre_reset_summary["edges"]
