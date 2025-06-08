from unittest.mock import AsyncMock, Mock, patch

import pytest

from strands.agent import Agent, AgentResult
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult
from strands.multiagent.graph import GraphBuilder, GraphEdge, GraphNode, GraphResult, GraphState, Status


@pytest.fixture
def mock_agent():
    """Create a mock Agent for testing."""
    agent = Mock(spec=Agent)
    agent.name = "test_agent"
    agent.id = "agent_1"

    # Mock the agent call to return an AgentResult
    mock_result = AgentResult(
        message={"role": "assistant", "content": [{"text": "Agent response"}]},
        stop_reason="end_turn",
        state={},
        metrics=Mock(
            accumulated_usage={"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            accumulated_metrics={"latencyMs": 100.0},
        ),
    )
    agent.return_value = mock_result
    agent.__call__ = Mock(return_value=mock_result)
    return agent


@pytest.fixture
def mock_multi_agent():
    """Create a mock MultiAgentBase for testing."""
    multi_agent = Mock(spec=MultiAgentBase)
    multi_agent.name = "test_multi_agent"
    multi_agent.id = "multi_agent_1"

    # Mock the execute method to return a MultiAgentResult
    mock_node_result = NodeResult(
        results=AgentResult(
            message={"role": "assistant", "content": [{"text": "Multi-agent response"}]},
            stop_reason="end_turn",
            state={},
            metrics={},
        )
    )
    mock_result = MultiAgentResult(
        results={"inner_node": mock_node_result},
        accumulated_usage={"inputTokens": 15, "outputTokens": 25, "totalTokens": 40},
        accumulated_metrics={"latencyMs": 150.0},
        execution_count=1,
        execution_time=150.0,
    )
    multi_agent.execute = AsyncMock(return_value=mock_result)
    return multi_agent


@pytest.mark.asyncio
async def test_comprehensive_graph_execution_with_diverse_topology(mock_agent, mock_multi_agent):
    """Test comprehensive graph execution with diverse nodes, conditional edges, and complex topology."""

    # Create additional mock agents for a diverse graph
    agent2 = Mock(spec=Agent)
    agent2.name = "conditional_agent"
    agent2.id = "agent_2"
    agent2_result = AgentResult(
        message={"role": "assistant", "content": [{"text": "Conditional response"}]},
        stop_reason="end_turn",
        state={},
        metrics=Mock(
            accumulated_usage={"inputTokens": 5, "outputTokens": 15, "totalTokens": 20},
            accumulated_metrics={"latencyMs": 75.0},
        ),
    )
    agent2.return_value = agent2_result
    agent2.__call__ = Mock(return_value=agent2_result)

    agent3 = Mock(spec=Agent)
    agent3.name = "final_agent"
    agent3.id = "agent_3"
    agent3_result = AgentResult(
        message={"role": "assistant", "content": [{"text": "Final response"}]},
        stop_reason="end_turn",
        state={},
        metrics=Mock(
            accumulated_usage={"inputTokens": 8, "outputTokens": 12, "totalTokens": 20},
            accumulated_metrics={"latencyMs": 50.0},
        ),
    )
    agent3.return_value = agent3_result
    agent3.__call__ = Mock(return_value=agent3_result)

    # Create a conditional function that checks if a specific node completed
    def condition_check_completion(state: GraphState) -> bool:
        return "start_agent" in state.completed_nodes

    # Build a complex graph with various node types and conditional edges
    builder = GraphBuilder()

    # Test adding nodes with auto-generated IDs and explicit IDs
    builder.add_node(mock_agent, "start_agent")  # Entry point
    builder.add_node(mock_multi_agent, "multi_node")  # MultiAgentBase node
    builder.add_node(agent2)  # Auto-generated node_id
    builder.add_node(agent3, "final_node")  # Explicit node_id

    # Test adding edges with and without conditions
    builder.add_edge("start_agent", "multi_node")
    builder.add_edge("start_agent", "agent_2", condition=condition_check_completion)
    builder.add_edge("multi_node", "final_node")
    builder.add_edge("agent_2", "final_node")

    # Test setting entry points
    builder.set_entry_point("start_agent")

    # Build the graph
    graph = builder.build()

    # Test graph structure and properties
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 4
    assert len(graph.entry_points) == 1
    assert "start_agent" in graph.entry_points

    # Test GraphNode properties
    start_node = graph.nodes["start_agent"]
    assert start_node.node_id == "start_agent"
    assert start_node.executor == mock_agent
    assert start_node.status == Status.PENDING
    assert len(start_node.dependencies) == 0

    multi_node = graph.nodes["multi_node"]
    assert multi_node.node_id == "multi_node"
    assert multi_node.executor == mock_multi_agent
    assert "start_agent" in multi_node.dependencies

    # Test GraphEdge properties and conditions
    conditional_edge = None
    for edge in graph.edges:
        if edge.from_node == "start_agent" and edge.to_node == "agent_2":
            conditional_edge = edge
            break

    assert conditional_edge is not None
    assert conditional_edge.condition is not None

    # Test edge condition evaluation
    empty_state = GraphState()
    assert not conditional_edge.should_traverse(empty_state)

    state_with_completion = GraphState(completed_nodes={"start_agent"})
    assert conditional_edge.should_traverse(state_with_completion)

    # Test graph string representation
    graph_str = str(graph)
    assert "Nodes (4):" in graph_str
    assert "start_agent (Mock)" in graph_str  # Updated to match Mock type
    assert "multi_node (Mock)" in graph_str
    assert "Entry Points: ['start_agent']" in graph_str
    assert "start_agent -> multi_node" in graph_str
    assert "[conditional]" in graph_str

    # Test graph execution with mocked timer
    with patch("strands.multiagent.graph.time.time") as mock_time:
        # Simulate realistic execution timeline over 42 seconds
        # Main start, then each node taking ~10 seconds to complete
        times = [
            0.0,  # Main execution start
            5.0,  # Node 1 start
            15.0,  # Node 1 end
            18.0,  # Node 2 start
            28.0,  # Node 2 end
            30.0,  # Node 3 start
            35.0,  # Node 3 end
            36.0,  # Node 4 start
            40.0,  # Node 4 end
            42.0,  # Main execution end
        ]
        mock_time.side_effect = times

        result = await graph.execute("Test complex graph execution")

        # Verify execution completed successfully
        assert result.status == Status.COMPLETED
        assert result.total_nodes == 4
        assert result.completed_nodes == 4
        assert result.failed_nodes == 0
        assert len(result.execution_order) == 4
        assert "start_agent" == result.execution_order[0]  # Entry point should execute first

        # Verify all agents were called
        mock_agent.assert_called_once()
        mock_multi_agent.execute.assert_called_once()
        agent2.assert_called_once()
        agent3.assert_called_once()

        # Verify accumulated metrics
        assert result.accumulated_usage["totalTokens"] > 0
        assert result.accumulated_metrics["latencyMs"] > 0
        assert result.execution_count >= 4
        assert result.execution_time == 40000  # 40 seconds (last node)

        # Verify timer was called multiple times for realistic execution
        assert mock_time.call_count >= 2

    # Verify node results are stored
    assert len(result.results) == 4
    assert "start_agent" in result.results
    assert "multi_node" in result.results
    assert "agent_2" in result.results
    assert "final_node" in result.results

    # Test result content extraction
    start_result = result.results["start_agent"]
    assert start_result.status == Status.COMPLETED
    agent_results = start_result.get_agent_results()
    assert len(agent_results) == 1
    assert "Agent response" in str(agent_results[0].message)

    # Verify graph state after execution
    assert graph.state.status == Status.COMPLETED
    assert len(graph.state.completed_nodes) == 4
    assert len(graph.state.failed_nodes) == 0

    # Test GraphResult properties
    assert isinstance(result, GraphResult)
    assert isinstance(result, MultiAgentResult)  # GraphResult extends MultiAgentResult
    expected_edges = {
        ("start_agent", "multi_node"),
        ("start_agent", "agent_2"),
        ("multi_node", "final_node"),
        ("agent_2", "final_node"),
    }
    assert set(result.edges) == expected_edges
    assert result.entry_points == ["start_agent"]


def test_graph_builder_validation_and_error_handling():
    """Test GraphBuilder validation, error handling, and edge cases."""

    # Test empty graph validation
    builder = GraphBuilder()
    with pytest.raises(ValueError, match="Graph must contain at least one node"):
        builder.build()

    # Test duplicate node IDs
    mock_agent1 = Mock(spec=Agent)
    mock_agent2 = Mock(spec=Agent)
    builder.add_node(mock_agent1, "duplicate_id")
    with pytest.raises(ValueError, match="Node 'duplicate_id' already exists"):
        builder.add_node(mock_agent2, "duplicate_id")

    # Test edge validation with non-existent nodes
    builder = GraphBuilder()
    builder.add_node(mock_agent1, "node1")
    with pytest.raises(ValueError, match="Target node 'nonexistent' not found"):
        builder.add_edge("node1", "nonexistent")

    with pytest.raises(ValueError, match="Source node 'nonexistent' not found"):
        builder.add_edge("nonexistent", "node1")

    # Test invalid entry point
    with pytest.raises(ValueError, match="Node 'invalid_entry' not found"):
        builder.set_entry_point("invalid_entry")

    # Test cycle detection - need to set an explicit entry point to avoid auto-detection
    mock_agent2 = Mock(spec=Agent)
    mock_agent3 = Mock(spec=Agent)
    builder = GraphBuilder()
    builder.add_node(mock_agent1, "a")
    builder.add_node(mock_agent2, "b")
    builder.add_node(mock_agent3, "c")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", "a")  # Creates cycle
    builder.set_entry_point("a")  # Set explicit entry point

    with pytest.raises(ValueError, match="Graph contains cycles"):
        builder.build()

    # Test auto-detection of entry points
    builder = GraphBuilder()
    builder.add_node(mock_agent1, "entry")  # No dependencies = entry point
    builder.add_node(mock_agent2, "dependent")
    builder.add_edge("entry", "dependent")

    graph = builder.build()
    assert graph.entry_points == {"entry"}

    # Test no entry points scenario (all nodes have dependencies in circular structure)
    builder = GraphBuilder()
    builder.add_node(mock_agent1, "a")
    builder.add_node(mock_agent2, "b")
    builder.add_edge("a", "b")
    builder.add_edge("b", "a")  # Both have dependencies

    with pytest.raises(ValueError, match="No entry points found - all nodes have dependencies"):
        builder.build()


def test_graph_dataclasses_and_enums():
    """Test dataclass initialization, properties, and enum behavior."""

    # Test Status enum
    assert Status.PENDING.value == "pending"
    assert Status.EXECUTING.value == "executing"
    assert Status.COMPLETED.value == "completed"
    assert Status.FAILED.value == "failed"

    # Test GraphState initialization and defaults
    state = GraphState()
    assert state.status == Status.PENDING
    assert len(state.completed_nodes) == 0
    assert len(state.failed_nodes) == 0
    assert state.task == ""
    assert state.accumulated_usage == {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    assert state.execution_count == 0

    # Test GraphState with custom values
    state = GraphState(status=Status.EXECUTING, task="custom task", total_nodes=5, execution_count=3)
    assert state.status == Status.EXECUTING
    assert state.task == "custom task"
    assert state.total_nodes == 5
    assert state.execution_count == 3

    # Test GraphEdge with and without condition
    edge_simple = GraphEdge("a", "b")
    assert edge_simple.from_node == "a"
    assert edge_simple.to_node == "b"
    assert edge_simple.condition is None
    assert edge_simple.should_traverse(GraphState())  # No condition always True

    def test_condition(state):
        return len(state.completed_nodes) > 0

    edge_conditional = GraphEdge("a", "b", condition=test_condition)
    assert edge_conditional.condition is not None
    assert not edge_conditional.should_traverse(GraphState())
    assert edge_conditional.should_traverse(GraphState(completed_nodes={"some_node"}))

    # Test GraphEdge hashing (for set operations)
    edge1 = GraphEdge("x", "y")
    edge2 = GraphEdge("x", "y")
    edge3 = GraphEdge("y", "x")
    assert hash(edge1) == hash(edge2)
    assert hash(edge1) != hash(edge3)

    # Test GraphNode initialization and ready check
    mock_agent = Mock(spec=Agent)
    node = GraphNode("test_node", mock_agent)
    assert node.node_id == "test_node"
    assert node.executor == mock_agent
    assert node.status == Status.PENDING
    assert len(node.dependencies) == 0
    assert node.is_ready(set())  # No dependencies = always ready

    node_with_deps = GraphNode("dependent_node", mock_agent, dependencies={"dep1", "dep2"})
    assert not node_with_deps.is_ready({"dep1"})  # Missing dep2
    assert node_with_deps.is_ready({"dep1", "dep2"})  # All deps satisfied
    assert node_with_deps.is_ready({"dep1", "dep2", "extra"})  # Extra deps OK


@pytest.mark.asyncio
async def test_graph_execution_error_handling():
    """Test graph execution with node failures and error propagation."""

    # Create a failing agent
    failing_agent = Mock(spec=Agent)
    failing_agent.name = "failing_agent"
    failing_agent.id = "fail_node"
    failing_agent.side_effect = Exception("Simulated failure")
    failing_agent.__call__ = Mock(side_effect=Exception("Simulated failure"))

    # Create successful agent
    success_agent = Mock(spec=Agent)
    success_agent.name = "success_agent"
    success_agent.return_value = AgentResult(
        message={"role": "assistant", "content": [{"text": "Success"}]}, stop_reason="end_turn", state={}, metrics={}
    )
    success_agent.__call__ = Mock(return_value=success_agent.return_value)

    # Build graph with failing node
    builder = GraphBuilder()
    builder.add_node(failing_agent, "fail_node")
    builder.add_node(success_agent, "success_node")
    builder.add_edge("fail_node", "success_node")
    builder.set_entry_point("fail_node")

    graph = builder.build()

    # Test that execution fails and propagates error
    with pytest.raises(Exception, match="Simulated failure"):
        await graph.execute("Test error handling")

    # Verify failure state
    assert graph.state.status == Status.FAILED
    assert "fail_node" in graph.state.failed_nodes
    assert len(graph.state.completed_nodes) == 0


@pytest.mark.asyncio
async def test_graph_resume_not_implemented():
    """Test that graph resume method raises NotImplementedError."""
    builder = GraphBuilder()
    mock_agent = Mock(spec=Agent)
    builder.add_node(mock_agent, "test_node")
    graph = builder.build()

    with pytest.raises(NotImplementedError, match="resume not implemented"):
        await graph.resume("test task", GraphState())
