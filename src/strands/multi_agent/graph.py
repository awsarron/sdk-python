"""Directed Acyclic Graph (DAG) Multi-Agent Pattern Implementation.

This module provides a deterministic DAG-based agent orchestration system where
agents or MultiAgentBase instances (like Swarm or AgentGraph) are nodes in a graph,
executed according to edge dependencies, with output from one node passed as input
to connected nodes.

Key Features:
- Agents and MultiAgentBase instances (Swarm, AgentGraph, etc.) as graph nodes
- Deterministic execution order based on DAG structure
- Output propagation along edges
- Topological sort for execution ordering
- Clear dependency management
- Supports nested graphs (AgentGraph as a node in another AgentGraph)
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..agent import Agent
from .base import MultiAgentBase, MultiAgentResult

logger = logging.getLogger(__name__)


class GraphStatus(Enum):
    """Graph execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeStatus(Enum):
    """Node execution status."""

    WAITING = "waiting"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GraphEdge:
    """Represents an edge in the agent graph with optional condition."""

    from_node: str
    to_node: str
    condition: Optional[Callable[[List[MultiAgentResult]], bool]] = None

    def __hash__(self) -> int:
        """Return hash for GraphEdge based on from_node and to_node."""
        return hash((self.from_node, self.to_node))

    def should_traverse(self, source_results: List[MultiAgentResult]) -> bool:
        """Check if this edge should be traversed based on condition."""
        if self.condition is None:
            return True

        try:
            return self.condition(source_results)
        except Exception as e:
            logger.warning(
                "from_node=<%s>, to_node=<%s>, error=<%s> | edge condition evaluation failed",
                self.from_node,
                self.to_node,
                e,
            )
            return False


@dataclass
class GraphNode:
    """Represents a node in the agent graph."""

    node_id: str
    executor: Union[Agent, MultiAgentBase]
    status: NodeStatus = NodeStatus.WAITING
    dependencies: Set[str] = field(default_factory=set)
    result: Optional[List[MultiAgentResult]] = None
    error: Optional[Exception] = None

    def is_ready(self, completed_nodes: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return self.dependencies.issubset(completed_nodes)


@dataclass
class GraphState:
    """Maintains the state of graph execution."""

    status: GraphStatus = GraphStatus.PENDING
    completed_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    node_results: Dict[str, List[MultiAgentResult]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    task: str = ""
    pending_user_input: Optional[str] = None

    def reset(self, nodes: Optional[Dict[str, GraphNode]] = None) -> None:
        """Reset the graph state for a new execution."""
        self.status = GraphStatus.PENDING
        self.completed_nodes.clear()
        self.failed_nodes.clear()
        self.node_results.clear()
        self.execution_order.clear()
        self.task = ""
        self.pending_user_input = None


class GraphBuilder:
    """Builder for constructing agent graphs."""

    def __init__(self) -> None:
        """Initialize GraphBuilder with empty nodes and edges."""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Set[GraphEdge] = set()
        self.entry_points: Set[str] = set()

    def add_node(self, executor: Union[Agent, MultiAgentBase], node_id: Optional[str] = None) -> "GraphBuilder":
        """Add an agent or MultiAgentBase instance as a node to the graph."""
        if node_id is None:
            if hasattr(executor, "id") and executor.id:
                node_id = executor.id
            elif hasattr(executor, "name") and executor.name:
                node_id = executor.name
            else:
                node_id = f"node_{len(self.nodes)}"

        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists")

        self.nodes[node_id] = GraphNode(node_id=node_id, executor=executor)

        return self

    def add_edge(
        self, from_node: str, to_node: str, condition: Optional[Callable[[List[MultiAgentResult]], bool]] = None
    ) -> "GraphBuilder":
        """Add an edge between two nodes with optional condition function."""
        if from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' not found")
        if to_node not in self.nodes:
            raise ValueError(f"Target node '{to_node}' not found")

        edge = GraphEdge(from_node=from_node, to_node=to_node, condition=condition)
        self.edges.add(edge)

        # Update dependencies - conditional edges still create dependencies for topological ordering
        # but won't be traversed if condition is false
        self.nodes[to_node].dependencies.add(from_node)

        return self

    def set_entry_point(self, node_id: str) -> "GraphBuilder":
        """Set a node as an entry point for graph execution."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not found")

        self.entry_points.add(node_id)
        return self

    def build(self) -> "AgentGraph":
        """Build and validate the graph."""
        if not self.nodes:
            raise ValueError("Graph must contain at least one node")

        # Auto-detect entry points if none specified
        if not self.entry_points:
            potential_entries = [node_id for node_id, node in self.nodes.items() if not node.dependencies]
            if potential_entries:
                self.entry_points.update(potential_entries)
            else:
                raise ValueError("No entry points found - all nodes have dependencies")

        # Validate entry points exist
        for entry_point in self.entry_points:
            if entry_point not in self.nodes:
                raise ValueError(f"Entry point '{entry_point}' not found in nodes")

        # Check for cycles
        if self._has_cycles():
            raise ValueError("Graph contains cycles - must be a DAG")

        return AgentGraph(nodes=self.nodes.copy(), edges=self.edges.copy(), entry_points=self.entry_points.copy())

    def _has_cycles(self) -> bool:
        """Detect cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self.nodes}

        def dfs(node_id: str) -> bool:
            if colors[node_id] == GRAY:
                return True  # Back edge found - cycle detected
            if colors[node_id] == BLACK:
                return False

            colors[node_id] = GRAY

            # Check all outgoing edges
            for edge in self.edges:
                if edge.from_node == node_id:
                    if dfs(edge.to_node):
                        return True

            colors[node_id] = BLACK
            return False

        for node_id in self.nodes:
            if colors[node_id] == WHITE:
                if dfs(node_id):
                    return True
        return False


class AgentGraph(MultiAgentBase):
    """Directed Acyclic Graph multi-agent coordination system.

    Orchestrates agents and MultiAgentBase instances as nodes in a graph with
    deterministic execution order based on DAG structure and dependency resolution.
    Can be used as a node in another AgentGraph for nested orchestration.
    """

    def __init__(self, nodes: Dict[str, GraphNode], edges: Set[GraphEdge], entry_points: Set[str]) -> None:
        """Initialize AgentGraph with nodes, edges, and entry points."""
        # For MultiAgentBase, we only track individual Agent instances
        # MultiAgentBase instances are treated as black boxes
        agents = []
        for node in nodes.values():
            if isinstance(node.executor, Agent):
                agents.append(node.executor)

        super().__init__(agents=agents)

        self.nodes = nodes
        self.edges = edges
        self.entry_points = entry_points
        self.state = GraphState()

        # Build adjacency lists for efficient traversal
        self.outgoing_edges: Dict[str, List[str]] = defaultdict(list)
        self.incoming_edges: Dict[str, List[str]] = defaultdict(list)

        for edge in self.edges:
            self.outgoing_edges[edge.from_node].append(edge.to_node)
            self.incoming_edges[edge.to_node].append(edge.from_node)

    async def execute(self, task: str) -> List[MultiAgentResult]:
        """Execute the graph with conditional edge routing."""
        logger.debug("task=<%s> | starting graph execution", task)

        # Initialize state
        self.state = GraphState(status=GraphStatus.RUNNING, task=task)

        try:
            # Execute starting from entry points
            await self._execute_conditional_flow()

            if self.state.status == GraphStatus.RUNNING:
                self.state.status = GraphStatus.COMPLETED

            logger.debug("status=<%s> | graph execution completed", self.state.status)

        except Exception as e:
            logger.error("error=<%s> | graph execution failed", e)
            self.state.status = GraphStatus.FAILED
            raise

        # Flatten all results from executed nodes
        all_results = []
        for results in self.state.node_results.values():
            all_results.extend(results)

        return all_results

    async def _execute_conditional_flow(self) -> None:
        """Execute nodes respecting conditional edges."""
        # Start with entry points
        ready_nodes = list(self.entry_points)

        while ready_nodes:
            current_batch = ready_nodes.copy()
            ready_nodes.clear()

            # Execute current batch
            for node_id in current_batch:
                if node_id not in self.state.completed_nodes:
                    await self._execute_node(node_id)

                    # Find nodes that might now be ready
                    for next_node_id in self.nodes:
                        if (
                            next_node_id not in self.state.completed_nodes
                            and next_node_id not in ready_nodes
                            and self._is_node_ready_conditional(next_node_id)
                        ):
                            ready_nodes.append(next_node_id)

    def _is_node_ready_conditional(self, node_id: str) -> bool:
        """Check if a node is ready considering conditional edges."""
        # Get all incoming edges to this node
        incoming_edges = [edge for edge in self.edges if edge.to_node == node_id]

        if not incoming_edges:
            # No incoming edges, so it's ready if it's an entry point
            return node_id in self.entry_points

        # Check if at least one incoming edge condition is satisfied
        for edge in incoming_edges:
            source_node_id = edge.from_node

            # Source must be completed
            if source_node_id not in self.state.completed_nodes:
                continue

            # Check edge condition
            if source_node_id in self.state.node_results:
                source_results = self.state.node_results[source_node_id]
                if edge.should_traverse(source_results):
                    logger.debug(
                        "node_id=<%s>, source_node_id=<%s> | node ready via satisfied condition",
                        node_id,
                        source_node_id,
                    )
                    return True
                else:
                    logger.debug(
                        "source_node_id=<%s>, node_id=<%s> | edge condition not satisfied", source_node_id, node_id
                    )

        return False

    # TODO: implement resume_from_user_input
    async def resume_from_user_input(self, user_response: str) -> List[MultiAgentResult]:
        """Resume execution after user input (for nodes that support it)."""
        logger.debug("user_response=<%s> | resuming graph execution with user input", user_response)

        if self.state.status != GraphStatus.RUNNING:
            raise ValueError("Cannot resume - graph is not in running state")

        # TODO: dummy logic below
        results = []

        for node_id, node in self.nodes.items():
            if isinstance(node.executor, MultiAgentBase) and hasattr(node.executor, "resume_from_user_input"):
                try:
                    node_results = await node.executor.resume_from_user_input(user_response)
                    results.extend(node_results)
                except Exception as e:
                    logger.error("node_id=<%s>, error=<%s> | node failed to resume", node_id, e)

        return results

    def _get_topological_order(self) -> List[str]:
        """Get topological ordering of nodes for deterministic execution."""
        in_degree = {
            node_id: len(deps) for node_id, deps in {n: self.nodes[n].dependencies for n in self.nodes}.items()
        }

        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            # Update in-degrees for dependent nodes
            for neighbor in self.outgoing_edges[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles or unreachable nodes")

        return result

    async def _execute_node(self, node_id: str) -> None:
        """Execute a single node with input from satisfied dependencies."""
        node = self.nodes[node_id]
        node.status = NodeStatus.EXECUTING

        logger.debug("node_id=<%s> | executing node", node_id)

        start_time = time.time()
        try:
            # Collect inputs from dependency nodes that led to this execution
            dependency_results = {}
            for edge in self.edges:
                if edge.to_node == node_id and edge.from_node in self.state.completed_nodes:
                    source_results = self.state.node_results[edge.from_node]
                    if edge.should_traverse(source_results):
                        dependency_results[edge.from_node] = source_results

            # Build input text from dependencies
            input_text = self._build_input_text(node_id, dependency_results)

            # Execute based on node type
            if isinstance(node.executor, MultiAgentBase):
                result = await node.executor.execute(input_text)  # TODO: apply message filter before execute()
            else:
                # Agent instances - call synchronously and convert result
                agent_response = node.executor(input_text)  # TODO: apply message filter before executor()
                execution_time = time.time() - start_time
                multi_agent_result = MultiAgentResult(
                    agent_name=node.executor.name or node_id,
                    task_id=node_id,
                    status="completed",
                    result=agent_response,
                    execution_time=execution_time,
                )
                result = [multi_agent_result]

            # Mark as completed
            node.status = NodeStatus.COMPLETED
            node.result = result
            self.state.completed_nodes.add(node_id)
            self.state.node_results[node_id] = result
            self.state.execution_order.append(node_id)

            logger.debug("node_id=<%s>, execution_time=<%.2fs> | node completed successfully", node_id, execution_time)

        except Exception as e:
            logger.error("node_id=<%s>, error=<%s> | node failed", node_id, e)
            node.status = NodeStatus.FAILED
            node.error = e
            self.state.failed_nodes.add(node_id)
            raise

    def _build_input_text(self, node_id: str, dependency_results: Dict[str, List[MultiAgentResult]]) -> str:
        """Build input text for a node based on dependency outputs."""
        if not dependency_results:
            return self.state.task

        # Combine task with dependency outputs
        input_parts = [f"Original Task: {self.state.task}"]

        if dependency_results:
            input_parts.append("\nInputs from previous nodes:")
            for dep_id, results in dependency_results.items():
                input_parts.append(f"\nFrom {dep_id}:")
                for result in results:
                    result_text = self._extract_result_text(result)
                    input_parts.append(f"  - {result.agent_name}: {result_text}")

        return "\n".join(input_parts)

    def _extract_result_text(self, result: MultiAgentResult) -> str:
        """Extract text content from an agent result."""
        if hasattr(result, "result"):
            result_data = result.result
        else:
            result_data = result

        if isinstance(result_data, dict):
            if "content" in result_data:
                if isinstance(result_data["content"], list):
                    texts = []
                    for item in result_data["content"]:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                        elif isinstance(item, str):
                            texts.append(item)
                    return "\n".join(texts)
                else:
                    return str(result_data["content"])
            else:
                return str(result_data)
        elif isinstance(result_data, str):
            return result_data
        else:
            return str(result_data)

    def reset_state(self) -> None:
        """Reset the graph state and node states for a new execution."""
        self.state.reset()

        # Reset individual node states
        for node in self.nodes.values():
            node.status = NodeStatus.WAITING
            node.result = None
            node.error = None

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of graph execution."""
        return {
            "status": self.state.status.value,
            "total_nodes": len(self.nodes),
            "completed_nodes": len(self.state.completed_nodes),
            "failed_nodes": len(self.state.failed_nodes),
            "execution_order": self.state.execution_order,
            "node_statuses": {node_id: node.status.value for node_id, node in self.nodes.items()},
            "node_types": {node_id: type(node.executor).__name__ for node_id, node in self.nodes.items()},
            "edges": [(edge.from_node, edge.to_node) for edge in self.edges],
            "entry_points": list(self.entry_points),
        }

    def __str__(self) -> str:
        """Create a simple text visualization of the DAG."""
        lines = ["Agent Graph Structure:"]
        lines.append(f"Nodes ({len(self.nodes)}):")
        for node_id, node in self.nodes.items():
            node_type = type(node.executor).__name__
            status_info = ""
            if node.status != NodeStatus.WAITING:
                status_info = f" [{node.status.value}]"
            lines.append(f"  {node_id} ({node_type}){status_info}")

            # If this node is an AgentGraph, show its sub-nodes
            if isinstance(node.executor, AgentGraph):
                sub_graph_viz = str(node.executor)
                # Indent the sub-graph visualization and add it
                for line in sub_graph_viz.split("\n"):
                    if line.strip():  # Skip empty lines
                        lines.append(f"    └─ {line}")
            # If this node is a MultiAgentBase (like Swarm but not AgentGraph), show its agents
            elif isinstance(node.executor, MultiAgentBase):
                for agent in node.executor.agents:
                    agent_name = getattr(agent, "name", "unknown")
                    lines.append(f"    └─ {agent_name} (Agent)")

        lines.append(f"Entry Points: {list(self.entry_points)}")
        lines.append("Edges:")
        for edge in sorted(self.edges, key=lambda e: (e.from_node, e.to_node)):
            condition_info = " [conditional]" if edge.condition is not None else ""
            lines.append(f"  {edge.from_node} -> {edge.to_node}{condition_info}")

        if self.state.execution_order:
            lines.append(f"Execution Order: {self.state.execution_order}")

        if self.state.completed_nodes:
            lines.append(f"Completed Nodes: {sorted(list(self.state.completed_nodes))}")

        if self.state.failed_nodes:
            lines.append(f"Failed Nodes: {sorted(list(self.state.failed_nodes))}")

        return "\n".join(lines)
