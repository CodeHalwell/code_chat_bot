"""AI Agents for performing specific tasks autonomously."""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentTask:
    """Task for an agent to execute."""
    name: str
    description: str
    parameters: Dict[str, Any]
    priority: int = 0


@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class BaseAgent(ABC):
    """Base class for AI agents."""

    def __init__(self, name: str, description: str):
        """
        Initialize agent.

        Args:
            name: Agent name
            description: Agent description
        """
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.history: List[AgentResult] = []

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute a task.

        Args:
            task: Task to execute

        Returns:
            AgentResult with execution results
        """
        pass

    def run(self, task: AgentTask) -> AgentResult:
        """
        Run a task and track status.

        Args:
            task: Task to execute

        Returns:
            AgentResult with execution results
        """
        self.status = AgentStatus.RUNNING
        try:
            result = self.execute(task)
            self.status = AgentStatus.COMPLETED if result.success else AgentStatus.FAILED
            self.history.append(result)
            return result
        except Exception as e:
            result = AgentResult(
                success=False,
                output=None,
                error=str(e)
            )
            self.status = AgentStatus.FAILED
            self.history.append(result)
            return result

    def reset(self):
        """Reset agent status."""
        self.status = AgentStatus.IDLE
        self.history = []


class ResearchAgent(BaseAgent):
    """Agent for researching information from documents."""

    def __init__(self, vector_store=None):
        """
        Initialize research agent.

        Args:
            vector_store: Vector store for document search
        """
        super().__init__(
            name="ResearchAgent",
            description="Researches information from uploaded documents"
        )
        self.vector_store = vector_store

    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute research task.

        Args:
            task: Task with 'query' parameter

        Returns:
            AgentResult with research findings
        """
        query = task.parameters.get("query", "")
        top_k = task.parameters.get("top_k", 5)

        if not self.vector_store:
            return AgentResult(
                success=False,
                output=None,
                error="Vector store not initialized"
            )

        try:
            results = self.vector_store.search(query, top_k=top_k)
            return AgentResult(
                success=True,
                output=results,
                metadata={"query": query, "results_count": len(results)}
            )
        except Exception as e:
            return AgentResult(
                success=False,
                output=None,
                error=str(e)
            )


class SummarizerAgent(BaseAgent):
    """Agent for summarizing text."""

    def __init__(self, ai_provider=None):
        """
        Initialize summarizer agent.

        Args:
            ai_provider: AI provider for generating summaries
        """
        super().__init__(
            name="SummarizerAgent",
            description="Summarizes long text into concise summaries"
        )
        self.ai_provider = ai_provider

    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute summarization task.

        Args:
            task: Task with 'text' and optional 'max_length' parameters

        Returns:
            AgentResult with summary
        """
        text = task.parameters.get("text", "")
        max_length = task.parameters.get("max_length", 200)

        if not text:
            return AgentResult(
                success=False,
                output=None,
                error="No text provided for summarization"
            )

        try:
            # Simple summarization by taking first sentences up to max_length
            sentences = text.split(". ")
            summary = ""
            for sentence in sentences:
                if len(summary) + len(sentence) < max_length:
                    summary += sentence + ". "
                else:
                    break

            if not summary:
                summary = text[:max_length] + "..."

            return AgentResult(
                success=True,
                output=summary.strip(),
                metadata={"original_length": len(text), "summary_length": len(summary)}
            )
        except Exception as e:
            return AgentResult(
                success=False,
                output=None,
                error=str(e)
            )


class DataAnalysisAgent(BaseAgent):
    """Agent for analyzing data and generating insights."""

    def __init__(self):
        super().__init__(
            name="DataAnalysisAgent",
            description="Analyzes data and generates insights"
        )

    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute data analysis task.

        Args:
            task: Task with 'data' parameter

        Returns:
            AgentResult with analysis
        """
        data = task.parameters.get("data", [])

        if not data:
            return AgentResult(
                success=False,
                output=None,
                error="No data provided for analysis"
            )

        try:
            # Simple analysis
            analysis = {
                "count": len(data),
                "type": type(data).__name__,
            }

            # Add numeric analysis if applicable
            if isinstance(data, list) and data and isinstance(data[0], (int, float)):
                analysis.update({
                    "min": min(data),
                    "max": max(data),
                    "avg": sum(data) / len(data) if data else 0
                })

            return AgentResult(
                success=True,
                output=analysis,
                metadata={"data_type": type(data).__name__}
            )
        except Exception as e:
            return AgentResult(
                success=False,
                output=None,
                error=str(e)
            )


class QuestionAnsweringAgent(BaseAgent):
    """Agent for answering questions using RAG."""

    def __init__(self, ai_provider=None, vector_store=None):
        """
        Initialize QA agent.

        Args:
            ai_provider: AI provider for generating answers
            vector_store: Vector store for retrieving context
        """
        super().__init__(
            name="QuestionAnsweringAgent",
            description="Answers questions using Retrieval-Augmented Generation"
        )
        self.ai_provider = ai_provider
        self.vector_store = vector_store

    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute question answering task.

        Args:
            task: Task with 'question' parameter

        Returns:
            AgentResult with answer
        """
        question = task.parameters.get("question", "")
        use_rag = task.parameters.get("use_rag", True)

        if not question:
            return AgentResult(
                success=False,
                output=None,
                error="No question provided"
            )

        try:
            context = ""
            if use_rag and self.vector_store:
                # Retrieve relevant context
                results = self.vector_store.search(question, top_k=3)
                context = "\n\n".join([r.content for r in results])

            # Generate answer (simplified - in real implementation, use AI provider)
            answer = f"Based on the provided information: {context[:200]}..." if context else "No context available."

            return AgentResult(
                success=True,
                output=answer,
                metadata={
                    "question": question,
                    "used_rag": use_rag,
                    "context_length": len(context)
                }
            )
        except Exception as e:
            return AgentResult(
                success=False,
                output=None,
                error=str(e)
            )


class AgentOrchestrator:
    """Orchestrates multiple agents to work together."""

    def __init__(self):
        """Initialize agent orchestrator."""
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[AgentTask] = []

    def register_agent(self, agent: BaseAgent):
        """
        Register an agent.

        Args:
            agent: Agent to register
        """
        self.agents[agent.name] = agent

    def unregister_agent(self, agent_name: str):
        """
        Unregister an agent.

        Args:
            agent_name: Name of agent to unregister
        """
        if agent_name in self.agents:
            del self.agents[agent_name]

    def execute_task(self, agent_name: str, task: AgentTask) -> AgentResult:
        """
        Execute a task with a specific agent.

        Args:
            agent_name: Name of agent to use
            task: Task to execute

        Returns:
            AgentResult with execution results
        """
        if agent_name not in self.agents:
            return AgentResult(
                success=False,
                output=None,
                error=f"Agent '{agent_name}' not found"
            )

        agent = self.agents[agent_name]
        return agent.run(task)

    def execute_pipeline(self, pipeline: List[tuple]) -> List[AgentResult]:
        """
        Execute a pipeline of tasks.

        Args:
            pipeline: List of (agent_name, task) tuples

        Returns:
            List of AgentResults
        """
        results = []
        for agent_name, task in pipeline:
            result = self.execute_task(agent_name, task)
            results.append(result)
            if not result.success:
                break  # Stop on first failure
        return results

    def get_agent_status(self, agent_name: str) -> Optional[AgentStatus]:
        """
        Get status of an agent.

        Args:
            agent_name: Name of agent

        Returns:
            AgentStatus or None if not found
        """
        agent = self.agents.get(agent_name)
        return agent.status if agent else None

    def get_all_agents(self) -> List[str]:
        """Get list of all registered agent names."""
        return list(self.agents.keys())

    def reset_all_agents(self):
        """Reset all agents."""
        for agent in self.agents.values():
            agent.reset()


# Factory functions for easy agent creation
def create_research_agent(vector_store=None) -> ResearchAgent:
    """Create a research agent."""
    return ResearchAgent(vector_store)


def create_summarizer_agent(ai_provider=None) -> SummarizerAgent:
    """Create a summarizer agent."""
    return SummarizerAgent(ai_provider)


def create_qa_agent(ai_provider=None, vector_store=None) -> QuestionAnsweringAgent:
    """Create a question-answering agent."""
    return QuestionAnsweringAgent(ai_provider, vector_store)


def create_data_analysis_agent() -> DataAnalysisAgent:
    """Create a data analysis agent."""
    return DataAnalysisAgent()
