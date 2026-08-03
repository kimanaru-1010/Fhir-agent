"""LangGraph pipeline builder — wires node functions into a StateGraph.

Pipeline flow:
  START -> visual_extract (multimodal vision analysis)
         -> clinical_planner_round1 — verify findings + generate 5 questions
         -> user_interview_round1 (interrupt loop) — collect 5 Yes/No answers
         -> clinical_planner_round2 — generate 5 NEW follow-up questions
         -> user_interview_round2 (interrupt loop) — collect 5 more answers
         -> diagnostic_reasoning — ranked diagnoses + evidence
         -> END
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from pipeline.nodes.diagnostic_reasoning import diagnostic_reasoning
from pipeline.nodes.interview import user_interview_round1, user_interview_round2
from pipeline.nodes.planner import clinical_planner_round1, clinical_planner_round2
from pipeline.nodes.visual_extract import visual_extract


class PipelineState(TypedDict):
    image_path: str                           # Original skin image path
    anamnesis: str                            # Chief complaint
    visual_observations: str                  # Visual observations with confidence
    visual_differentials: list[str]           # Top-10 differential disease names
    updated_visual_findings: str              # Planner-verified + enriched findings
    round1_questions: list[dict]              # Questions from Clinical Planner (Round 1)
    round2_questions: list[dict]              # Questions from Clinical Planner (Round 2)
    round1_qa_pairs: list                     # Q&A pairs from Round 1
    round2_qa_pairs: list                     # Q&A pairs from Round 2
    qa_history: str                           # Full Q&A history (both rounds, formatted)
    ranked_diagnoses: list                    # Final ranked diagnoses
    reasoning: str                            # Clinical synthesis in Vietnamese


def build_pipeline():
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("visual_extract", visual_extract)
    graph.add_node("clinical_planner_round1", clinical_planner_round1)
    graph.add_node("user_interview_round1", user_interview_round1)
    graph.add_node("clinical_planner_round2", clinical_planner_round2)
    graph.add_node("user_interview_round2", user_interview_round2)
    graph.add_node("diagnostic_reasoning", diagnostic_reasoning)

    # Linear edges
    graph.add_edge(START, "visual_extract")
    graph.add_edge("visual_extract", "clinical_planner_round1")
    graph.add_edge("clinical_planner_round1", "user_interview_round1")
    graph.add_edge("user_interview_round1", "clinical_planner_round2")
    graph.add_edge("clinical_planner_round2", "user_interview_round2")
    graph.add_edge("user_interview_round2", "diagnostic_reasoning")
    graph.add_edge("diagnostic_reasoning", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
