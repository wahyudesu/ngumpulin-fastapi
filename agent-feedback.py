import os
import getpass
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing_extensions import TypedDict

# Jika perlu, set API key Groq
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# _set_env("GROQ_API_KEY")  # uncomment jika diperlukan

# Inisialisasi LLM dari Groq (misal menggunakan model "gemma2-9b-it")
llm = ChatGroq(model="gemma2-9b-it")

# ----- Tipe Data State -----
class AssignmentMeta(TypedDict):
    title: str
    description: str

class State(TypedDict):
    assignment_meta: AssignmentMeta
    assignment_content: str
    summary: str
    relevance_analysis: str
    feedback_analysis: str
    personalized_feedback: str
    combined_output: str

# ----- Definisi Node -----
def input_meta(state: State) -> dict:
    return {"assignment_meta": state["assignment_meta"]}

def input_content(state: State) -> dict:
    return {"assignment_content": state["assignment_content"]}

def summarizer_agent(state: State) -> dict:
    content = state["assignment_content"]
    msg = llm.invoke(f"Summarize the following assignment content:\n\n{content}")
    return {"summary": msg.content}

def relevance_agent(state: State) -> dict:
    title = state["assignment_meta"]["title"]
    desc = state["assignment_meta"]["description"]
    prompt = f"""Analyze the relevance between the following title and description of an assignment:

Title: {title}
Description: {desc}

Does the title appropriately reflect the content described? Provide analysis."""
    msg = llm.invoke(prompt)
    return {"relevance_analysis": msg.content}

def aggregator(state: State) -> dict:
    summary = state["summary"]
    relevance = state["relevance_analysis"]
    feedback_prompt = f"""You are an academic evaluator. Provide constructive feedback based on the following:

SUMMARY:
{summary}

RELEVANCE ANALYSIS:
{relevance}"""
    feedback = llm.invoke(feedback_prompt).content
    personalization_prompt = f"""As Professor Dr. Ahmad Yunus, personalize the following academic feedback for a student:

FEEDBACK:
{feedback}"""
    personalized = llm.invoke(personalization_prompt).content
    combined = f"🎓 Final Feedback by Dr. Ahmad Yunus:\n\n{personalized}"
    return {
        "feedback_analysis": feedback,
        "personalized_feedback": personalized,
        "combined_output": combined
    }

# ----- Bangun Workflow Graph -----
builder = StateGraph(State)
builder.add_node("input_meta", input_meta)
builder.add_node("input_content", input_content)
builder.add_node("relevance_agent", relevance_agent)
builder.add_node("summarizer_agent", summarizer_agent)
builder.add_node("aggregator", aggregator)

builder.add_edge(START, "input_meta")
builder.add_edge(START, "input_content")
builder.add_edge("input_meta", "relevance_agent")
builder.add_edge("input_content", "summarizer_agent")
builder.add_edge("relevance_agent", "aggregator")
builder.add_edge("summarizer_agent", "aggregator")
builder.add_edge("aggregator", END)

workflow = builder.compile()

def process_assignment(assignment_meta: AssignmentMeta, assignment_content: str) -> dict:
    """
    Memproses assignment dengan workflow agent feedback.
    
    :param assignment_meta: Dictionary dengan key "title" dan "description".
    :param assignment_content: Isi assignment berupa string.
    :return: Hasil feedback yang dihasilkan oleh agent.
    """
    initial_state: State = {
        "assignment_meta": assignment_meta,
        "assignment_content": assignment_content,
        "summary": "",
        "relevance_analysis": "",
        "feedback_analysis": "",
        "personalized_feedback": "",
        "combined_output": ""
    }
    result = workflow.invoke(initial_state)
    return result

# Untuk pengujian lokal
if __name__ == "__main__":
    sample_meta = {
        "title": "Sustainable Urban Planning",
        "description": "Analyze urban sustainability strategies in Southeast Asian megacities."
    }
    sample_content = (
        "Urban sustainability in Southeast Asia faces challenges like population density, flooding, "
        "and transport inefficiencies. Green corridors, public transit, and zoning laws can enhance livability. "
        "Jakarta and Bangkok are case studies with both success and setbacks."
    )
    feedback = process_assignment(sample_meta, sample_content)
    print(feedback.get("combined_output"))