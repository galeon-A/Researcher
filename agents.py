"""
agents/researcher.py
Defines all 4 CrewAI agents for the research paper pipeline:
  1. Researcher   – fetches & indexes arXiv papers
  2. Summarizer   – RAG-based summarization
  3. Writer       – full paper drafting
  4. Reviewer     – critique & refinement
"""

from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI


def build_llm(api_key: str, model: str = "gemini-2.0-flash") -> ChatGoogleGenerativeAI:
    """Build the shared Gemini LLM used by all agents."""
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True,
    )


def create_researcher_agent(llm, arxiv_tool) -> Agent:
    """
    Agent 1 – Research Specialist
    Responsibility: Search arXiv, gather top papers on the topic.
    """
    return Agent(
        role="Senior Research Specialist",
        goal=(
            "Search arXiv for the most relevant and recent academic papers on the given topic. "
            "Gather comprehensive source material including paper titles, authors, abstracts, "
            "publication dates, and key findings to build a solid knowledge base."
        ),
        backstory=(
            "You are an expert academic librarian with decades of experience in scientific "
            "literature discovery. You know how to craft precise search queries to surface "
            "the most impactful papers across all domains. You are meticulous about capturing "
            "full citation information and identifying the key contributions of each paper."
        ),
        tools=[arxiv_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def create_summarizer_agent(llm) -> Agent:
    """
    Agent 2 – Research Summarizer
    Responsibility: Synthesize retrieved papers into structured summaries.
    """
    return Agent(
        role="Research Synthesis Expert",
        goal=(
            "Analyze the collected research papers and produce clear, structured summaries "
            "that capture the key methodologies, findings, contributions, and limitations "
            "of each paper. Identify common themes, research gaps, and emerging trends."
        ),
        backstory=(
            "You are a seasoned academic who has reviewed thousands of papers across multiple "
            "disciplines. You excel at distilling complex technical content into clear summaries "
            "that preserve scientific accuracy. You are skilled at identifying connections between "
            "different works and spotting research gaps that need to be addressed."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def create_writer_agent(llm) -> Agent:
    """
    Agent 3 – Academic Paper Writer
    Responsibility: Write the full research paper from summaries.
    """
    return Agent(
        role="Academic Paper Writer",
        goal=(
            "Write a comprehensive, well-structured research paper using the summaries and "
            "source material. The paper must include: Abstract, Introduction, Related Work, "
            "Methodology/Discussion, Results & Analysis, Conclusion, and References. "
            "Follow academic writing conventions with precise, formal language."
        ),
        backstory=(
            "You are a prolific academic writer with publications in top-tier venues. "
            "You understand the structure and conventions of research papers across all fields. "
            "You write with clarity and precision, weaving together evidence from multiple sources "
            "into a coherent narrative. You always support claims with citations and maintain "
            "academic integrity throughout."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def create_reviewer_agent(llm) -> Agent:
    """
    Agent 4 – Critical Reviewer & Editor
    Responsibility: Review, critique, and polish the paper.
    """
    return Agent(
        role="Senior Academic Reviewer & Editor",
        goal=(
            "Critically review the research paper draft for scientific accuracy, logical coherence, "
            "completeness, and writing quality. Identify weaknesses, suggest improvements, ensure "
            "all sections are well-developed, citations are properly formatted, and the paper meets "
            "academic publication standards. Produce a final polished version."
        ),
        backstory=(
            "You are a veteran journal editor who has reviewed papers for Nature, IEEE, and ACM. "
            "You have a sharp eye for logical gaps, unsupported claims, weak methodology, and "
            "unclear writing. Your reviews are constructive but rigorous. You ensure every paper "
            "you touch is publication-ready: well-argued, properly cited, and clearly written."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )
