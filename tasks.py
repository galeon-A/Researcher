"""
agents/tasks.py
Defines the 4 CrewAI tasks corresponding to each agent in the pipeline:
  Task 1 – Research Task      (Researcher Agent)
  Task 2 – Summarization Task (Summarizer Agent)
  Task 3 – Writing Task       (Writer Agent)
  Task 4 – Review Task        (Reviewer Agent)
"""

from crewai import Task


def create_research_task(agent, topic: str, num_papers: int = 10) -> Task:
    """Task 1: Search arXiv and gather papers on the topic."""
    return Task(
        description=(
            f"Search arXiv for research papers on the topic: '{topic}'.\n\n"
            f"Instructions:\n"
            f"1. Use the arxiv_search tool to fetch at least {num_papers} relevant papers.\n"
            f"2. Try multiple search queries if needed (e.g., use synonyms, sub-topics).\n"
            f"3. For each paper, capture: title, authors, publication date, arXiv ID, URL, and full abstract.\n"
            f"4. Focus on recent, high-impact papers (last 3-5 years preferred).\n"
            f"5. Cover different aspects and sub-topics within '{topic}'.\n\n"
            f"Output a structured list of all papers with their full details."
        ),
        expected_output=(
            "A structured list of research papers with:\n"
            "- Paper number, title, authors, date, arXiv ID, URL\n"
            "- Full abstract for each paper\n"
            "- Brief note on why each paper is relevant\n"
            f"Minimum {num_papers} papers covering various aspects of '{topic}'."
        ),
        agent=agent,
    )


def create_summarization_task(agent, topic: str, context_docs: str) -> Task:
    """Task 2: Summarize the collected papers with RAG context."""
    return Task(
        description=(
            f"Using the research papers collected on '{topic}', produce comprehensive summaries.\n\n"
            f"Context from knowledge base:\n{context_docs[:3000]}\n\n"
            f"Instructions:\n"
            f"1. Summarize each paper: key problem, methodology, main findings, contributions.\n"
            f"2. Identify 3-5 major themes or research directions across all papers.\n"
            f"3. Note agreements and disagreements between papers.\n"
            f"4. Identify research gaps and open problems in '{topic}'.\n"
            f"5. Create a thematic synthesis — don't just list papers, connect them.\n"
            f"6. Extract the most important quotes/statistics for the final paper.\n\n"
            f"Be thorough and precise. This will be the foundation for the research paper."
        ),
        expected_output=(
            "A comprehensive research synthesis containing:\n"
            "1. Individual summaries of each paper (150-200 words each)\n"
            "2. Major themes and research directions identified\n"
            "3. Agreements and contradictions in the literature\n"
            "4. Research gaps and open questions\n"
            "5. Key statistics and findings to include in the paper\n"
            "6. Suggested citation references for each claim"
        ),
        agent=agent,
    )


def create_writing_task(agent, topic: str, context_docs: str) -> Task:
    """Task 3: Write the full research paper."""
    return Task(
        description=(
            f"Write a complete, publication-quality research paper on '{topic}'.\n\n"
            f"Context from retrieved papers:\n{context_docs[:3000]}\n\n"
            f"Use ALL the summaries and research gathered in previous tasks.\n\n"
            f"Paper structure (write each section fully):\n"
            f"1. **Title** – Informative, specific title\n"
            f"2. **Abstract** – 200-250 words: problem, approach, findings, significance\n"
            f"3. **1. Introduction** – Background, motivation, problem statement, contributions, paper structure\n"
            f"4. **2. Related Work** – Survey of existing approaches, how this work differs\n"
            f"5. **3. Background / Theoretical Framework** – Key concepts and foundations\n"
            f"6. **4. Analysis & Discussion** – Deep dive into findings, methodologies, comparisons\n"
            f"7. **5. Results & Insights** – Synthesized results, patterns, implications\n"
            f"8. **6. Conclusion** – Summary, limitations, future work\n"
            f"9. **References** – All cited papers in IEEE/APA format with arXiv URLs\n\n"
            f"Requirements:\n"
            f"- Minimum 2500 words\n"
            f"- Formal academic language\n"
            f"- Cite papers using [Author, Year] format throughout\n"
            f"- Every claim must be supported by evidence from the papers\n"
            f"- Use markdown formatting with proper headers"
        ),
        expected_output=(
            "A complete research paper in Markdown format with:\n"
            "- All 9 sections fully written (min 2500 words total)\n"
            "- Proper in-text citations [Author, Year]\n"
            "- Formal academic tone\n"
            "- Logical flow between sections\n"
            "- Full reference list at the end"
        ),
        agent=agent,
    )


def create_review_task(agent, topic: str) -> Task:
    """Task 4: Review and polish the paper."""
    return Task(
        description=(
            f"Critically review and improve the research paper on '{topic}' written in the previous task.\n\n"
            f"Review checklist:\n"
            f"1. **Structure**: Are all sections present and well-developed?\n"
            f"2. **Argument**: Is there a clear, logical research narrative?\n"
            f"3. **Evidence**: Are all claims supported by citations?\n"
            f"4. **Clarity**: Is the writing clear, precise, and free of ambiguity?\n"
            f"5. **Completeness**: Is the related work thorough? Are gaps addressed?\n"
            f"6. **Abstract**: Does it accurately reflect the full paper?\n"
            f"7. **References**: Are they complete and properly formatted?\n"
            f"8. **Coherence**: Do sections flow naturally into each other?\n\n"
            f"Instructions:\n"
            f"- First, write a brief review report (strengths & weaknesses)\n"
            f"- Then produce the FINAL improved version of the complete paper\n"
            f"- Fix all issues identified in your review\n"
            f"- Enhance weak sections, strengthen transitions, clarify confusing parts\n"
            f"- Ensure the paper is ready for academic publication\n"
            f"- Keep all citations and add any missing ones\n"
            f"- Output the FULL final paper, not just the changes"
        ),
        expected_output=(
            "Two parts:\n"
            "**REVIEW REPORT:**\n"
            "- List of strengths\n"
            "- List of issues found and fixes applied\n\n"
            "**FINAL PAPER:**\n"
            "The complete, polished research paper in Markdown with:\n"
            "- All improvements incorporated\n"
            "- Minimum 2500 words\n"
            "- All sections complete\n"
            "- Proper citations throughout\n"
            "- Full references section"
        ),
        agent=agent,
    )
