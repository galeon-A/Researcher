"""
utils/pipeline.py
Orchestrates the full multi-agent research paper generation pipeline:
  Step 1 – Researcher fetches arXiv papers
  Step 2 – Papers indexed into FAISS (Google embeddings)
  Step 3 – Summarizer synthesizes with RAG
  Step 4 – Writer drafts the full paper
  Step 5 – Reviewer refines and finalizes
"""

import os
from typing import Callable, Optional
from crewai import Crew, Process

from agents.agents import (
    build_llm,
    create_researcher_agent,
    create_summarizer_agent,
    create_writer_agent,
    create_reviewer_agent,
)
from agents.tasks import (
    create_research_task,
    create_summarization_task,
    create_writing_task,
    create_review_task,
)
from tools.arxiv_tool import ArxivSearchTool, fetch_papers_as_documents
from vectorstore.faiss_manager import FAISSManager


class ResearchPipeline:
    """
    Full multi-agent pipeline for generating research papers.
    
    Workflow:
    1. Researcher agent queries arXiv
    2. Documents indexed into FAISS with Google embeddings
    3. Summarizer uses RAG to synthesize literature
    4. Writer drafts the complete paper
    5. Reviewer polishes and finalizes
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        os.environ["GOOGLE_API_KEY"] = api_key

        # Build shared LLM
        self.llm = build_llm(api_key, model)

        # FAISS manager
        self.faiss_manager = FAISSManager(api_key)

        # arXiv tool
        self.arxiv_tool = ArxivSearchTool()

        # Build agents
        self.researcher = create_researcher_agent(self.llm, self.arxiv_tool)
        self.summarizer = create_summarizer_agent(self.llm)
        self.writer = create_writer_agent(self.llm)
        self.reviewer = create_reviewer_agent(self.llm)

    def run(
        self,
        topic: str,
        num_papers: int = 10,
        use_cache: bool = True,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> dict:
        """
        Run the full pipeline for a given topic.
        
        Args:
            topic: Research topic to write about
            num_papers: Number of arXiv papers to fetch
            use_cache: Whether to reuse existing FAISS index for this topic
            progress_callback: Optional fn(message, percent) for progress updates
        
        Returns:
            dict with keys: paper, review_report, references, papers_fetched
        """

        def update(msg: str, pct: int):
            if progress_callback:
                progress_callback(msg, pct)

        update("🔍 Step 1/4 — Researcher fetching arXiv papers...", 5)

        # ─── Step 1: Fetch papers from arXiv ──────────────────────────────
        update("📥 Fetching papers from arXiv...", 10)
        raw_docs = fetch_papers_as_documents(topic, max_results=num_papers)

        if not raw_docs:
            raise ValueError(f"No papers found on arXiv for topic: '{topic}'")

        papers_fetched = len(raw_docs)
        update(f"✅ Fetched {papers_fetched} papers from arXiv", 20)

        # ─── Step 2: Build / load FAISS index ─────────────────────────────
        update("🗂️ Step 2/4 — Indexing papers into FAISS vector store...", 25)

        cache_loaded = False
        if use_cache:
            cache_loaded = self.faiss_manager.load(topic)
            if cache_loaded:
                update("⚡ Loaded existing FAISS index from cache", 35)

        if not cache_loaded:
            update("🔨 Building FAISS index with Google embeddings...", 30)
            self.faiss_manager.build_from_documents(raw_docs, topic)
            update("✅ FAISS index built and saved to disk", 40)

        # ─── Step 3: RAG context for agents ───────────────────────────────
        update("📚 Retrieving context from vector store...", 42)
        context_summary = self.faiss_manager.format_context(
            f"summary and key findings of {topic}", k=8
        )
        context_methods = self.faiss_manager.format_context(
            f"methodology approaches for {topic}", k=6
        )
        context_combined = context_summary + "\n\n" + context_methods

        references = self.faiss_manager.get_all_references()
        update(f"✅ Retrieved context from {len(references)} unique papers", 45)

        # ─── Step 4: Run Research Agent ────────────────────────────────────
        update("🤖 Step 3/4 — Agents writing your paper (this takes a few minutes)...", 50)
        update("   🔎 Researcher agent gathering paper details...", 52)

        research_task = create_research_task(self.researcher, topic, num_papers)
        research_crew = Crew(
            agents=[self.researcher],
            tasks=[research_task],
            process=Process.sequential,
            verbose=False,
        )
        research_result = research_crew.kickoff()
        research_output = str(research_result)
        update("   ✅ Researcher done", 60)

        # ─── Step 5: Run Summarizer Agent ─────────────────────────────────
        update("   📝 Summarizer agent synthesizing literature...", 62)

        summarization_task = create_summarization_task(
            self.summarizer, topic, context_combined
        )
        summary_crew = Crew(
            agents=[self.summarizer],
            tasks=[summarization_task],
            process=Process.sequential,
            verbose=False,
        )
        # Inject research output as context
        summarization_task.context = [research_task]
        summary_result = Crew(
            agents=[self.summarizer],
            tasks=[summarization_task],
            process=Process.sequential,
            verbose=False,
        ).kickoff(inputs={"research_output": research_output})

        summary_output = str(summary_result)
        update("   ✅ Summarizer done", 72)

        # ─── Step 6: Run Writer Agent ──────────────────────────────────────
        update("   ✍️ Writer agent drafting the full paper...", 74)

        # Enrich writer with both research and summary context
        full_context = (
            f"RESEARCH FINDINGS:\n{research_output[:2000]}\n\n"
            f"LITERATURE SYNTHESIS:\n{summary_output[:2000]}\n\n"
            f"RAG CONTEXT FROM PAPERS:\n{context_combined[:2000]}"
        )

        writing_task = create_writing_task(self.writer, topic, full_context)
        write_crew = Crew(
            agents=[self.writer],
            tasks=[writing_task],
            process=Process.sequential,
            verbose=False,
        )
        write_result = write_crew.kickoff(
            inputs={
                "research_output": research_output,
                "summary_output": summary_output,
                "context": context_combined,
            }
        )
        paper_draft = str(write_result)
        update("   ✅ Writer done", 85)

        # ─── Step 7: Run Reviewer Agent ────────────────────────────────────
        update("🔎 Step 4/4 — Reviewer agent polishing the paper...", 87)

        review_task = create_review_task(self.reviewer, topic)
        review_crew = Crew(
            agents=[self.reviewer],
            tasks=[review_task],
            process=Process.sequential,
            verbose=False,
        )
        review_result = review_crew.kickoff(
            inputs={"paper_draft": paper_draft, "topic": topic}
        )
        review_output = str(review_result)
        update("✅ Reviewer done", 95)

        # ─── Parse final output ────────────────────────────────────────────
        final_paper, review_report = self._parse_review_output(
            review_output, paper_draft
        )

        update("🎉 Paper generation complete!", 100)

        return {
            "paper": final_paper,
            "review_report": review_report,
            "references": references,
            "papers_fetched": papers_fetched,
            "topic": topic,
            "model": self.model,
        }

    def _parse_review_output(self, review_output: str, fallback: str) -> tuple[str, str]:
        """
        Split reviewer output into review report and final paper.
        """
        review_report = ""
        final_paper = ""

        # Look for the FINAL PAPER marker
        markers = ["**FINAL PAPER:**", "FINAL PAPER:", "## FINAL PAPER", "# FINAL PAPER"]
        for marker in markers:
            if marker in review_output:
                parts = review_output.split(marker, 1)
                review_report = parts[0].strip()
                final_paper = parts[1].strip()
                return final_paper, review_report

        # Look for REVIEW REPORT marker
        review_markers = ["**REVIEW REPORT:**", "REVIEW REPORT:", "## REVIEW REPORT"]
        for marker in review_markers:
            if marker in review_output:
                parts = review_output.split(marker, 1)
                # Try to find paper after report
                if len(parts) > 1:
                    review_report = parts[1][:800]
                    # Rest might be the paper
                    remaining = review_output[len(review_report):]
                    if len(remaining) > 500:
                        final_paper = remaining
                        return final_paper, review_report

        # If no clear separation, use the whole output as the paper
        # (reviewer may have just returned the improved paper)
        if len(review_output) > len(fallback) * 0.7:
            return review_output, "Review incorporated directly into the final paper."
        else:
            return fallback, review_output
