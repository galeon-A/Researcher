"""
tools/arxiv_tool.py
Custom LangChain-compatible tool that searches arXiv and returns
paper metadata + abstracts for ingestion into FAISS.
"""

import arxiv
from langchain.tools import BaseTool
from pydantic import Field
from typing import Optional, Type
from pydantic import BaseModel


class ArxivSearchInput(BaseModel):
    query: str = Field(description="The search query for arXiv papers")
    max_results: int = Field(default=10, description="Maximum number of papers to fetch")


class ArxivSearchTool(BaseTool):
    """Search arXiv for research papers and return structured results."""

    name: str = "arxiv_search"
    description: str = (
        "Search arXiv for academic research papers on any topic. "
        "Returns paper titles, authors, abstracts, and arXiv IDs. "
        "Use this to gather source material for research."
    )
    args_schema: Type[BaseModel] = ArxivSearchInput

    def _run(self, query: str, max_results: int = 10) -> str:
        """Search arXiv and return formatted paper information."""
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            papers = []
            for result in client.results(search):
                paper = {
                    "title": result.title,
                    "authors": ", ".join(str(a) for a in result.authors[:5]),
                    "abstract": result.summary[:1500],
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "published": str(result.published.date()),
                    "url": result.entry_id,
                    "categories": ", ".join(result.categories[:3]),
                }
                papers.append(paper)

            if not papers:
                return f"No papers found for query: {query}"

            formatted = []
            for i, p in enumerate(papers, 1):
                formatted.append(
                    f"[{i}] Title: {p['title']}\n"
                    f"    Authors: {p['authors']}\n"
                    f"    Published: {p['published']}\n"
                    f"    ArXiv ID: {p['arxiv_id']}\n"
                    f"    URL: {p['url']}\n"
                    f"    Categories: {p['categories']}\n"
                    f"    Abstract: {p['abstract']}\n"
                    f"    ---"
                )
            return "\n".join(formatted)

        except Exception as e:
            return f"Error searching arXiv: {str(e)}"

    async def _arun(self, query: str, max_results: int = 10) -> str:
        return self._run(query, max_results)


def fetch_papers_as_documents(query: str, max_results: int = 10) -> list[dict]:
    """
    Fetch arXiv papers and return them as a list of dicts
    suitable for embedding into FAISS.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    documents = []
    for result in client.results(search):
        # Combine title + abstract as the chunk content
        content = (
            f"Title: {result.title}\n"
            f"Authors: {', '.join(str(a) for a in result.authors[:5])}\n"
            f"Published: {result.published.date()}\n"
            f"Categories: {', '.join(result.categories[:3])}\n"
            f"Abstract: {result.summary}\n"
        )
        documents.append({
            "content": content,
            "metadata": {
                "title": result.title,
                "authors": ", ".join(str(a) for a in result.authors[:5]),
                "arxiv_id": result.entry_id.split("/")[-1],
                "url": result.entry_id,
                "published": str(result.published.date()),
            },
        })

    return documents
