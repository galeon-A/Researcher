# 📄 Multi-Agent Research Paper Writer

A production-grade AI system that **automatically generates academic research papers** using a multi-agent pipeline powered by **CrewAI**, **Google Gemini**, **LangChain**, **FAISS**, and **arXiv**.

---

## 🏗️ Architecture

```
User Input (Topic)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     arXiv Knowledge Base                        │
│  Fetches top N papers → chunks → Google Embeddings → FAISS     │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────┐   ┌───────────┐   ┌────────┐   ┌──────────┐
│Researcher│ → │Summarizer │ → │ Writer │ → │Reviewer  │
│  Agent   │   │  Agent    │   │ Agent  │   │  Agent   │
└──────────┘   └───────────┘   └────────┘   └──────────┘
      │               │              │             │
  Searches         RAG over       Drafts       Polishes
  arXiv API        FAISS          paper        & refines
                                               
      └──────────────────────────────────────────┘
                          │
                          ▼
              📄 Research Paper (Markdown + PDF)
```

### Tech Stack

| Layer         | Technology                      |
|---------------|---------------------------------|
| Agent Framework | **CrewAI** (4 agents, sequential) |
| LLM           | **Google Gemini 2.0 Flash**     |
| Embeddings    | **Google text-embedding-001**   |
| Vector Store  | **FAISS** (persisted to disk)   |
| Knowledge Base| **arXiv API**                   |
| RAG Framework | **LangChain**                   |
| UI            | **Streamlit**                   |
| PDF Export    | **fpdf2**                       |

---

## 🤖 The 4 Agents

### 1. 🔍 Research Specialist
- **Goal**: Search arXiv for the most relevant papers
- **Tools**: `ArxivSearchTool` (LangChain BaseTool)
- **Output**: Structured list of papers with metadata + abstracts

### 2. 📝 Research Synthesis Expert
- **Goal**: Synthesize papers into themes, findings, and gaps
- **Tools**: RAG over FAISS vector store
- **Output**: Thematic synthesis with key findings

### 3. ✍️ Academic Paper Writer
- **Goal**: Write a complete research paper (2500+ words)
- **Sections**: Abstract, Introduction, Related Work, Background, Analysis, Results, Conclusion, References
- **Output**: Full paper in Markdown

### 4. 🔎 Senior Academic Reviewer
- **Goal**: Review and polish the paper to publication quality
- **Checks**: Logic, evidence, citations, clarity, completeness
- **Output**: Review report + final improved paper

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.10+
- Google API Key ([get one free](https://aistudio.google.com/app/apikey))

### Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd research_writer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
research_writer/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── .streamlit/
│   └── config.toml               # Streamlit theme config
│
├── agents/
│   ├── agents.py                 # 4 CrewAI agent definitions
│   └── tasks.py                  # 4 CrewAI task definitions
│
├── tools/
│   └── arxiv_tool.py             # LangChain arXiv search tool
│
├── vectorstore/
│   ├── faiss_manager.py          # FAISS index manager (Google embeddings)
│   └── data/                     # Persisted FAISS indexes (auto-created)
│
├── utils/
│   ├── pipeline.py               # Main orchestration pipeline
│   └── pdf_exporter.py           # Markdown → PDF converter
│
└── output/                       # Generated papers (auto-created)
```

---

## ⚙️ Configuration

All settings are adjustable in the Streamlit sidebar:

| Setting | Options | Default |
|---------|---------|---------|
| Gemini Model | 2.0-flash, 1.5-flash, 1.5-pro | gemini-2.0-flash |
| Papers to fetch | 5–25 | 10 |
| Cache FAISS index | Yes/No | Yes |

---

## 📊 Output

The system generates:
1. **Complete Research Paper** (Markdown, displayed in UI)
   - Title, Abstract, Introduction, Related Work
   - Background, Analysis, Results, Conclusion
   - Full References with arXiv links
2. **Reviewer's Report** (strengths/improvements)
3. **Reference List** (all fetched arXiv papers)
4. **PDF Download** (styled with headings, page numbers)
5. **Markdown Download** (for editing in Notion/Obsidian)

---

## 💡 Example Topics

- `"Retrieval-Augmented Generation for large language models"`
- `"CRISPR-Cas9 therapeutic applications in cancer treatment"`
- `"Federated learning with differential privacy guarantees"`
- `"Vision transformers vs CNNs for image classification"`
- `"Quantum error correction surface codes"`
- `"Reinforcement learning from human feedback RLHF"`

---

## ⏱️ Performance

| Phase | Time |
|-------|------|
| arXiv fetch (10 papers) | ~5-10s |
| FAISS indexing | ~15-30s |
| Agent execution (4 agents) | ~90-180s |
| PDF generation | ~5s |
| **Total** | **~2-4 minutes** |

---

## 🔒 Notes

- Your API key is used only for Gemini API calls and never stored
- FAISS indexes are cached locally in `vectorstore/data/`
- Papers are fetched live from arXiv at runtime
- All generated content is based on real arXiv papers

---

## 📜 License

MIT License — free to use and modify.
