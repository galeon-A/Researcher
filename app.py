"""
app.py
Multi-Agent Research Paper Writer — Streamlit Application

Pipeline: arXiv → FAISS (Google Embeddings) → CrewAI (4 agents) → Paper
Agents:   Researcher → Summarizer → Writer → Reviewer
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Astra",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Global */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #1a237e 0%, #1565c0 50%, #0288d1 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(21, 101, 192, 0.3);
}
.main-header h1 { color: white; font-size: 2.4rem; font-weight: 700; margin: 0 0 0.5rem; }
.main-header p  { color: rgba(255,255,255,0.85); font-size: 1.05rem; margin: 0; }

/* Agent pipeline display */
.pipeline-container {
    display: flex; align-items: center; justify-content: center;
    gap: 0; margin: 1.5rem 0; flex-wrap: wrap;
}
.agent-box {
    background: white; border: 2px solid #e3f2fd;
    border-radius: 12px; padding: 0.9rem 1.2rem;
    text-align: center; min-width: 120px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: all 0.3s;
}
.agent-box.active  { border-color: #1565c0; background: #e3f2fd; box-shadow: 0 4px 16px rgba(21,101,192,0.2); }
.agent-box.done    { border-color: #2e7d32; background: #e8f5e9; }
.agent-icon  { font-size: 1.8rem; }
.agent-label { font-size: 0.78rem; font-weight: 600; color: #37474f; margin-top: 4px; }
.arrow       { font-size: 1.4rem; color: #90a4ae; padding: 0 0.3rem; }

/* Cards */
.info-card {
    background: #f8f9ff; border: 1px solid #e3f2fd;
    border-radius: 12px; padding: 1.2rem 1.5rem; margin: 1rem 0;
}
.info-card h4 { color: #1565c0; margin: 0 0 0.5rem; font-size: 1rem; }
.info-card p  { color: #455a64; margin: 0; font-size: 0.9rem; line-height: 1.6; }

/* Paper output */
.paper-output {
    background: white; border: 1px solid #e0e0e0;
    border-radius: 12px; padding: 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    max-height: 700px; overflow-y: auto;
}

/* Reference chips */
.ref-chip {
    display: inline-block; background: #e3f2fd;
    border: 1px solid #90caf9; border-radius: 20px;
    padding: 4px 12px; margin: 3px;
    font-size: 0.8rem; color: #1565c0;
}

/* Metric boxes */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 120px;
    background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
    border-radius: 10px; padding: 1rem; text-align: center;
    border: 1px solid #e0e0e0;
}
.metric-val  { font-size: 1.8rem; font-weight: 700; color: #1565c0; }
.metric-label { font-size: 0.75rem; color: #607d8b; margin-top: 2px; }

/* Review report */
.review-box {
    background: #fffde7; border-left: 4px solid #f9a825;
    border-radius: 0 10px 10px 0; padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #f5f7ff; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="main-header">
    <h1>📄 Astra - Research Paper Writer</h1>
    <p>Powered by CrewAI · Google Gemini · LangChain · FAISS · arXiv</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Agent pipeline visual ──────────────────────────────────────────────────────
st.markdown(
    """
<div class="pipeline-container">
  <div class="agent-box">
    <div class="agent-icon">🔍</div>
    <div class="agent-label">Researcher</div>
  </div>
  <div class="arrow">→</div>
  <div class="agent-box">
    <div class="agent-icon">📝</div>
    <div class="agent-label">Summarizer</div>
  </div>
  <div class="arrow">→</div>
  <div class="agent-box">
    <div class="agent-icon">✍️</div>
    <div class="agent-label">Writer</div>
  </div>
  <div class="arrow">→</div>
  <div class="agent-box">
    <div class="agent-icon">🔎</div>
    <div class="agent-label">Reviewer</div>
  </div>
  <div class="arrow">→</div>
  <div class="agent-box">
    <div class="agent-icon">📄</div>
    <div class="agent-label">Paper + PDF</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # API Key
    api_key_input = st.text_input(
        "Google API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Get your key at https://aistudio.google.com/app/apikey",
        placeholder="AIza...",
    )

    st.markdown("---")

    # Model selection
    model_choice = st.selectbox(
        "Gemini Model",
        options=[
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        index=0,
        help="gemini-2.0-flash is recommended for speed & quality",
    )

    # Number of papers
    num_papers = st.slider(
        "Papers to fetch from arXiv",
        min_value=5,
        max_value=25,
        value=10,
        step=1,
        help="More papers = richer content but slower indexing",
    )

    # Cache toggle
    use_cache = st.checkbox(
        "Reuse cached FAISS index",
        value=True,
        help="Skip re-embedding if you already searched this topic",
    )

    st.markdown("---")

    # Architecture info
    st.markdown("### 🏗️ Architecture")
    st.markdown(
        """
        | Component | Technology |
        |-----------|-----------|
        | Agents    | CrewAI    |
        | LLM       | Gemini    |
        | Embeddings| Google    |
        | Vector DB | FAISS     |
        | Knowledge | arXiv     |
        | UI        | Streamlit |
        """
    )

    st.markdown("---")
    st.markdown("### 📌 Tips")
    st.markdown(
        """
        - Be specific with your topic
        - e.g. *"transformer attention mechanisms"*
        - e.g. *"CRISPR gene editing cancer"*
        - e.g. *"federated learning privacy"*
        - Generation takes **2–5 minutes**
        """
    )

# ── Main input area ────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Large Language Models for code generation",
        help="Enter any academic topic — the agents will search arXiv and write a paper",
        label_visibility="visible",
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button(
        "🚀 Generate Paper",
        use_container_width=True,
        type="primary",
    )

# ── Validation ────────────────────────────────────────────────────────────────
if generate_btn:
    if not api_key_input:
        st.error("⚠️ Please enter your Google API Key in the sidebar.")
        st.stop()
    if not topic or len(topic.strip()) < 5:
        st.error("⚠️ Please enter a research topic (at least 5 characters).")
        st.stop()

    # ── Import pipeline (lazy to avoid slow imports on startup) ───────────
    with st.spinner("Loading pipeline modules..."):
        try:
            from utils.pipeline import ResearchPipeline
            from utils.pdf_exporter import markdown_to_pdf
        except ImportError as e:
            st.error(f"Import error: {e}\n\nMake sure you've run: `pip install -r requirements.txt`")
            st.stop()

    # ── Progress tracking ──────────────────────────────────────────────────
    progress_bar = st.progress(0)
    status_text  = st.empty()
    stage_cols   = st.columns(4)
    stage_status = [col.empty() for col in stage_cols]

    def update_progress(message: str, percent: int):
        progress_bar.progress(percent / 100)
        status_text.markdown(f"**{message}**")

        # Update stage indicators
        if percent < 45:
            stage_status[0].markdown("🔵 **Researcher** — running...")
            stage_status[1].markdown("⚪ Summarizer")
            stage_status[2].markdown("⚪ Writer")
            stage_status[3].markdown("⚪ Reviewer")
        elif percent < 65:
            stage_status[0].markdown("✅ **Researcher** — done")
            stage_status[1].markdown("🔵 **Summarizer** — running...")
            stage_status[2].markdown("⚪ Writer")
            stage_status[3].markdown("⚪ Reviewer")
        elif percent < 85:
            stage_status[0].markdown("✅ **Researcher** — done")
            stage_status[1].markdown("✅ **Summarizer** — done")
            stage_status[2].markdown("🔵 **Writer** — running...")
            stage_status[3].markdown("⚪ Reviewer")
        elif percent < 100:
            stage_status[0].markdown("✅ **Researcher** — done")
            stage_status[1].markdown("✅ **Summarizer** — done")
            stage_status[2].markdown("✅ **Writer** — done")
            stage_status[3].markdown("🔵 **Reviewer** — running...")
        else:
            for s in stage_status:
                s.markdown("✅ Done")

    # ── Run pipeline ───────────────────────────────────────────────────────
    start_time = time.time()
    try:
        pipeline = ResearchPipeline(api_key=api_key_input, model=model_choice)
        result = pipeline.run(
            topic=topic.strip(),
            num_papers=num_papers,
            use_cache=use_cache,
            progress_callback=update_progress,
        )
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Pipeline failed: {str(e)}")
        st.exception(e)
        st.stop()

    elapsed = time.time() - start_time
    progress_bar.progress(1.0)
    status_text.success(f"✅ Paper generated in {elapsed:.0f} seconds!")

    # ── Store result in session ────────────────────────────────────────────
    st.session_state["result"] = result
    st.session_state["topic"] = topic

# ── Display results ────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    paper  = result.get("paper", "")
    topic  = result.get("topic", "Research Paper")

    st.markdown("---")

    # Metrics row
    word_count = len(paper.split())
    ref_count  = len(result.get("references", []))
    papers_n   = result.get("papers_fetched", 0)

    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-box">
            <div class="metric-val">{word_count:,}</div>
            <div class="metric-label">Words Generated</div>
          </div>
          <div class="metric-box">
            <div class="metric-val">{papers_n}</div>
            <div class="metric-label">arXiv Papers Fetched</div>
          </div>
          <div class="metric-box">
            <div class="metric-val">{ref_count}</div>
            <div class="metric-label">Unique References</div>
          </div>
          <div class="metric-box">
            <div class="metric-val">4</div>
            <div class="metric-label">AI Agents Used</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs
    tab_paper, tab_review, tab_refs, tab_download = st.tabs(
        ["📄 Research Paper", "🔍 Review Report", "📚 References", "⬇️ Download"]
    )

    # ── Tab 1: Paper ──────────────────────────────────────────────────────
    with tab_paper:
        st.markdown(
            '<div class="paper-output">', unsafe_allow_html=True
        )
        st.markdown(paper)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: Review Report ──────────────────────────────────────────────
    with tab_review:
        review = result.get("review_report", "No review report available.")
        st.markdown(
            f'<div class="review-box">{review}</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 3: References ─────────────────────────────────────────────────
    with tab_refs:
        refs = result.get("references", [])
        if refs:
            st.markdown(f"**{len(refs)} papers retrieved from arXiv:**")
            for i, ref in enumerate(refs, 1):
                with st.expander(
                    f"[{i}] {ref.get('title', 'Unknown title')[:90]}"
                ):
                    st.markdown(f"**Authors:** {ref.get('authors', 'N/A')}")
                    st.markdown(f"**Published:** {ref.get('published', 'N/A')}")
                    st.markdown(f"**arXiv ID:** `{ref.get('arxiv_id', 'N/A')}`")
                    if ref.get("url"):
                        st.markdown(f"**URL:** [{ref['url']}]({ref['url']})")
        else:
            st.info("Reference details not available.")

    # ── Tab 4: Download ───────────────────────────────────────────────────
    with tab_download:
        st.markdown("### Download Your Paper")

        col_md, col_pdf = st.columns(2)

        # Markdown download
        with col_md:
            st.markdown(
                """
                <div class="info-card">
                <h4>📝 Markdown (.md)</h4>
                <p>Raw markdown format — great for Notion, Obsidian, GitHub, or further editing.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                label="⬇️ Download Markdown",
                data=paper.encode("utf-8"),
                file_name=f"research_paper_{topic[:30].replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # PDF download
        with col_pdf:
            st.markdown(
                """
                <div class="info-card">
                <h4>📄 PDF Document</h4>
                <p>Formatted PDF with styled headings, proper layout, and page numbers.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("🔨 Generate PDF", use_container_width=True, type="secondary"):
                with st.spinner("Generating PDF..."):
                    try:
                        from utils.pdf_exporter import markdown_to_pdf
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            suffix=".pdf", delete=False
                        ) as tmp:
                            pdf_path = markdown_to_pdf(paper, topic, tmp.name)

                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()

                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name=f"research_paper_{topic[:30].replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.success("✅ PDF ready!")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")

# ── Empty state ────────────────────────────────────────────────────────────────
else:
    st.markdown(
        """
        <div class="info-card">
        <h4>🚀 How it works</h4>
        <p>
        1. Enter your Google API key in the sidebar<br>
        2. Type any research topic<br>
        3. Click <strong>Generate Paper</strong><br>
        4. 4 AI agents collaborate: Researcher → Summarizer → Writer → Reviewer<br>
        5. Download your paper as Markdown or PDF
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    examples = [
        ("🤖 AI / ML", "Retrieval-Augmented Generation for LLMs"),
        ("🧬 Biology", "CRISPR-Cas9 therapeutic applications"),
        ("🔐 Security", "Federated learning and differential privacy"),
    ]
    for col, (icon_label, example_topic) in zip([col1, col2, col3], examples):
        with col:
            st.markdown(
                f"""
                <div class="info-card" style="cursor:pointer; text-align:center;">
                <h4>{icon_label}</h4>
                <p><em>"{example_topic}"</em></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
