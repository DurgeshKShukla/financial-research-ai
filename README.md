# Financial Research AI — Day 5 Portfolio MVP

A personal **financial research and analytics** project built in Python. It combines:

- **SEC EDGAR** financial statement ingestion (ticker → CIK → SEC XBRL Company Facts)
- **Python-calculated** annual revenue growth, margins, ROA and ROE
- An **illustrative**, adjustable cash-flow DCF sensitivity model (not a stock-price target)
- Local **PDF RAG**: PyMuPDF → Sentence Transformers → FAISS → Ollama/Qwen
- Follow-up questions, diversified PDF retrieval, conservative numerical guardrails, and optional local **SQLite** chat archiving

## Requirements

- macOS / Python 3.11 Conda environment `finance_ai` used during development
- Ollama installed and `qwen2.5:1.5b` downloaded (`ollama pull qwen2.5:1.5b`)
- Internet to fetch SEC filings and (first time) an embedding model
- **SEC identification**: automated SEC requests must include a genuine app/contact User-Agent, configured privately. No public-facing email field.

## Set up in your existing working folder

1. Back up the working Day 4 files.
2. Copy `app.py`, `finance_analytics.py`, `financial_guardrails.py`, `sec_financials.py`, `history_store.py`, `pdf_retrieval.py`, `test_guardrails.py`, `test_day5.py`, `test_pdf_retrieval.py`, `.gitignore`, and `README.md` into the **same existing project folder**.
3. Keep your working `finance_ai` environment. Dependencies are recorded in `requirements.txt`; **do not reinstall indiscriminately** if your environment already passes checks.
4. Create `.streamlit/secrets.toml` if you haven't already (or keep the existing working one):

   ```toml
   SEC_USER_AGENT = "FinancialResearchAI your-actual-email@domain.com"
   ```

   Use your actual contact address and keep this file **out of Git**. Alternatively use a `SEC_USER_AGENT` environment variable.
5. Launch:

   ```bash
   conda activate finance_ai
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export TOKENIZERS_PARALLELISM=false
   python -m streamlit run app.py --server.port 8503
   ```

   Visit http://localhost:8503. If that port is busy, change it to 8504.

## Day 5 additions

- **Qualitative finance follow-ups:** short questions like “was that good or bad?” inherit the prior metric and use verified Python-calculated values for a conservative assessment.
- **PDF follow-up retrieval (revised):** vague requests like “something more,” “apart from this,” and “what else” trigger topic-diverse searches across financial results, flows, acquisitions, cash, operating expenses, strategy, and risks. Previously surfaced source passages and obvious near-duplicates are excluded; the model sees past answers only as exclusions. A repeated response is suppressed with an explicit limitation message. This does not guarantee that all generated facts are correct or fully novel.
- **Local opt-in SQLite history:** click **Save SEC conversation** or **Save PDF conversation** in the sidebar. No conversations are automatically saved. Database lives at `data/research_history.sqlite3`, excluded from Git. Recent saved-session metadata is listed in the sidebar. Saved sessions are archives, not automatically restored chats.
- **Tests and GitHub hygiene:** deterministic unit tests and a private configuration template.

## Validate before pushing to GitHub

Run from the project folder:

```bash
python -m unittest discover -v
python -m pip check
```

Manual smoke tests:

1. In **SEC + Analytics**, fetch `AAPL`, `MSFT` and `BLK` one at a time. Verify figures against actual SEC filings; check missing data remains missing.
2. Ask “How did operating margin change?” → “How did you calculate that?” → “Was that good or bad?” The last answer should refer only to the same metric.
3. Ask for EBITDA with missing D&A; the system should refuse to fabricate the number.
4. Upload a text-based annual report PDF and ask about revenue or risks, then “What other important findings are there?” Confirm new retrieved excerpts and page citations. Some repeated themes may still occur.
5. Try a DCF discount rate that is less than or equal to terminal growth; model should reject it.
6. Save a session and check that saved metadata appears in the sidebar.

## GitHub

Check that `.streamlit/secrets.toml` and `data/` are not staged:

```bash
git status
git check-ignore .streamlit/secrets.toml data/research_history.sqlite3
```

If the folder is not already a Git repository, initialize one only once:

```bash
git init
git add .
git status
git commit -m "Add financial research AI portfolio MVP"
```

Create an empty GitHub repository, then follow GitHub's displayed instructions to connect and push it. Never commit a real API key, personal email in configuration, private PDFs or SQLite archives.

## Deployment note

**Ollama runs on your Mac.** Publishing a Streamlit frontend alone does not make your local Ollama accessible to a hosted app. This ZIP is a **working local MVP**, not a public hosted demo. To deploy publicly, you must provide a suitable remote inference endpoint or change the app to use a hosted model, and add appropriate security, authentication, request-limit and SEC-access handling. Do not expose an unauthenticated Ollama server to the Internet.

## Known limitations

- SEC XBRL tags and fiscal-year selection differ by company; validation against actual source filings remains required.
- Bank financials need industry-specific accounting conventions and metrics.
- PDF reading order may be inaccurate for multi-column statements and financial tables.
- FAISS semantic similarity does not imply factual correctness; the local 1.5B model can hallucinate.
- DCF is an **educational cash-flow-proxy sensitivity**, not FCFF/FCFE or intrinsic share value.
- SQLite stores saved conversation text locally and is not encrypted. Avoid archiving confidential information.
- No multi-user authentication, production logging, continuous integration or hosted LLM is included.

**CV description (accurate):** Built a Python financial-research prototype integrating SEC EDGAR/XBRL financial data, deterministic ratio analytics, a Streamlit dashboard, and local FAISS/Ollama-based document Q&A with source references and test coverage.


## Day 5 PDF follow-up fix (2026-11-07)

If upgrading an existing Day 5 folder, replace `app.py` and add `pdf_retrieval.py` and `test_pdf_retrieval.py`; other Python modules remain unchanged. Keep your existing `.streamlit/secrets.toml` and local `data/` folder. The bundled `.streamlit/secrets.toml.example` is only a template and must not replace your personal configuration.

Re-run `python -m unittest discover -v` (19 tests). Then in the PDF Research Chat, upload the BlackRock report and ask: (1) “Give me five concrete financial and strategic findings with page numbers”; (2) “Something more, excluding all previous findings.” Look for distinct, supported content and inspect the sources. If the same answer still appears, the model will attempt to flag substantial repetition, but its factual reliability requires human review.
