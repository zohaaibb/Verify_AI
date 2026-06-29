# Verify_AI (TruthLens) — AI Agent Guide

## Quick Overview
Verify_AI is a Flask-based disinformation analysis platform. A user submits text, a URL, or an image; the system runs it through 8 parallel analysis modules (AI detection, OSINT, VirusTotal, forensics, etc.), then calculates a final **trust score (0-100)**.

## Session Context (June 11, 2026)

### Latest: Error Card Fallback + Always-Editable OCR
- All 9 modules in `mapBackendReport.ts` now have `else if (module)` error branches — failed modules show as `status: "Failed"` entries instead of being silently skipped
- `Report.tsx` `ModuleCard` renders a red error card (`AlertTriangle` + `XCircle` + error message) when `m.status === "Failed"` or `(m as any).error` is set
- `EditableOcrCard` always shows an editable `<Textarea>` + "Save & Re-analyze" — no Edit/Cancel toggle
- Backend runs on port 5000 (`python app/main.py` from `venv/Scripts/python.exe`), frontend on port 5173 (`npm run dev` from `sparky-react-vite/`)

### Research Paper vs Codebase Comparison (June 5, 2026)
A research paper titled "TruthLens: A CPU-Efficient Multi-Modal Fake News Detection Framework" was compared against this codebase. Key findings:

**Paper is ~90% accurate.** Major discrepancies found:

1. **Wikidata integration claimed but NOT implemented** — `trusted_sources.py` docstring says "Google News + Wikidata" but only Google News RSS exists. Remove from paper or implement.

2. **Evaluation numbers unverified** — Paper claims 88% accuracy, 91% precision, 85% F1 on FAKE, 12% SUSPICIOUS from "100 diverse claims" but NO evaluation scripts exist (`tests/` is empty). The paper describes this as "Text module evaluation" (the 3-model NLP ensemble, not the full orchestrator).

3. **"65+ trusted sources" is inflated** — code has ~45-50 unique trusted/referenced domains, not 65+.

4. **Minor weight mismatches** — Text model weights (paper: 0.35/0.35/0.30 vs code: 0.35/0.40/0.25). Deepfake weights (paper: 3/3/4 vs code: 1.2/0.9/1.0).

5. **CPU-only claim is inaccurate** — Code auto-detects GPU and uses it when available. Not restricted to CPU.

6. **Deepfake Platt scaling claimed but not implemented** — only text module has Platt scaling; deepfake ensemble uses softmax + weighted average.

### Evaluation Script Created
File: `tests/evaluate_text_pipeline.py`
- 100 claims (50 FAKE + 50 REAL) with ground truth
- Tests full orchestrator pipeline via POST /submit with modules: [ai, osint, trusted, scraper, wikipedia]
- Requires backend running, outputs confusion matrix + metrics
- User will run and share results

### TruthLens = Verify_AI
The paper uses the name "TruthLens" but refers to this same codebase. Naming difference is intentional and known.

---

## Architecture

```
User → React Frontend → Flask API (port 5000) → Orchestrator → Parallel Modules → Scoring Engine → JSON Response
```

- **Backend:** Flask 3.x, SQLAlchemy (SQLite), JWT auth, CORS
- **Frontend (active):** `truthlens-truth-detector/` — React+TypeScript+Vite+Tailwind (NOT wired to backend yet)
- **Frontend (boilerplate):** `frontend/` — Stock Vite template, ignore this
- **Microservice:** `services/roberta_service/` — Standalone Flask on port 5001 for single-model RoBERTa inference

---

## How the AI Modules Work

### 1. AI Detection Module
Handles all 3 input types:

**For images** → `services/deepfake_detector.py` (~1300 lines)
- Extracts faces using MTCNN (OpenCV fallback if no face found)
- Runs 3 HuggingFace models in parallel:
  - `prithivMLmods/Deepfake-Detection-Exp-02-21` (ViT, 98.84%) — weight 3
  - `prithivMLmods/deepfake-detector-model-v1` (SigLIP, 94.44%) — weight 3
  - `Wvolf/ViT_Deepfake_Detection` (ViT, 98.70%) — weight 4
- Platt-scales per-model confidence → weighted average → final score

**For text** → `services/text_processor.py` (~600 lines)
- Loads 3 NLP models via `utils/model_loader.py` (parallel load with 60s timeout):
  - `Arko007/fake-news-roberta-5M` (weight 0.35, **inverted** — REAL = high score)
  - `jy46604790/Fake-News-Bert-Detect` (weight 0.35, normal)
  - `hamzab/roberta-fake-news-classification` (weight 0.30, normal)
- Platt-scales → entropy penalty for model disagreement → sensitive-topic filter (+10 penalty)
- Returns `fake_probability` + `confidence` + flags

**For URLs** → `services/url_ml_detector.py` (~750 lines)
- 3-layer pipeline:
  1. Google Safe Browsing API (if API key present)
  2. `CrabInHoney/urlbert-tiny-v4-*` ML model
  3. Heuristic engine (suspicious TLD list, keyword analysis, URL length checks)
- WHOIS domain age override (domains < 1 year get suspicion boost)
- Returns `is_malicious` + `confidence` + `suspicious_flags`

### 2. OSINT Source Verification → `services/source_osint.py` (~1100 lines)
3-layer scoring for any domain:
- **Infrastructure:** DNS records, SSL cert age, registrar, IP reputation
- **Presence:** Social media links, Wikipedia entry, Alexa rank, Wayback Machine age
- **Reach:** Backlinks, traffic estimates
- **Floor rule:** Any resolvable domain with valid SSL > 30 days gets minimum 50/100

### 3. VirusTotal → `services/virustotal_client.py` (~200 lines)
- Submits URL to VirusTotal API v3, polls for completion
- Returns detection ratio + scan results

### 4. Image Forensics → `services/image_forensics.py` + `image_analysis_pipeline.py`
- Magic byte verification (detects double-extension attacks)
- EXIF metadata extraction (camera, GPS, timestamps)
- Hidden payload scanning (trailing data after valid image end)
- Steganography detection (LSB chi-square + RS analysis)
- OCR via Tesseract (text extraction from images)
- NSFW detection
- SVG sanitization

### 5. Trusted Sources → `services/trusted_sources.py` (~400 lines)
- 65+ hardcoded domains with credibility scores (Reuters 98, BBC 95, Dawn 88, etc.)
- Google News RSS search for matching articles
- Wikidata API fallback for entity lookup

### 6. News Scraper → `services/news_scraper.py` (~600 lines)
5 strategies in order: Google News RSS > NewsAPI > site scrape > DuckDuckGo News > Bing RSS
- TF-IDF + SentenceTransformer hybrid similarity matching

### 7. Web Search → `services/google_search_client.py` (~400 lines)
3-tier fallback: Serper.dev API > Google CSE API > direct SERP scrape (Lynx UA)

---

## Scoring Engine (`orchestrator.py`)

Each module returns a score 0.0–1.0. `calculate_trust_score()` computes:

1. **Weighted average** of all module scores
2. **Penalties** applied based on evidence:
   - High-confidence FAKE → -20 points
   - Very-high-confidence FAKE → -40 points
   - Suspicious keywords → -10
   - Sensitive/dividing topics → -10
   - Newly registered domain → -15
   - Negative OSINT reach → -10
3. **Final trust score:** 0–100
   - >70: Likely real
   - 40–70: Uncertain/requires review
   - <40: Likely fake

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/register` | Register user with plan (free/pro/enterprise) |
| POST | `/api/auth/login` | Login → JWT + plan info |
| GET | `/api/analysis/modules` | List available modules |
| POST | `/api/analysis/submit` | Submit text/URL/image for analysis |
| GET | `/health` | Health check |
| GET | `/` | Root (API running confirmation) |

---

## Database Models

| Model | Table | Key Fields |
|-------|-------|------------|
| User | `users` | id, username, email, password_hash, subscription (free/pro/enterprise), created_at |
| AnalysisRequest | `analysis_requests` | id, input_type, input_data/file_path, selected_modules (JSON), all 8 result columns (JSON), trust_score, status, timestamps |

---

## Configuration

- **`backend/.env`** — All API keys (VirusTotal, Google FactCheck, GNews, NewsData, Serper, Hybrid Analysis, CAPE, ANY.RUN)
- **`backend/app/config.py`** — DevelopmentConfig vs ProductionConfig, 16MB max upload, JWT expiry
- **Plan limits:** Free=5/day, Pro=50/day, Enterprise=unlimited

---

## Known Issues / Gotchas

1. **`requirements.txt` is binary/corrupt** — can't install directly from it
2. **3 stub files are empty:** `cuckoo_client.py`, `sandbox_client.py`, `reverse_engineering.py` — imports commented out in orchestrator
3. **Orchestrator history:** `orchestrator.py` is active; `orchestrator_backup.py` and `orchestrator_new.py` are stale
4. **Graceful degradation** — Every API call wrapped in try/except. Missing/invalid API keys → module skipped, system continues
5. **Deprecated models preserved** in `utils/deepfake_config.py` with `weights=0.0` and `deprecated=True` for backward compatibility
6. **Frontend not wired** — Neither frontend project actually calls the backend API yet
7. **RoBERTa microservice** (`services/roberta_service/`) runs separately on port 5001, not part of main pipeline

---

## Working on This Project

- Run backend: `cd backend && python run.py` or `python app/main.py`
- Run microservice: `cd services/roberta_service && python app.py`
- Never modify `orchestrator_backup.py` or `orchestrator_new.py` — delete them if safe
- Always check `.env` for required API keys before testing modules
- When adding new models, follow the ensemble pattern (weighted voting + Platt scaling + graceful skip)

---

## Session Context (June 7, 2026) — Paper Hyperlinking

### Paper Referencing Complete
Added all links from `E:\Paper\links.docx` (84 papers across 9 module tables) into the TruthLens paper.

**Output file:** `E:\Paper\TruthLens_Final_Hyperlinked.docx` (also `E:\Paper\TruthLens_Complete_Paper_Hyperlinked.docx` from earlier run)

**What was done:**
1. Parsed 84 paper entries from `links.docx` (9 tables across modules: Text, OSINT, URL, Forensics, Web Evidence, Fusion, Deepfake)
2. Mapped **72/75** existing references [1]–[75] to their URLs (3 unmapped — Platt [18], Google Safe Browsing [38], Hashmi [15] — no links in links.docx)
3. Added **10 new references** [76]–[85] with full paper titles fetched from their actual URLs
4. Hyperlinked **99 in-text citations** (e.g., `[1]`, `[1,2]`, `[4-6]`) throughout the paper body
5. **Total: 181 HYPERLINK field codes** (82 in refs + 99 in body)

### Hyperlink Method (WPS-Compatible)
Used OOXML **HYPERLINK field codes** (`w:instrText` with `HYPERLINK "url"`) rather than `w:hyperlink` XML elements. This is compatible with WPS Office, Microsoft Word, and LibreOffice. In WPS, links are followed via **Ctrl+Click** by default (can be changed in WPS Settings → Edit).

### Full Titles for New References [76]–[85]
- [76] Niyaoui & Reda — "URLBERT: A Contrastive and Adversarial Pre-trained Model for URL Classification," AI2SD/Springer 2023
- [77] Shah et al. — "VERITAS-NLI: Validation and Extraction of Reliable Information Through Automated Scraping and Natural Language Inference," arXiv 2024
- [78] Liao et al. — "MUSER: A MUlti-Step Evidence Retrieval Enhancement Framework for Fake News Detection," KDD 2023
- [79] Thejeshwar et al. — "Enhancing Social Media Trust: Identifying Fake News with Domain Reputation and Content Insights," IEEE 2025
- [80] Bharati et al. — "A Novel Fuzzy Logic-Based Hybrid Framework for Detecting Fake News," Discover Computing 2026
- [81] Birthriya et al. — "A Dual-Layer Deep Learning Model for Parallel Analysis of URL and HTML Features in Phishing Website Detection," Knowl. Inf. Syst. 2026
- [82] Meng et al. — "Proactive Image Manipulation Detection and Tracing in Fake News," IEEE Trans. Multimedia 2026
- [83] Wu et al. — "KGV: Integrating Large Language Models With Knowledge Graphs for Cyber Threat Intelligence Credibility Assessment," IEEE TIFS 2026
- [84] Tah et al. — "HybridNet: A Fusion Framework for Multimodal Fake News Detection," CVPR Workshops (PP-MisDet) 2026
- [85] Girón et al. — "LANTERN: Discourse-Driven Detection of Multimodal Misinformation," IEEE Access 2025

### Scripts Created (in `C:\Users\zohai\AppData\Local\Temp\opencode\`)
- `link_paper_refs_v3.py` — Original hyperlinking using `w:hyperlink` elements
- `full_pipeline.py` — Improved version with full title replacement
- `wps_compatible.py` — Final version using field codes (WPS-friendly)
- `extract_paper.py` / `extract_links.py` — Parsing utilities
- `to_pdf.py` — PDF conversion attempt (requires Word/LibreOffice)

### Pending
- **PDF conversion**: Needs Word, LibreOffice, or WPS — neither is readily available on this machine. User should open the DOCX in WPS and do File → Save As → PDF.
- **Paper vs codebase fixes** (from previous session): Trusted sources count, Wikidata implementation, GPU enforcement, eval numbers verification.

---

## Learning Hub (Session Persistence)

**Root:** `E:\Learning_Hub\`

All learning materials, cheatsheets, projects, notes, and session history are stored here.
- `cheatsheets/` — HTML/PDF cheatsheets
- `projects/` — standalone project work
- `notes/` — study notes
- `resources/` — bookmarks, links, references
- `sessions.md` — session log (read this first in future sessions)

**Mandatory:** Read `sessions.md` at the **start** of each session to recall context. Update it at the **end** of each session with what was done, decisions made, and next steps — so I always know where we left off.

### JARVIS Mode (AI Greeting)
At the start of every session, greet the user like JARVIS:
1. **Greeting** — "Good morning/afternoon/evening, [name]"
2. **Current topic** — "You're learning [X]. Ready to continue?"
3. **Agenda** — Read today's items from `sessions.md` "What was done" or check if they have a goal
4. **Tone** — Calm, confident, slightly formal like JARVIS. Keep it brief — they want to work.
5. **Session persistence** — After every chat, update `sessions.md` with what was done, decisions, and next steps so the next session picks up seamlessly.
