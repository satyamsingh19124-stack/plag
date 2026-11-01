# Verity — Plagiarism Checker

A Streamlit plagiarism-checking app based on the workflow in the supplied code:
- PDF, DOCX, and TXT uploads, plus pasted text
- Google search via SerpAPI
- TF-IDF/cosine similarity
- Sentence-transformer semantic similarity (`all-MiniLM-L6-v2`)
- Sentence-level source matches
- Downloadable PDF report

## 1. Install

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Configure SerpAPI

The API key pasted into the chat should be considered exposed. Regenerate it in your SerpAPI account.

Create `.streamlit/secrets.toml`:

```toml
SERPAPI_KEY = "YOUR_NEW_SERPAPI_KEY"
```

Alternatively, set an environment variable named `SERPAPI_KEY`.

Do not commit `secrets.toml` to GitHub. Add it to `.gitignore`.

## 3. Run

```bash
streamlit run app.py
```

The first run downloads the NLTK tokenizer data and the sentence-transformer model.

## Notes about accuracy

This is a similarity-screening tool, not a Turnitin-equivalent database. It searches selected representative sentences through Google/SerpAPI and compares returned snippets. Search engines do not expose the full text of every indexed page, so results can miss matches. The displayed overall score is a heuristic similarity indicator, not a verified percentage of plagiarized text. Always review citations and context manually.

## Security

Uploaded files are processed in the Streamlit session and are not stored in a database by this app. Use HTTPS and appropriate access controls before deploying it publicly.
