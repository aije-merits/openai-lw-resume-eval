# logic.py
import os
import io
import json
import pdfplumber
import numpy as np

OPENAI_MODEL_RUBRIC = os.getenv("OPENAI_MODEL_RUBRIC", "gpt-4o-mini")
OPENAI_MODEL_EMBED = os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-large")

try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None


def extract_pdf_text(file_like) -> str:
    """Extract text from an uploaded PDF file-like object."""
    # Handle different file object types from Gradio
    if hasattr(file_like, 'read'):
        # Standard file-like object
        raw = file_like.read()
    elif hasattr(file_like, 'name'):
        # Gradio file object - read from the file path
        with open(file_like.name, 'rb') as f:
            raw = f.read()
    else:
        # Try to read directly if it's already bytes
        raw = file_like
    
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages)


def _embed_texts(texts):
    if not client:
        raise RuntimeError("OpenAI client not available. Check SDK and OPENAI_API_KEY.")
    if not texts:
        return []
    resp = client.embeddings.create(model=OPENAI_MODEL_EMBED, input=texts)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def chunk_text(text: str, max_words: int = 1200):
    words = text.split()
    out, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) >= max_words:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def semantic_overlap(jd_text: str, resume_text: str) -> float:
    jd_chunks = chunk_text(jd_text, 1200)
    rs_chunks = chunk_text(resume_text, 1200)
    jd_vecs = _embed_texts(jd_chunks)
    rs_vecs = _embed_texts(rs_chunks)
    if not jd_vecs or not rs_vecs:
        return 0.0
    a_best = [max((_cosine(a, b) for b in rs_vecs), default=0.0) for a in jd_vecs]
    b_best = [max((_cosine(b, a) for a in jd_vecs), default=0.0) for b in rs_vecs]
    return float((np.mean(a_best) + np.mean(b_best)) / 2.0)


def rubric_evaluate(role: str, jd_text: str, resume_text: str) -> dict:
    if not client:
        raise RuntimeError("OpenAI client not available. Check SDK and OPENAI_API_KEY.")

    system = f"""You are a precise evaluator for hiring.
You only evaluate for the role: {role}.
Be strict, transparent, and actionable. Output VALID JSON only."""
    user = f"""
Evaluate candidate RESUME against JOB DESCRIPTION for role "{role}".

JOB DESCRIPTION:
\"\"\"{jd_text[:12000]}\"\"\"

RESUME:
\"\"\"{resume_text[:12000]}\"\"\"

Return JSON with this schema:
{{
  "role": "{role}",
  "rubric_score_0_to_100": <integer>,
  "summary": "<2-3 sentence overview of fit>",
  "strengths": ["..."],
  "gaps": ["..."],
  "must_have_keywords_missing": ["..."],
  "recommended_bullet_improvements": [
    {{
      "current": "<quoted line from resume, or 'N/A'>",
      "improved": "<rewrite that is quantified, impact-centered, and JD-aligned>"
    }}
  ],
  "priority_actions_in_next_48h": ["<concise tasks>"]
}}
Rules:
- Keep "rubric_score_0_to_100" an integer.
- If resume is generic or role-mismatched, reflect that in gaps and lower score.
- Prefer measurable outcomes.
"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL_RUBRIC,
        temperature=0.2,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}]
    )
    raw = resp.choices[0].message.content.strip()

    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        data = json.loads(raw)
    except Exception:
        data = {
            "role": role,
            "rubric_score_0_to_100": 0,
            "summary": "Could not parse evaluator output.",
            "strengths": [],
            "gaps": ["Parsing failure, please try again."],
            "must_have_keywords_missing": [],
            "recommended_bullet_improvements": [],
            "priority_actions_in_next_48h": []
        }
    return data


def blended_score(similarity_0_to_1: float, rubric_0_to_100: int) -> int:
    sim_pct = max(0.0, min(1.0, similarity_0_to_1)) * 100.0
    score = 0.4 * sim_pct + 0.6 * float(rubric_0_to_100)
    return int(round(max(0.0, min(100.0, score))))

def evaluate_resume_vs_jd_text(jd_text: str, resume_file_like, role: str) -> dict:
    """
    Evaluate resume against job description text directly (no URL fetching).
    """
    if not jd_text or len(jd_text.split()) < 50:
        raise ValueError("Job description text too short. Please provide at least 50 words.")

    resume_text = extract_pdf_text(resume_file_like)
    if not resume_text or len(resume_text.split()) < 50:
        raise ValueError("Resume text too short or unreadable from PDF.")

    similarity = semantic_overlap(jd_text, resume_text)
    rubric = rubric_evaluate(role, jd_text, resume_text)
    final = blended_score(similarity, int(rubric.get("rubric_score_0_to_100", 0)))

    return {
        "role": role,
        "similarity_pct": round(max(0.0, min(1.0, similarity)) * 100.0, 1),
        "rubric": rubric,
        "final_score": final,
        "jd_text": jd_text,  # Add JD text for debugging
    }