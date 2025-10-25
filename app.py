# app_gradio.py
import gradio as gr
from logic import evaluate_resume_vs_jd_text

ROLE_OPTIONS = [
    "Data Engineer",
    "Data Scientist",
    "Data Analyst",
    "Machine Learning Engineer",
    "Technical Product Manager",
    "Cloud/DevOps Engineer"
    ]

def evaluate_ui(role, resume_pdf, jd_text_input):
    if not jd_text_input or resume_pdf is None or role not in ROLE_OPTIONS:
        return gr.update(value=""), gr.update(value="Please provide job description text, select a role, and upload a PDF."), None, None, None, None, None, None

    try:
        result = evaluate_resume_vs_jd_text(jd_text_input.strip(), resume_pdf, role)
    except Exception as e:
        return gr.update(value=""), gr.update(value=str(e)), None, None, None, None, None, None

    # Build readable outputs
    score_text = f"{result['final_score']}/100"
    similarity_text = f"{result['similarity_pct']}"
    rubric_score = int(result["rubric"].get("rubric_score_0_to_100", 0))
    # Calculate the actual blend
    similarity_float = float(similarity_text)
    actual_blend = round(0.4 * similarity_float + 0.6 * rubric_score, 1)
    
    details_md = f"""
**How we scored**
- Embedding similarity proxy: **{similarity_text}** (40% weight)
- Rubric score: **{rubric_score}** (60% weight)
- **Final blended score: {actual_blend}/100**
- Formula: (0.4 × {similarity_text}) + (0.6 × {rubric_score}) = {actual_blend}

**Debug Info**
- Job description text length: {len(result.get('jd_text', ''))} characters
- Word count: {len(result.get('jd_text', '').split())} words
"""

    summary = result["rubric"].get("summary", "")
    strengths = result["rubric"].get("strengths", [])
    gaps = result["rubric"].get("gaps", [])
    missing = result["rubric"].get("must_have_keywords_missing", [])
    bullets = result["rubric"].get("recommended_bullet_improvements", [])
    actions = result["rubric"].get("priority_actions_in_next_48h", [])

    strengths_md = "\n".join([f"- {s}" for s in strengths]) or "None listed."
    gaps_md = "\n".join([f"- {g}" for g in gaps]) or "None listed."
    missing_md = ", ".join(sorted(set(missing))) if missing else "None flagged."
    bullets_md = "\n\n".join(
        [f"**Current**\n```\n{b.get('current','N/A')}\n```\n**Improved**\n> {b.get('improved','')}" for b in bullets]
    ) or "No specific bullet rewrites provided."
    actions_md = "\n".join([f"- {a}" for a in actions]) or "None."

    jd_text = result.get('jd_text', 'No JD text available')
    
    return (
        score_text,
        "",
        details_md,
        summary,
        strengths_md,
        gaps_md,
        f"**Missing or weak keywords**\n\n{missing_md}\n\n**Bullet-level improvements**\n\n{bullets_md}\n\n**Priority actions in the next 48 hours**\n\n{actions_md}",
    )

with gr.Blocks(title="Resume ↔ JD Evaluator") as demo:
    gr.Markdown("# Resume ↔ JD Evaluator")
    gr.Markdown("Specialized evaluator with a role dropdown. Paste job description text, upload your resume PDF, and get a match score and actionable improvements.")

    with gr.Row():
        role = gr.Dropdown(choices=ROLE_OPTIONS, value="Machine Learning Engineer", label="Role")
    
    jd_text_input = gr.Textbox(
        label="Job Description Text", 
        placeholder="Paste the job description text here...",
        lines=8,
        max_lines=15
    )

    resume_pdf = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])

    btn = gr.Button("Evaluate", variant="primary")

    with gr.Row():
        score = gr.Number(label="Overall Match 0-100", interactive=False)
        error = gr.Textbox(label="Errors", interactive=False)

    details = gr.Markdown(label="How this was scored")
    summary = gr.Markdown(label="Evaluator Summary")
    strengths = gr.Markdown(label="Strengths")
    gaps = gr.Markdown(label="Gaps")
    extras = gr.Markdown(label="Keywords, Bullet rewrites, and Actions")

    btn.click(
        evaluate_ui,
        inputs=[role, resume_pdf, jd_text_input],
        outputs=[score, error, details, summary, strengths, gaps, extras],
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)