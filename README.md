# Resume ↔ JD Evaluator

AI-powered resume analyzer that evaluates how well your resume matches a job description for data/ML roles. Get a 0-100 match score, strengths, gaps, missing keywords, bullet improvements, and prioritized actions.

## Features

- **Semantic Similarity Analysis**: Uses OpenAI embeddings to measure content overlap
- **Rubric-Based Scoring**: GPT-4 powered evaluation with structured feedback
- **Blended Scoring**: Combines embedding similarity (40%) and rubric score (60%)
- **Actionable Insights**: Get specific bullet rewrites, missing keywords, and priority actions
- **Role-Specific Evaluation**: Tailored for Data Scientist, ML Engineer, Data Engineer, and more

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up OpenAI API key:**
```bash
export OPENAI_API_KEY="your-key-here"
```

3. **Run the app:**
```bash
python app.py
```

The app will launch at `http://127.0.0.1:786x` with a public shareable link.

## How It Works

1. **Upload your resume** (PDF format)
2. **Paste the job description** text
3. **Select your target role** from the dropdown
4. **Click Evaluate** to get:
   - Overall match score (0-100)
   - Detailed scoring breakdown
   - Strengths and gaps analysis
   - Missing keywords
   - Bullet-level improvement suggestions
   - Priority actions for the next 48 hours

## Scoring Formula

```
Final Score = (0.4 × Embedding Similarity) + (0.6 × Rubric Score), feel free to change the weighting as needed.
```

The embedding similarity measures semantic overlap between resume and JD content, while the rubric score comes from GPT-4 evaluation with role-specific criteria.

## Supported Roles

- Data Engineer
- Data Scientist
- Data Analyst
- Machine Learning Engineer
- Technical Product Manager
- Cloud/DevOps Engineer