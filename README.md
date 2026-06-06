# LLM LATAM Content Pipeline

Python pipeline for building and reviewing **prompt–response training samples** focused on Latin America, with rule-based bias checks and quality scoring.

*Portfolio project reflecting LLM trainer work: dataset curation, evaluation criteria, and bilingual technical content (engineering / water / compliance).*

## What it does

1. **Curated pairs** — Hand-written examples (ES/EN) on infrastructure, water, and cultural context in LATAM.
2. **Template expansion** — Fills category templates with regional variables (country, regulation, stakeholders).
3. **Quality validation** — Scores length, technical density, LATAM context, language match, and cultural sensitivity.
4. **Bias detection** — Rule-based keyword checks (gender, regional, elitism, economic patterns).
5. **Evaluation report** — Distribution metrics, quality stats, and recommendations; uses `eval_rubric.json` when present.

> This is a **demonstration pipeline**, not a production LLM training system. Bias detection is heuristic, not model-based.

## Quick start

```bash
git clone https://github.com/EnithV/llm-latam-content-pipeline.git
cd llm-latam-content-pipeline
pip install -r requirements.txt
python content_pipeline.py
python -m unittest discover -s tests -v
```

**Outputs** (gitignored locally):

- `llm_training_dataset.csv`
- `evaluation_report.json`

**Sample without running:** [sample_output.json](sample_output.json)  
**Scoring rubric:** [eval_rubric.json](eval_rubric.json)

## Project structure

```
llm-latam-content-pipeline/
├── content_pipeline.py   # Main pipeline
├── eval_rubric.json      # Evaluation dimensions & thresholds
├── sample_output.json    # Example records & metrics
├── requirements.txt
└── README.md
```

## Content categories

Engineering infrastructure, water management, environmental compliance, project management, technical documentation, regulatory frameworks, sustainability, data analysis, cultural context, business communications.

## Quality & bias (summary)

| Check | Approach |
|-------|----------|
| Quality score | Weighted deductions for short text, low technical density, missing LATAM context |
| Bias | Pattern lists by severity (high / medium) |
| Cultural sensitivity | Positive / negative indicator words |
| Approval threshold | Default ≥ 0.8 (`quality_threshold` in code) |

## Professional context

Built to show skills relevant to **AI evaluation** and **training data QA**:

- Rubric-style scoring and rejection thresholds  
- Bilingual LATAM technical content  
- Bias awareness in dataset design  
- Reporting for dataset review workflows  

Grounded in civil engineering practice (water treatment, rural aqueducts, compliance in Colombia).

## Tests

```bash
python -m unittest discover -s tests -v
```

Covers bias detection, quality scoring, language/culture checks, synthetic generation, and rubric reporting.

## Possible next steps

- [ ] Human review queue (approve / reject / flag)  
- [ ] Export rubric scores per dimension in CSV  
- [ ] More curated pairs per category (less template fill)  
- [ ] Optional integration with external annotator guidelines  

## Contact

**Gicela Vargas**

- Email: ingegvargas@gmail.com  
- LinkedIn: [linkedin.com/in/gicelavargas](https://www.linkedin.com/in/gicelavargas/)  
- Portfolio: [enithv.github.io/portfolio-website](https://enithv.github.io/portfolio-website/)
