---
license: apache-2.0
model_name: "General Fake News Detector (5M)"
pretty_name: "General Fake News Detector (RoBERTa-base fine-tuned on 5M)"
model_id: "Arko007/fake-news-roberta-5M"
tags:
  - roberta
  - text-classification
  - news-verification
  - fake-news
  - misinformation
  - english
  - pytorch
  - transformers
language: en
library_name: "transformers"
license_review_notes: "Model released under Apache-2.0. Upstream datasets (ISOT, Kaggle datasets, etc.) may have their own licenses; verify rights for redistribution and attribution as needed."
---

# General Fake News Detector (5M)

Short description
A RoBERTa-base model adapted and fine-tuned on a large curated dataset (~5M samples) of news and article-level examples for binary fake vs real classification. Built for long-form news verification and to serve as the domain-adaptive foundation for downstream models (e.g., LIAR political fact-checker).

Model repository: Arko007/fake-news-roberta-5M

## Model snapshot / overview

- Base model: RoBERTa-base (125M parameters)
- Task: Binary classification — FAKE vs REAL (article-level)
- Domain: News articles, long-form content, broad topical coverage (news, science, health)
- Primary use-cases: news verification, content moderation, fact-checking pipelines

## Key performance (reported / validation)

- Validation accuracy (checkpoint-14000): 99.28%
- Validation F1: 99.30%
- Expected final validation: ~99%+

Notes:
- Validation distribution highly imbalanced (FAKE ~2.8%, REAL ~97.2%).
- High scores are consistent with prior models trained on ISOT-style datasets; confirm generalization on stronger OOD tests.

## Training & fine-tuning pipeline

Two-stage approach:
1. RoBERTa pretraining (official)
2. Fine-tuned on ~5M curated news samples (mixture of ISOT, Kaggle corpora, and other public sources)

Training samples & splits:
- Training: ~4.5M samples (ongoing training run reported)
- Validation: held-out set (class distribution: FAKE 2.8%, REAL 97.2%)
- Checkpoint example: step 14,000 (validation acc 99.28%)

Training hyperparameters (example run):
- Optimizer: AdamW
- Learning rate: 2e-5
- Batch size: 8
- Total steps: ~94,466 (1 epoch)
- Training precision: BF16 mixed precision
- Hardware: NVIDIA L4 (24 GB VRAM)
- Time: ~24 hours for one full epoch

Notes:
- Gradient accumulation used to control effective batch size.
- Cosine scheduling and class-weighted loss were used to address imbalance.
- Auto-upload checkpoints to Hugging Face used in the pipeline.

## Data sources & provenance

Primary sources include (non-exhaustive):
- ISOT Fake News Dataset (major component)
- Multiple Kaggle fake-news datasets
- Other public news/fake-news corpora
- Curated scraping / aggregation to reach ~5M samples

Data caveats:
- Large class imbalance (dominant REAL class).
- ISOT-like datasets are known to be relatively easier (may inflate metrics).
- Ensure legal review before redistribution of scraped or 3rd-party content.

## Evaluation & generalization

- The model transfers to downstream LIAR task (after fine-tuning), yielding LIAR final test performance ~71% when used as a starting point.
- High validation metrics on in-domain news; evaluate thoroughly on OOD, short-statement tasks, and tougher fact-check benchmarks before deployment.

## Advantages & limitations

Advantages:
- Works on full news articles (long context)
- Strong validation accuracy on in-domain distribution
- Good foundation for domain-adaptive transfer learning

Limitations:
- Class imbalance: FAKE is under-represented
- Primary sources include ISOT, which can be "easier" than real-world adversarial fake news
- May perform poorly on short, context-free claims (use LIAR model for short statements)
- English-only

## Intended uses & cautions

Appropriate uses:
- Research and large-scale content triage for news sites
- Feature in human-in-loop fact-checking pipelines
- Starting checkpoint for specialized fine-tuning (political, medical, scientific)

Cautions:
- Avoid using as the only signal for content takedown or high-stakes decisions
- Evaluate and mitigate bias (topic / publisher skew)
- Respect dataset licenses and web-source policies

## Usage example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_id = "Arko007/fake-news-roberta-5M"  # replace with HF repo id when uploaded
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)  # device=-1 for CPU

article = """<paste long news article text here>"""
result = clf(article, truncation=True, max_length=4096)
print(result)
```

## Reproducibility & checkpoints

Training scripts in the repository contain:
- preprocessing steps (tokenization, article truncation)
- class-weighting code
- checkpointing & auto-upload logic
- hyperparameter and schedule definitions

## Citation

Suggested model citation:
```
@misc{fake-news-5m-2025,
  title = {General Fake News Detector (5M)},
  author = {Arko007},
  year = {2025},
  howpublished = {Hugging Face model hub: Arko007/fake-news-roberta-5M},
  note = {RoBERTa-base fine-tuned on a 5M-sample curated news corpus}
}
```

Also cite ISOT dataset and other upstream sources per their instructions.

## Contact & maintainer

Maintainer: Arko007 (https://huggingface.co/Arko007) 
Repository: https://github.com/Arko007/fake-news-roberta-5M

If you find licensing issues, provenance problems, or unsafe outputs, please open an issue in the repo.
