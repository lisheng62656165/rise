# DeepPlanning Travel Partial Paired Comparison

Model: `nvidia/nemotron-3.5-lightning-30b-a3b`

This is a diagnostic comparison over the currently available StateTrace reports,
not the final 120-case full-cohort result. Travel-ZH contains 93 report IDs and
Travel-EN contains 94 report IDs. Vanilla is evaluated on the same report IDs.

## Original partial-cohort denominators

Conversion failures contribute zero to quality metrics, following the official
evaluator's denominator behavior.

| Cohort | Metric | Vanilla | StateTrace-DSR | Delta |
|---|---|---:|---:|---:|
| Travel-ZH (93) | Delivery rate | 84.95% (79/93) | 86.02% (80/93) | +1.08pp |
| Travel-ZH (93) | Commonsense score | 37.10% | 38.17% | +1.08pp |
| Travel-ZH (93) | Personalized score | 25.81% | 25.81% | 0.00pp |
| Travel-ZH (93) | Composite score | 31.45% | 31.99% | +0.54pp |
| Travel-ZH (93) | Case accuracy | 0.00% | 0.00% | 0.00pp |
| Travel-EN (94) | Delivery rate | 94.68% (89/94) | 93.62% (88/94) | -1.06pp |
| Travel-EN (94) | Commonsense score | 45.74% | 45.88% | +0.13pp |
| Travel-EN (94) | Personalized score | 22.34% | 19.15% | -3.19pp |
| Travel-EN (94) | Composite score | 34.04% | 32.51% | -1.53pp |
| Travel-EN (94) | Case accuracy | 0.00% | 0.00% | 0.00pp |

## Common converted IDs

This removes asymmetric conversion coverage. Both methods are evaluated on the
same successfully converted IDs: 79 for ZH and 87 for EN.

| Cohort | Metric | Vanilla | StateTrace-DSR | Delta |
|---|---|---:|---:|---:|
| Travel-ZH (79) | Delivery rate | 100.00% | 100.00% | 0.00pp |
| Travel-ZH (79) | Commonsense score | 43.67% | 44.30% | +0.63pp |
| Travel-ZH (79) | Personalized score | 30.38% | 30.38% | 0.00pp |
| Travel-ZH (79) | Composite score | 37.03% | 37.34% | +0.32pp |
| Travel-ZH (79) | Case accuracy | 0.00% | 0.00% | 0.00pp |
| Travel-EN (87) | Delivery rate | 100.00% | 100.00% | 0.00pp |
| Travel-EN (87) | Commonsense score | 48.56% | 49.14% | +0.57pp |
| Travel-EN (87) | Personalized score | 22.99% | 20.69% | -2.30pp |
| Travel-EN (87) | Composite score | 35.78% | 34.91% | -0.86pp |
| Travel-EN (87) | Case accuracy | 0.00% | 0.00% | 0.00pp |

## Interpretation

- ZH shows a small positive StateTrace-DSR effect.
- EN improves commonsense consistency but loses personalized-constraint accuracy,
  producing a negative composite delta.
- These results cannot be presented as the final DeepPlanning full-test result;
  method generation is still missing 27 ZH and 26 EN reports, and LLM conversion
  remains incomplete.
