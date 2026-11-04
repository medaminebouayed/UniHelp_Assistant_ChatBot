# Implementation Plan - Synthetic Data Optimization

Reviewing the `synthetic_chunks.py` audit, we need to shift from "quantity" to "diversity and grounding". The goal is to reach a 60-80% real / 20-40% synthetic ratio with high-quality, non-redundant samples.

## Proposed Changes

### 1. Style & Diversity Balance
We will redistribute `STYLE_WEIGHTS` to reduce query dominance and boost QA pairs.
- **Query**: 0.25 -> 0.15 (It's already very easy to generate, so it over-represents).
- **QA**: 0.30 -> 0.45 (Need to actively favor this style).
- **Reformulate**: 0.25 -> 0.20.

### 2. Redundancy Control (Similarity Thresholds)
Lowering the upper bounds to ensure synthetic data provides "new semantic angles" rather than just "synonym swaps".
- `reformulate`: Upper bound 0.95 -> **0.90**
- `structured`: Upper bound 0.92 -> **0.88**
- `student`: Upper bound 0.88 -> **0.85**
- `qa`: Upper bound 0.90 -> **0.87**

### 3. Boosting QA Pair Quality
Update `PROMPT_QA` to be more descriptive and less prone to "Empty QA" failures.
- Enhance instructions to generate at least 4 pairs.
- Relax `is_good_question_strict` just enough to allow shorter but valid Tunisian-style questions.

### 4. Anchoring with Real Data
Modify the default behavior or the `save_output` function to ensure that "Real" chunks are preserved as the ground truth anchor.
- Update `save_output` to explicitly handle the 70/30 ratio if requested.

## Open Questions
- Do you want the script to automatically calculate the ratio (e.g. if you ask for 500 chunks total, it takes 350 real and generates 150 synthetic)?
- Or do you prefer to keep the `--merge` flag and I just ensure the output stats clearly show the "Real vs Synthetic" balance?

## Verification Plan

### Automated Verification
- Run `py utils/augmentation/synthetic_chunks.py --count 50 --output data/synthetic/audit_fix.json`.
- Verify the stats:
  - QA pairs per chunk should rise from 0.6 to >1.2.
  - Max similarity should drop below 0.90.
  - Style distribution should be more balanced (QA ~30-40%).

### Manual Verification
- Inspect `audit_fix.json` to ensure the text remains "Clean Factual" (no markdown noise).
