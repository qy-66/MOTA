# Simulated CVPR 2026 Review — MoTA

**Paper Title:** Domain-Adaptive Demoiréing via Architecture-Flexible Test-Time Adaptation
**Target Venue:** CVPR 2026
**Review Date:** 2026-05-25

---

## Reviewer 1 (Methodology Expert)

**Overall Score**: 5/10
**Confidence**: 4/5

### Summary

This paper proposes MoTA, a test-time adaptation framework for single-image demoiréing. The core idea is to insert a lightweight Fourier Domain Adapter (FDA, ~1% of backbone parameters) into a frozen pre-trained demoiréing model, then optimize only the adapter at inference time using a self-supervised signal (MASS) derived from wavelet-domain moiré frequency attenuation. The paper also explores a Spatial Gating Adapter (SGA) variant and honestly reports that parallel FDA+SGA underperforms either adapter alone.

### Strengths

1. **Clean problem formulation (Sec 3.2).** The domain shift in demoiréing is clearly characterized as capture-specific (sensor, lens, screen type, pixel pitch), and the TTA formulation as optimizing adapter parameters φ on a single image is well-posed.

2. **MASS exploits genuine domain knowledge (Sec 3.4).** The observation that moiré energy concentrates in mid-frequency DWT subbands (LH, HL) while content occupies LL and texture occupies HH is physically grounded. Using this to design frequency-selective attenuation for pseudo-clean target generation is the paper's most creative contribution.

3. **FDA design is well-motivated (Sec 3.5).** Channel-wise 1D rFFT with low-rank diagonal spectral modulation is a sensible design choice for correcting per-channel frequency-domain amplitude shifts caused by domain change. The use of the diagonal of AB^T (rather than the full C×C matrix) correctly addresses the dimension mismatch with the rFFT output. The residual connection ensures near-identity initialization.

4. **FDA/SGA competition finding (Sec 3.6, Sec 5.2).** The empirical result that FDA+SGA (+0.87 dB) underperforms FDA-only (+2.63 dB) and SGA-only (+2.07 dB) individually is an honest and potentially valuable finding for the adapter community. The "representation competition at shared insertion points" hypothesis is plausible and motivates future work on adapter orchestration.

5. **Code quality.** The implementation (tta.py, adapters.py, mass.py, mfd.py) is clean and directly matches the paper's description. The "keep best intermediate" strategy (Algorithm 1, Line 10-12) to avoid single-image overfitting is correctly implemented.

### Weaknesses

1. **Theoretical analysis is not a theory (Sec 3.7).** Equation 1 is presented as a "cross-domain generalization bound" but the paper immediately admits it is "descriptive rather than predictive" and that estimating γ "remains an open challenge." This section adds no actionable insight — it formalizes the obvious intuition that CD performance depends on MFD accuracy, which is already empirically demonstrated. A descriptive inequality without computable quantities is not a theory contribution. I recommend either making γ computable (e.g., via MFD prediction confidence) or removing this section entirely.

2. **FDA novelty relative to existing Fourier adapters.** The paper distinguishes FDA from FraIR (anonymous, 2026) by noting FraIR targets task transfer while FDA targets domain transfer. However, both use Fourier-domain low-rank modulation with a residual connection. The architectural distinction (diagonal-only modulation in FDA vs. potentially full matrix in FraIR) is thin. The core novelty lies in *when and how* the adapter is used (TTA with MASS self-supervision), not in the adapter architecture itself. The paper should be more upfront about this.

3. **"Architecture-flexible" claim not backed by method design.** The adapter insertion strategy (Sec 3.5.1: before first layer, after last layer) is described in general terms, but the code reveals that adapter channel matching drives hook placement. The claim of architecture-flexibility would be stronger with a principled insertion protocol — e.g., a rule for selecting which layers to insert adapters at given an arbitrary backbone, rather than the current heuristic (match Conv2d output channels ≥ 32, skip decode/reconstruction layers).

4. **MASS signal has an inherent tension.** MASS generates pseudo-clean targets by attenuating moiré-affected frequency bands, but the quality of those targets depends on MFD's detection accuracy. When MFD works well (ID setting), TTA improves output. When MFD degrades (CD setting), TTA makes things *worse* than frozen inference (17.50 vs. 18.91 dB). This means MoTA's TTA is a **high-variance intervention**: it improves quality when conditions are favorable and degrades it when they are not. Without a mechanism to detect *when* TTA will help vs. hurt, the method cannot be safely deployed. The paper identifies this as a limitation but doesn't propose a gating mechanism.

5. **Algorithm 1 hides important details.** Lines 3-5 (MASS generation) describe DWT → MFD → attenuation → IDWT, but the actual implementation (mass.py) includes several non-trivial steps: (a) luminance-channel shortcut (mean over RGB), (b) resolution matching (MFD output interpolation to subband size), (c) clamp to [0,1], (d) padding for odd dimensions. These are important for reproducibility.

### Questions for Authors

1. Have you considered using MFD's prediction confidence or the magnitude of MASS loss as a gating signal to decide whether to apply TTA or fall back to frozen inference? This could address the "high-variance intervention" concern.

2. The adapter competition finding is attributed to "shared insertion points." Did you try inserting FDA and SGA at *different* layers (e.g., FDA at input/output, SGA at a middle layer)? This would test whether the competition is truly about insertion points rather than gradient interaction through the backbone.

3. Why is the full C×C matrix not used for FDA? You mention "dimension mismatch" but a C×C complex matrix could operate on the frequency dimension C/2+1 through appropriate projection. Is the diagonal restriction primarily for parameter efficiency, or is there a deeper reason?

### Minor Issues

- Line 136-137: "$r = \max(4, C/32)$" — for a typical ResNet block with C=64, this gives r=4. For C=256, r=8. The formula should be explained (why C/32?).
- Equation 3: The notation `RFFT_channel(X)` is non-standard. Clarify that this is a 1D FFT along the channel dimension, applied independently per spatial position.
- Lines 177-178: The text says "Channel 2 (HH) is trained as an auxiliary task but not used for attenuation at inference" — but the sentence is immediately followed by "where σ is the sigmoid activation" (Line 179), which appears to be a copy-paste artifact from an earlier equation placement.

### Recommendation

**Weak Accept.** The paper addresses a genuine gap (no architecture-flexible TTA for demoiréing), proposes a technically sound method grounded in domain knowledge, and reports results honestly including negative findings. However, the theoretical contribution is negligible, the adapter architecture novelty is incremental, and the method cannot be safely deployed without a TTA-gating mechanism. With the theoretical section removed or strengthened, and a gating mechanism added, this would be a solid paper.

---

## Reviewer 2 (Experimentalist)

**Overall Score**: 4/10
**Confidence**: 5/5

### Summary

This paper evaluates MoTA on LCDMoire (ID) and TIP2018 (CD) with WDNet as the primary backbone and DDA for frozen baseline comparison. The central experimental finding is that MoTA improves ID performance (+2.63 dB PSNR with FDA-only at T=15) but degrades CD performance (−1.41 dB below frozen). The paper conducts thorough ablation studies and cross-domain diagnostics to understand the CD failure. I have carefully examined both the reported numbers and the evaluation code (eval.py, diagnose_cd.py).

### Strengths

1. **Honest reporting of negative results.** The paper does not hide the CD failure — it is stated clearly in the abstract, introduction, and results. Table 2 (tab:main_results) shows MoTA+FDA at 17.50 dB, below frozen WDNet at 18.91 dB. This level of transparency is commendable and should be the norm in our community.

2. **Thorough ablation design (Table 3).** The 6-way ablation (Full/FDA-only/SGA-only/w/o MASS/w/o L_reg/frozen) correctly isolates each component's contribution. The finding that removing MASS drops PSNR by 7.68 dB and removing L_reg drops it by 8.91 dB strongly validates both components in the ID setting. The T sensitivity sweep (T=0,1,5,10,15,20) with a clear peak at T=15 is well-executed.

3. **Cross-domain diagnostics (Sec 4.4.1) are well-designed.** The 4-diagnostic protocol (MASS quality, MFD behavior, adapter drift, CD ablation) systematically investigates the CD failure from multiple angles. The finding that MASS pseudo-clean achieves only 17.57 dB (2.76 dB below frozen 20.33 on the subset) directly explains why TTA fails: the optimization target is worse than the starting point.

4. **Code-verified evaluation protocol.** I verified that the evaluation metrics (PSNR, SSIM via scikit-image, LPIPS via AlexNet), data preprocessing ([-1,1] range, random crop training, full-resolution testing), and TTA loop (Adam optimizer, L1 loss, best-intermediate selection) match between the paper and eval.py + tta.py. The cudnn.deterministic=True setting and multi-seed verification (42, 123, 2026) confirming std < 0.02 are properly implemented.

5. **LoRA baseline is a fair comparison.** Using LoRA with rank=4 applied to all Conv2d layers, trained under identical TTA protocol (same T=15, same MASS loss, same L_reg), provides a meaningful parameter-efficient baseline. The result that FDA-only (1.1% params, 24.12 dB) outperforms LoRA (4.5% params, 23.31 dB) is a useful data point for the adapter design space.

### Weaknesses

1. **Single-backbone TTA evaluation invalidates "architecture-flexible" claim.** The paper's title and central claim is "Architecture-Flexible Test-Time Adaptation," yet full TTA evaluation is conducted on exactly ONE backbone (WDNet). DDA is only reported as a frozen baseline; the paper acknowledges that "DDA MBCNN 1024→256... TTA optimization cannot converge" (experiment_guide.md). This is a serious gap: the architecture-flexibility claim requires demonstrating TTA on at least two architecturally distinct backbones. Without this, the paper should be retitled to remove "architecture-flexible" or limit the claim to "an adapter for wavelet-based demoiréing networks."

2. **CD diagnostic on unrepresentative 100-image subset.** Table 6 (tab:cd_diagnostics) shows CD frozen PSNR of 20.33 dB on the 100-image subset, vs. 18.91 dB on the full 11,851-image TIP2018 test set — a +1.42 dB deviation. The paper acknowledges this caveat (Note under Table 6), but then builds the entire CD failure analysis (Factors 1–3, Sec 4.4.1) on this potentially unrepresentative subset. The central finding that "MASS pseudo-clean is 2.76 dB below frozen on this subset" needs verification on at least 500+ images. A 100-image subset with +1.42 dB bias relative to the full set is insufficient for the paper's primary analytical claim.

3. **Missing metrics for best-performing variant.** Table 2 reports SSIM and LPIPS for Full MoTA (FDA+SGA, 22.36 dB) but uses "---" for FDA-only (24.12 dB, the recommended configuration). The paper states "SSIM and LPIPS for FDA-only ID were not computed." Since FDA-only is the recommended configuration and achieves the best PSNR, omitting its SSIM and LPIPS is a significant gap. These metrics should be straightforward to compute (the eval.py `compute_metrics` function already supports them).

4. **Adapter drift diagnostic not executed.** Table 6 reports "---" for adapter drift with footnote "Requires full TTA loop per image; not run." This is the one diagnostic that could distinguish between "MASS signal is too noisy" and "adapters have insufficient capacity" as the primary CD bottleneck. Running it on even 10 images would provide valuable insight.

5. **No computational cost analysis for CD.** The inference efficiency table (Table 5) shows MoTA+FDA takes 788.6ms per image (83× slower than frozen WDNet at 9.4ms). This is reported for T=15, but the paper doesn't discuss whether T could be reduced for CD (where adaptation doesn't help anyway) or whether MASS generation dominates the TTA overhead. A breakdown of: MASS generation time vs. per-step adapter update time would be informative.

6. **DDA CD result requires explanation.** DDA drops from 29.01 dB (ID) to 12.26 dB (CD), a catastrophic 16.75 dB collapse. The paper attributes this to "internal downsampling layers incompatible with TIP2018 resolution" (Table 2 footnote), but this explanation is insufficient. If DDA's 4× downsampling is the issue, the model should still produce reasonable outputs at 256×256 (upsampled to TIP2018 resolution). A 12.26 dB PSNR on a 400×400 image suggests something more fundamental is broken — possibly input range mismatch ([0,1] vs [-1,1]) or the LCDMoire→TIP2018 domain gap interacting with DDA's specific architecture. As this is one of only two backbones evaluated, understanding this failure is important.

7. **No comparison to the most directly related method.** MoiréXNet (li2025moirexnet) is discussed extensively in Related Work as "the only prior work combining test-time training with demoiréing" with four specific differences listed. Yet there is no experimental comparison, not even a frozen baseline comparison. The paper cites MoiréXNet results from its original publication but doesn't attempt to evaluate it under the same cross-domain protocol. I understand that MoiréXNet requires multi-frame RAW input (different input modality), but at minimum, a discussion of why direct comparison is infeasible should be included in the experiments section.

### Questions for Authors

1. Why wasn't adapter drift (Diagnostic 3) executed? The code exists in diagnose_cd.py and supports --skip_drift. Even 10 images would provide useful data on whether CD gradients are weaker (suggesting MASS is the bottleneck) or stronger but misdirected (suggesting adapter capacity is the bottleneck).

2. DDA was trained on LCDMoire at 256×256 patch resolution. Was the 12.26 dB CD result computed at native TIP2018 resolution (~400×400) with bilinear upsampling from 256×256 output, or at a different resolution? The 16.75 dB CD-Gain for DDA seems extreme even for a severe domain gap.

3. Have you tested FDA-only on the CD setting with fewer adaptation steps (T=1,3,5)? The T sensitivity sweep (Table 3, bottom) shows monotonic improvement from T=1 to T=15 in ID, but in CD where MASS is unreliable, fewer steps might reduce the degradation.

4. The MASS preprocess baseline (MASS attenuation without TTA optimization) achieves 18.90 dB, essentially identical to frozen (18.91 dB). Does this mean the MASS attenuation alone (without adapter optimization) provides zero benefit? If so, this suggests MASS is not removing moiré — it's just creating a target for optimization.

### Minor Issues

- Table 2: The footnote about DDA CD mentions "bilinear upsampling to target resolution." Clarify the exact resolution chain: 400×400 input → 256×256 internal → bilinear upsampled to what?
- Figure 3 (MASS visualization): I cannot verify the content of this figure. The heatmap overlay and subband comparisons should clearly label which subband is shown and the attenuation factor applied.
- The CD ablation (Diagnostic 4) uses a try/except fallback to frozen inference on failure (diagnose_cd.py lines 278-279, 300-302). If these fallbacks were triggered during data collection, the reported numbers would be contaminated with frozen outputs. Please report whether any exceptions occurred.

### Recommendation

**Borderline (tending Weak Reject).** The paper's strengths are its honest negative-result reporting, thorough ID ablation, and systematic CD diagnostics. However, the central "architecture-flexible" claim is experimentally unsupported (only one backbone with full TTA), the CD analysis rests on a 100-image subset that deviates +1.42 dB from the full set, and key metrics for the best variant are missing. These are fixable issues, but in their current state, the experimental validation is insufficient for the claims made. I would raise my score to Weak Accept if: (a) a second backbone with full TTA evaluation is added, (b) CD diagnostics are expanded to ≥500 images, and (c) SSIM/LPIPS for FDA-only are reported.

---

## Reviewer 3 (Writing & Presentation Critic)

**Overall Score**: 4.5/10
**Confidence**: 4/5

### Summary

The paper is generally well-written with a clear narrative arc: problem (domain overfitting) → solution (TTA with adapters + MASS) → ID success (+2.63 dB) → CD failure → honest diagnostics. The writing quality is above average for a first submission. However, there are several presentation issues that collectively reduce the paper's quality, most critically the page count.

### Strengths

1. **Clear narrative structure.** The paper follows a logical progression: Introduction establishes the domain overfitting problem → Related Work identifies the gap at the intersection of demoiréing, TTA, and PEFT → Method presents the three-stage MoTA pipeline → Experiments deliver honest results with diagnostics. The reader always knows why each section exists.

2. **Honest tone throughout.** From the abstract ("this in-domain improvement does not transfer to the cross-domain setting") to the conclusion ("extending this principle to cross-domain settings... remains an open challenge"), the paper consistently acknowledges limitations. This builds credibility with the reader.

3. **Well-organized Related Work (Sec 2).** The five-subsection structure (Image Demoiréing → High-Res → RAW-Domain → Video → Data-Centric → TTA → PEFT) is comprehensive and each subsection ends by connecting back to the paper's contribution. The summary paragraph (lines 95-96) clearly states the gap MoTA fills.

4. **Effective use of tables for complex information.** Table 1 (tab:analysis) concisely summarizes six method classes with their core ideas, strengths, and limitations. Table 2 (tab:datasets) clearly contrasts LCDMoire and TIP2018. Table 5 (tab:cd_diagnostics) efficiently presents the multi-faceted CD diagnostic results.

5. **Algorithm 1 is readable.** The three-stage structure with inline comments and a "Key innovations" summary box is effective for communicating the method.

### Weaknesses

1. **CRITICAL: Page count far exceeds CVPR limit.** The paper is 16 pages in two-column CVPR format. CVPR 2026 enforces an 8-page limit (excluding references). At double the allowed length, this submission would be desk-rejected before review. The paper must be trimmed to ≤8 pages. Suggested cuts: (a) Collapse Sec 2.2 (High-Res Demoiréing) and Sec 2.3 (RAW-Domain) into brief paragraphs within Sec 2.1 — these sub-communities are relevant but don't each need their own subsection. (b) Remove or drastically shorten Sec 3.7 (Theoretical Analysis) — as Reviewer 1 notes, this is not a theory contribution. (c) Move the CD diagnostic methodology details to supplementary material, keeping only the key results in the main paper. (d) Reduce the Limitations section from 6 items to the 3 most important. (e) Merge Sec 4.5 (Efficiency Analysis) into Sec 4.4 (Main Results).

2. **7 of 28 references are anonymous arXiv preprints.** anonymous2025moirenet, anonymous2025mffnet, anonymous2023dda, anonymous2025fpanet, anonymous2024undem, anonymous2025sidme, anonymous2026frair — 25% of the bibliography lacks author attribution and peer review. While some of these may be under review, citing 7 anonymous works weakens the paper's scholarly foundation. For CVPR submission, replace anonymous citations with identifiable versions (author names, updated arXiv versions with deanonymized authors) wherever possible.

3. **Copy-paste artifact in Sec 3.4.1.** Lines 177-178 describe MFD output channels, and Line 179 begins "where σ is the sigmoid activation" — but Equation 2 (M = σ(MFD(...))) appears on Line 174. The sentence on Line 179 appears to be a remnant from an earlier draft where the equation was presented after the channel description.

4. **Inconsistent level of detail.** The paper provides exhaustive implementation details (optimizer, batch size, LR schedule, VGG warmup epochs, wavelet loss weights) but is vague about other important details: MFD architecture ("lightweight 4-layer CNN" — what kernel sizes? stride pattern?), adapter initialization protocol ("pre-initialized to approximate identity mappings via 10-epoch training" — on what data? what loss?), and the exact channel counts for WDNet backbone stages.

5. **Missing Figure 1 content verification.** The framework overview (Fig 1) is referenced as `figures/fig1_framework.jpg` at 0.95\linewidth within a single column. A framework diagram with three stages and colored boxes at single-column width may be difficult to read. Consider making this a `figure*` spanning both columns, or simplifying the diagram.

6. **Table 3 (ablation) is three separate tabular environments.** The ablation results, T sensitivity sweep, and adapter efficiency comparison appear as three separate `tabular` blocks within one `table` environment. This is confusing — readers may think "Table 3" refers only to the first block. Either merge them into a single comprehensive table or split into clearly labeled sub-tables (Table 3a, 3b, 3c).

7. **The MASS visualization (Fig 3) is referenced but its content cannot be assessed from the LaTeX source alone.** The caption describes "MFD output overlaid as heatmap, and frequency subbands before and after MASS attenuation." At 0.28\textheight with `keepaspectratio`, the subband comparisons may be too small to read. Ensure individual subband panels are labeled and the attenuation effect is visually clear.

8. **Overfull hbox warnings.** The CLAUDE.md and PAPER_SPEC.md mention 14 Overfull hbox warnings (max 77pt on Table 1). These should be resolved — at minimum, the worst offenders should be fixed before submission.

### Questions for Authors

1. The paper oscillates between "FDA+SGA is the full method" (early sections, abstract) and "FDA-only is recommended" (later sections, ablation, conclusion). Have you considered restructuring to present FDA-only as the primary method from the start, with SGA as a studied-but-rejected design alternative? This would simplify the narrative.

2. The abstract mentions "<2% of backbone parameters" for FDA, but the main text reports 1.1%. The abstract also mentions "SGA variant is explored but found to compete with FDA" which is an unusual amount of detail for an abstract. Consider a tighter abstract focused on FDA-only.

### Minor Issues

- Line 32: Title is "Domain-Adaptive Demoiréing via Architecture-Flexible Test-Time Adaptation" but the CLAUDE.md calls it "Model-Agnostic" — the abstract uses "plug-and-play" and the text uses "architecture-flexible." Pick one term and use it consistently.
- Line 44: "As smartphone photography of screens becomes ubiquitous in daily life — from photographing documents and QR codes to sharing social media posts — the demand for robust demoiréing continues to grow." This sentence has an em-dash parenthetical that makes it hard to parse. Consider simplifying.
- Line 139-140 (Table 1): CLEAR (CVPR 2026) is cited as concurrent work but its relationship to MoTA is described in one sentence. Either expand this to a brief paragraph or remove it from the table.
- The footnotes under tables use inconsistent font sizes (\small, \footnotesize, \scriptsize).

### Recommendation

**Borderline (tending Weak Reject).** The writing is clear and the narrative is honest, which I appreciate. However, the paper is literally twice the allowed page limit for CVPR — this alone would trigger desk rejection. The high proportion of anonymous references and several presentation issues (copy-paste artifact, inconsistent terminology, table organization) further reduce the presentation quality. These are all fixable, but in the current version, the paper does not meet CVPR's presentation standards.

---

## Anchor Comparison

**Reference point — CVPR 2026 reviewing standards:**
CVPR 2026 evaluates papers on technical novelty, experimental validation, and presentation quality. Papers with solid experimental validation and valuable insights may be considered for the new "Findings" track even if technical novelty is incremental. The key criteria are: will this paper interest CVPR attendees, is it technically sound, and does it contribute meaningfully?

**Anchor Paper A — Strong Accept (~7.5/10) equivalent:**
A hypothetical CVPR 2026 paper on test-time adaptation for image restoration that: (a) demonstrates TTA on ≥3 architecturally distinct backbones, (b) provides theoretical guarantees on adaptation quality, (c) achieves consistent improvement across ≥3 diverse domain shifts, (d) releases code and model weights.

**Anchor Paper B — Borderline (~5.5/10) equivalent:**
A paper that: (a) proposes a novel adapter for a specific restoration task, (b) validates on 2 datasets with 1 backbone, (c) reports both positive and negative results, (d) provides ablation studies.

**This paper's position relative to anchors:**
- **vs. Anchor A:** MoTA falls short on backbone diversity (1 vs. 3), lacks computable theoretical guarantees, and shows CD degradation rather than consistent improvement. The code quality and ablation design are competitive, but the scope of validation is narrower.
- **vs. Anchor B:** MoTA matches or exceeds Anchor B in ablation quality, diagnostic thoroughness, and honest reporting. It falls short on backbone diversity (Anchor B would have 2 backbones with full evaluation) and has the page count issue.
- **Estimated position:** Between Anchor B and a full reject — closer to the borderline. The paper has genuine strengths (honest negative results, thorough ablation, clean code) but has several issues that need addressing before it meets the CVPR bar.

---

## Meta-Review (Area Chair)

**Average Score**: 4.5 / 10
**Score Range**: 4.0 – 5.0

### Consensus Strengths

1. **Honest and transparent reporting of negative results.** All three reviewers commend the paper's willingness to report and analyze the cross-domain failure. This is rare and valuable — the community needs more papers that honestly document what doesn't work.

2. **Well-designed ablation and diagnostics.** The 6-way ablation correctly isolates component contributions, the T sensitivity sweep shows a clear peak at T=15, and the 4-diagnostic CD analysis systematically investigates the failure modes. The FDA/SGA competition finding is a genuine scientific insight.

3. **Technically sound method grounded in domain knowledge.** MASS exploits the true frequency structure of moiré patterns (mid-frequency energy concentration). FDA's channel-wise spectral modulation is a sensible design for correcting frequency-domain domain shifts. The code implementation matches the paper description.

4. **Clean, well-organized writing with clear narrative.** The paper is easy to follow and logically structured.

### Consensus Weaknesses

1. **"Architecture-flexible" claim is experimentally unsupported.** Full TTA evaluation on only one backbone (WDNet). The paper's title and central claim require demonstration on at least two architecturally distinct backbones. (R1, R2 agree)

2. **CD diagnostics on unrepresentative 100-image subset.** The subset shows frozen PSNR of 20.33 dB vs. 18.91 dB on the full TIP2018 test set (+1.42 dB bias). The paper's primary analytical finding (MASS pseudo-clean quality as CD bottleneck) needs verification on a larger subset. (R2)

3. **Page count is 16 pages — 2× the CVPR 8-page limit.** This alone would trigger desk rejection. (R3)

4. **Weak theoretical contribution.** The "generalization bound" is admitted to be descriptive rather than predictive. (R1, R3 agree this section should be cut or strengthened.)

5. **7 of 28 references are anonymous arXiv preprints.** (R3)

6. **Missing key experimental data.** SSIM/LPIPS for FDA-only (best variant) not reported. Adapter drift diagnostic not executed. (R2)

### Key Disagreements

- **Severity of single-backbone limitation.** R2 considers this a critical weakness that invalidates the central claim. R1 considers it addressable but notes the method's adapter insertion is described in general terms. Both agree more backbones are needed; the disagreement is about whether this is fixable within the revision cycle.

### Accept/Reject Prediction

**Prediction**: **Weak Reject**
**Confidence**: Medium

**Reasoning**: The paper has genuine strengths — honest negative-result reporting, thorough ablation, clean method design — that would make it a valuable contribution to the community. The core idea (TTA for demoiréing with frequency-aware self-supervision) fills a genuine gap.

However, three issues are individually sufficient to prevent acceptance: (1) the 16-page length violates CVPR's submission policy and would trigger desk rejection, (2) the "architecture-flexible" claim is not experimentally validated (single backbone with full TTA), and (3) the CD failure analysis — the paper's central analytical contribution — rests on 100 images that deviate +1.42 dB from the full dataset.

If the authors address these three issues (trim to 8 pages, add a second backbone with full TTA, expand CD diagnostics to ≥500 images), I would expect the paper to reach the borderline/weak-accept range. The new CVPR 2026 "Findings" track may be a good fit for this work given its honest documentation of negative results and valuable diagnostic insights — provided the page limit and backbone diversity issues are resolved.

---

## 修改优先级

### P0 (Must Fix — won't pass without these)

1. **Trim paper to ≤8 pages (CVPR limit).** → Cut: Sec 3.7 (Theoretical Analysis), merge Sec 2.2+2.3 into Sec 2.1, move CD diagnostic methodology details to supplementary, reduce Limitations to top 3 items, merge Efficiency into Main Results. Estimated savings: 8 pages.

2. **Add full TTA evaluation on a second backbone to support "architecture-flexible" claim.** → If DDA TTA doesn't converge (as noted), try MBCNN, FHDe2Net, or MoiréNet. If no second backbone can be made to work with TTA, change the title to remove "architecture-flexible" and limit the claim to "a TTA adapter for wavelet-based demoiréing networks."

3. **Expand CD diagnostics to ≥500 images.** → The 100-image subset's +1.42 dB bias makes the current analysis unreliable. Running the existing diagnose_cd.py with --n_images 500 should be computationally feasible (only MASS quality and MFD behavior are per-image; ablation is per-image but can be subsetted). At minimum, verify that the MASS pseudo-clean < frozen result holds on a representative sample.

### P1 (Should Fix — significantly strengthens the paper)

4. **Report SSIM and LPIPS for FDA-only (best variant).** → Run compute_metrics with FDA-only, T=15 on the ID test set. These metrics are already implemented in eval.py.

5. **Run adapter drift diagnostic on 10 images.** → Remove --skip_drift and run diagnose_cd.py with --n_drift_images 10. This distinguishes MASS quality from adapter capacity as the CD bottleneck.

6. **Replace or de-anonymize anonymous references.** → For references that have been accepted/published since submission, update with real author names and venues. For those still under review, consider whether they are essential or can be replaced with published alternatives.

7. **Add T-sensitivity for CD.** → Test FDA-only at T=1,3,5 on CD to see if fewer adaptation steps reduce the degradation below frozen.

8. **Fix the copy-paste artifact at Line 179** (σ description after MFD output channel explanation).

### P2 (Nice to Have — improves quality but not blocking)

9. **Resolve 14 Overfull hbox warnings** (particularly the 77pt overflow on Table 1). → Adjust column widths or switch to `table*`.

10. **Add inference cost breakdown** (MASS generation time vs. per-step adapter update time vs. backbone forward pass).

11. **Clarify DDA CD result.** → Explain what specifically causes 12.26 dB (range mismatch, resolution chain, or genuine domain gap sensitivity).

12. **Standardize terminology** ("model-agnostic" vs. "architecture-flexible" vs. "plug-and-play"). Use one term consistently.

13. **Consider restructuring to present FDA-only as primary method** → Move SGA discussion to a clearly labeled "Explored Design Alternative" subsection.

14. **Rename or remove the "Theoretical Analysis" section.** → If γ cannot be computed, the inequality is not a theory contribution. Either make γ computable (e.g., from MFD prediction confidence on the specific test image) or replace with a qualitative discussion of when MASS is expected to work.

15. **Add discussion of why direct MoiréXNet comparison is infeasible** (different input modality: multi-frame RAW vs. single sRGB) in the Experiments section.

---

*Review generated by simulated CVPR 2026 review panel (3 reviewers + meta-review). Based on thorough reading of paper.tex + code verification + CVPR 2026 review guidelines.*
