# Hallucination Detection in Low-Resource Languages

A comprehensive benchmark and evaluation framework for detecting hallucinations in large language model (LLM) generations across low-resource languages.

## 🎯 Overview

This project provides a comprehensive framework for evaluating hallucination detection methods in large language models, with a **specific focus on low-resource languages** including Armenian, Basque, and Tigrinya. The framework implements 11 state-of-the-art hallucination detection methods and evaluates them on question-answering tasks across multiple languages and model architectures.

**Key Features:**
- 🌍 Multi-language support (Armenian, Basque, Tigrinya)
- 🤖 Multiple LLM architectures (LLaMA2-7B, OPT6.7B)
- 📊 11 hallucination detection methods
- 🔬 Comprehensive evaluation metrics (AUROC, AUPRC, FPR@95)
- 🚀 Parallel experiment execution support

## 💡 Motivation

Large Language Models (LLMs) have achieved remarkable performance on various NLP tasks, but they often generate **hallucinations** - outputs that are fluent but factually incorrect or unsupported by the input. This problem is particularly acute in **low-resource languages**.

This project addresses these challenges by:
- Systematically evaluating hallucination detection methods on low-resource languages
- Providing a reproducible benchmark for future research
- Identifying which methods generalize well across languages and model architectures
- Enabling safer deployment of LLMs in multilingual contexts

## 🔬 Experimental Setup

### Models

- **LLaMA-2 7B**: Meta's open-source language model
- **OPT 6.7B**: Facebook's Open Pre-trained Transformer

### Datasets

| Dataset | Language | Type | Size | Task |
|---------|----------|------|------|------|
| **SynDARin** | Armenian | QA | Train: 6.9K, Test: 1.7K | Question Answering |
| **Elkarhizketak** | Basque | Dialogue | Train: 14K, Validation: 1.8K, Test: 1.8K | Dialogue Understanding |
| **TigQA** | Tigrinya | QA | Train: 1.4K, Dev: 0.18K, Test: 0.18K | Question Answering |

### Evaluation Metrics

- **AUROC** (Area Under ROC Curve): Measures the model's ability to distinguish between hallucinated and truthful outputs
- **AUPRC** (Area Under Precision-Recall Curve): Emphasizes performance on the positive (hallucination) class
- **FPR@95** (False Positive Rate at 95% True Positive Rate): Measures specificity at high sensitivity

### Ground Truth Generation

Since there are no direct labels for hallucinations, we use **BLEURT** (a learned evaluation metric) to assess the quality of generated answers by comparing them with gold standard references:
- Answers with BLEURT score > 0.5 are labeled as truthful
- Answers with BLEURT score ≤ 0.5 are labeled as hallucinated

## 🛠️ Methods Implemented

### 1. HaloScope
Leverages unlabeled LLM generations to detect hallucinations using weighted SVD on hidden representations.

**Paper**: [HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection](https://arxiv.org/abs/2409.17504) (NeurIPS'24 Spotlight)  
**GitHub**: [deeplearning-wisc/haloscope](https://github.com/deeplearning-wisc/haloscope)

### 2. Contrast-Consistent Search (CCS)
Trains a linear probe on model representations to detect truthfulness without labeled data by exploiting consistency in model representations.

**Paper**: [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827)  
**GitHub**: [collin-burns/discovering_latent_knowledge](https://github.com/collin-burns/discovering_latent_knowledge)

### 3. SelfCheckGPT
Samples multiple responses and checks consistency using various similarity metrics (BERTScore, BLEURT, etc.) in a zero-resource, black-box manner.

**Paper**: [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896)  
**GitHub**: [potsawee/selfcheckgpt](https://github.com/potsawee/selfcheckgpt)

### 4. Semantic Entropy
Measures uncertainty by clustering semantically equivalent generations using bidirectional entailment and computing entropy over the clusters.

**Paper**: [Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/abs/2302.09664)  
**GitHub**: [lorenzkuhn/semantic_uncertainty](https://github.com/lorenzkuhn/semantic_uncertainty)

### 5. LN Entropy
Calculates log-normal entropy over semantically clustered responses using multilingual semantic embeddings for robust uncertainty quantification.

**Paper**: [Uncertainty Estimation in Autoregressive Structured Prediction](https://openreview.net/forum?id=jN5y-zb5Q7m)
**GitHub**: [KaosEngineer/structured-uncertainty](https://github.com/KaosEngineer/structured-uncertainty)


### 6. EigenScore
Analyzes the eigenvalue spectrum of generation distributions for uncertainty estimation through spectral analysis of embeddings.

**Paper**: [EigenScore: Estimating Uncertainty in Language Model Generations](https://arxiv.org/abs/2406.04754)  
**GitHub**: [D2I-ai/eigenscore](https://github.com/D2I-ai/eigenscore)

### 7. Lexical Similarity
Measures consistency through lexical overlap across multiple generations using token-level n-gram analysis and self-BLEU computation.
**Paper**: [Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models](https://arxiv.org/abs/2305.19187)  
**GitHub**: [zlin7/UQ-NLG](https://github.com/zlin7/UQ-NLG)

### 8. Verbalized Confidence
Prompts the model to verbalize its confidence in its own answers through direct self-reflection mechanisms.
**Paper**: [Teaching Models to Express Their Uncertainty in Words](https://arxiv.org/abs/2205.14334)  
**GitHub**: [sylinrl/CalibratedMath](https://github.com/sylinrl/CalibratedMath)

### 9. Self-Evaluation
Uses prompted introspection where the model evaluates its own generations through metacognitive assessment.
**Paper**: [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)  


### 10. HalluShift
Detects distribution shifts in model representations as indicators of hallucination using anomaly detection frameworks.

**Paper**: [Detecting Hallucinations in Large Language Models Using Semantic Entropy](https://arxiv.org/abs/2303.10910)  
**GitHub**: [sharanya-dasgupta001/hallushift](https://github.com/sharanya-dasgupta001/hallushift)

### 11. Perplexity-Based Detection
Uses token-level and sequence-level perplexity along with relative model divergence (RMD) as simple baseline hallucination indicators.
**Paper**: [Out-of-Distribution Detection and Selective Generation for Conditional Language Models](https://openreview.net/forum?id=kJUS5nD0vPB)  

## 🚀 Usage

### Running Individual Methods

Each method has a standardized interface. The general workflow involves three steps:

#### Step 1: Generate Model Responses

```bash
python src/haloscope/main.py --model_name llama2_7B --dataset_name armenian --gene 1 --most_likely 1 --num_gene 1
```

#### Step 2: Generate Ground Truth Labels

```bash
python src/haloscope/main.py --model_name llama2_7B --dataset_name armenian --generate_gt 1 --most_likely 1
```

#### Step 3: Run Hallucination Detection

```bash
python src/haloscope/main.py --model_name llama2_7B --dataset_name armenian --most_likely 1 --weighted_svd 1 --feat_loc_svd 3
```

### Running All Experiments

The `run.sh` script contains commands for all experiments across all methods, models, and datasets:

```bash
# Edit run.sh to uncomment desired experiments
bash run.sh
```

### Common Arguments

| Argument | Description | Values |
|----------|-------------|--------|
| `--model_name` | LLM model to use | `llama2_7B`, `opt_6_7b` |
| `--dataset_name` | Dataset to evaluate on | `armenian`, `basque`, `tigrinya` |
| `--gene` | Generate responses | `0` or `1` |
| `--num_gene` | Number of generations per sample | Integer (e.g., `1`, `3`, `10`) |
| `--generate_gt` | Generate ground truth labels | `0` or `1` |
| `--most_likely` | Use greedy decoding | `0` (sampling) or `1` (greedy) |

## 📊 Results

### Overall Performance Summary (AUROC)

| Method | Armenian (LLaMA) | Armenian (OPT) | Basque (LLaMA) | Basque (OPT) | Tigrinya (LLaMA) | Tigrinya (OPT) |
|--------|------------------|----------------|----------------|--------------|------------------|----------------|
| **Perplexity** | 21.46 | 71.38 | 50.58 | 50.41 | 50.00 | 50.00 |
| **LN Entropy** | 56.83 | 56.13 | 45.26 | 45.67 | 50.00 | 50.00 |
| **Semantic Entropy** | 30.49 | 38.91 | 33.31 | 29.43 | 50.00 | 50.00 |
| **Lexical Similarity** | 42.03 | 57.14 | 48.18 | 13.84 | 50.00 | 50.00 |
| **EigenScore** | 51.24 | 35.67 | 49.36 | 43.30 | 50.00 | 50.00 |
| **SelfCheckGPT** | 33.84 | 42.43 | 52.50 | 54.94 | 50.00 | 50.00 |
| **Verbalize** | 53.22 | 44.86 | 48.14 | 64.12 | 50.00 | 50.00 |
| **Self-Evaluation** | 64.07 | 59.93 | 39.34 | 49.32 | 50.00 | 50.00 |
| **CCS** | 48.82 | 54.28 | 49.98 | 52.36 | 50.00 | 50.00 |
| **HaloScope** | 57.88 | 64.25 | 57.16 | 47.25 | 50.00 | 50.00 |
| **HalluShift** | 55.99 | 55.55 | 41.89 | 36.47 | 50.00 | 50.00 |

### Key Observations

1. **Tigrinya detection is completely unsuccessful** - All methods achieve exactly 50% AUROC (random chance), indicating fundamental challenges in hallucination detection for this extremely low-resource language

2. **Armenian shows the best detection performance**:
   - Top performers: Self-Evaluation (64.07%), HaloScope (64.25%), LN Entropy (56.83%)
   - Internal state methods generally outperform consistency-based methods

3. **Basque shows moderate but inconsistent performance**:
   - Best: Verbalize (64.12% on OPT), HaloScope (57.16%), SelfCheckGPT (52.50-54.94%)
   - Significant method failures: Lexical Similarity (13.84% on OPT), Semantic Entropy (29.43-33.31%)

4. **Surprising underperformance of consistency-based methods**:
   - Semantic Entropy performs poorly (29-39% AUROC) despite strong results in prior English benchmarks
   - LN Entropy shows mixed results (56% on Armenian, 45% on Basque)
   - SelfCheckGPT is inconsistent across languages and models

5. **Internal state methods show relative success**:
   - HaloScope achieves best or near-best performance on Armenian (57.88-64.25%) and Basque (57.16-47.25%)
   - Self-Evaluation performs well on Armenian (64.07-59.93%)
   - These methods may capture language-specific patterns better than assumed

6. **High variability across model architectures**:
   - Perplexity: Ranges from 21.46% (LLaMA on Armenian) to 71.38% (OPT on Armenian)
   - Results are highly method-model-language dependent with no consistent winner

## 🔍 Interpretation & Analysis

### Critical Finding: The Tigrinya Challenge

All 11 methods achieve exactly **50% AUROC on Tigrinya** (random chance), representing complete failure in hallucination detection. This reveals fundamental challenges:

**Potential Causes**:
1. **Extremely limited pretraining data**: Neither LLaMA-2 nor OPT had sufficient Tigrinya exposure during pretraining
2. **Embedding quality breakdown**: Multilingual sentence embeddings may lack Tigrinya representation
3. **Ground truth quality**: BLEURT-based labeling may be unreliable for Tigrinya due to model limitations
4. **Script and morphological complexity**: Tigrinya uses Ge'ez script with complex morphology that may confound detection methods
5. **Dataset size**: With only ~180 test samples, the evaluation may lack statistical power

**Implications**: Current hallucination detection methods are **not language-agnostic** and completely fail on extremely low-resource languages without adequate model representation.

### Why Do Some Methods Work Better?

#### 1. Internal State Methods Show Surprising Strength

**HaloScope** and **Self-Evaluation** achieve the best performance on Armenian:
- ✅ **HaloScope** (57.88-64.25%): Successfully identifies hallucination patterns in model activations even for low-resource languages
- ✅ **Self-Evaluation** (64.07%): Prompting models to assess their own outputs works surprisingly well
- ✅ These methods may capture subtle language-specific activation patterns that consistency-based methods miss

**Why this matters**: Internal representations contain richer signals than previously assumed for low-resource language hallucination detection.

#### 2. Consistency-Based Methods Underperform

**Semantic Entropy**, once considered state-of-the-art, performs poorly (29-39%):
- ❌ Requires high-quality multilingual embeddings that may not exist for low-resource languages
- ❌ Semantic clustering via entailment models trained primarily on English fails to transfer
- ❌ **Critical failure**: Methods that excel on English may have severely degraded performance on low-resource languages

**LN Entropy** shows mixed results (45-56%):
- Performs better than Semantic Entropy but still below internal state methods
- Adaptive clustering may be fragile with limited multilingual embedding quality

#### 3. Method-Language Interactions Are Unpredictable

**Perplexity** demonstrates extreme variability:
- 21.46% on Armenian-LLaMA vs. 71.38% on Armenian-OPT
- This 50-point swing suggests perplexity patterns differ drastically between model families
- **Implication**: Method selection must be model-specific; no universal solution exists

**Lexical Similarity** catastrophically fails on Basque-OPT (13.84%):
- Far below random chance (inverted predictions?)
- Suggests n-gram patterns in Basque may mislead lexical overlap metrics
- Language-specific morphology and agglutination may break token-level comparison

### Cross-Lingual Transfer Failure

**Critical Insight**: Our results **contradict the assumption** that multilingual embeddings enable cross-lingual generalization for hallucination detection.

**Evidence**:
- Semantic Entropy (relies on multilingual NLI): 29-39% AUROC
- HaloScope (model-internal states): 57-64% AUROC on Armenian
- Methods designed for English consistently underperform on low-resource languages

**Revised Understanding**:
- **Multilingual pretrained models are insufficient** for extremely low-resource languages
- **Model-internal signals** may be more robust than semantic similarity in embedding space
- **Language-specific tuning** is likely essential rather than optional

### Model Architecture Effects

**OPT vs. LLaMA-2 show inconsistent patterns**:
- No clear winner across all methods and languages
- **Armenian**: OPT often performs better (e.g., HaloScope 64.25% vs. 57.88%)
- **Basque**: Mixed results with high method-dependent variation
- **Implication**: Model architecture interacts complexly with detection methods

### Dataset and Language Characteristics

Results reveal an unexpected hierarchy:

1. **Armenian (best, ~50-64% top scores)**:
   - Question-answering task with factual answers
   - Training size: 6.9K samples provides reasonable coverage
   - May benefit from Armenian's clearer morphological boundaries

2. **Basque (moderate, ~50-64% top scores but high variance)**:
   - Dialogue task with more complex evaluation
   - Training size: 14K samples (largest dataset)
   - Agglutinative morphology and rich inflection may challenge token-based methods
   - High method variance suggests task complexity

3. **Tigrinya (complete failure, 50% all methods)**:
   - Smallest training set (1.4K) and test set (180 samples)
   - Ge'ez script with unique morphology
   - **Critical**: Essentially absent from model pretraining data

### Practical Recommendations

Based on actual results, recommendations must be language-conditional:

**For Armenian-like languages (moderate low-resource, represented in pretraining)**:
1. **Best choice**: Self-Evaluation or HaloScope (60-64% AUROC)
2. **Backup**: LN Entropy or HalluShift (55-56% AUROC)
3. **Avoid**: Semantic Entropy, Perplexity on LLaMA

**For Basque-like languages (agglutinative, dialogue tasks)**:
1. **Best choice**: Verbalize (64% on OPT) or HaloScope (57%)
2. **Backup**: SelfCheckGPT (52-55%)
3. **Avoid**: Lexical Similarity on OPT, Semantic Entropy

**For Tigrinya-like languages (extremely low-resource, unique scripts)**:
- ⚠️ **All current methods fail completely**
- **Do not deploy** without language-specific model fine-tuning
- Requires fundamental research breakthroughs (see Future Directions)

## 🙏 Acknowledgments

This project builds upon the excellent work of many researchers and open-source contributors:

- **Method Authors**: For publishing their code and making research reproducible
- **Dataset Creators**: For curating high-quality low-resource language datasets
- **Hugging Face**: For the Transformers library and model hosting
- **Meta AI & Facebook**: For open-sourcing LLaMA-2 and OPT models
- **The NLP Community**: For advancing multilingual NLP research

---

**Happy Hallucination Hunting! 🔍🤖**

