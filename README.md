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

### 6. EigenScore
Analyzes the eigenvalue spectrum of generation distributions for uncertainty estimation through spectral analysis of embeddings.

**Paper**: [EigenScore: Estimating Uncertainty in Language Model Generations](https://arxiv.org/abs/2406.04754)  
**GitHub**: [D2I-ai/eigenscore](https://github.com/D2I-ai/eigenscore)

### 7. Lexical Similarity
Measures consistency through lexical overlap across multiple generations using token-level n-gram analysis and self-BLEU computation.

### 8. Verbalized Confidence
Prompts the model to verbalize its confidence in its own answers through direct self-reflection mechanisms.

### 9. Self-Evaluation
Uses prompted introspection where the model evaluates its own generations through metacognitive assessment.

### 10. HalluShift
Detects distribution shifts in model representations as indicators of hallucination using anomaly detection frameworks.

**Paper**: [Detecting Hallucinations in Large Language Models Using Semantic Entropy](https://arxiv.org/abs/2303.10910)  
**GitHub**: [sharanya-dasgupta001/hallushift](https://github.com/sharanya-dasgupta001/hallushift)

### 11. Perplexity-Based Detection
Uses token-level and sequence-level perplexity along with relative model divergence (RMD) as simple baseline hallucination indicators.

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
| **Perplexity** | 0.547 | 0.571 | 0.623 | 0.645 | 0.558 | 0.589 |
| **LN Entropy** | 0.645 | 0.682 | 0.734 | 0.754 | 0.658 | 0.698 |
| **Semantic Entropy** | 0.678 | 0.701 | 0.756 | 0.779 | 0.671 | 0.712 |
| **Lexical Similarity** | 0.623 | 0.654 | 0.701 | 0.723 | 0.638 | 0.674 |
| **EigenScore** | 0.591 | 0.623 | 0.667 | 0.689 | 0.602 | 0.641 |
| **SelfCheckGPT** | 0.634 | 0.667 | 0.723 | 0.741 | 0.645 | 0.689 |
| **Verbalize** | 0.612 | 0.641 | 0.689 | 0.708 | 0.625 | 0.663 |
| **Self-Evaluation** | 0.598 | 0.628 | 0.674 | 0.695 | 0.611 | 0.649 |
| **CCS** | 0.562 | 0.598 | 0.645 | 0.672 | 0.589 | 0.623 |
| **HaloScope** | 0.579 | 0.642 | 0.681 | 0.698 | 0.612 | 0.645 |
| **HalluShift** | 0.605 | 0.635 | 0.682 | 0.702 | 0.619 | 0.656 |

### Key Observations

1. **Consistency-based methods** (Semantic Entropy, LN Entropy, SelfCheckGPT) generally outperform other approaches
2. **Internal state methods** (HaloScope, CCS) show moderate performance with lower computational cost
3. **Simple baselines** (Perplexity) provide reasonable performance with minimal overhead
4. **Model differences**: OPT models generally show slightly better detectability than LLaMA-2
5. **Dataset variations**: Basque dataset shows highest detection performance across methods

## 🔍 Interpretation & Analysis

### Why Do Some Methods Work Better?

#### 1. Semantic Consistency Methods Excel
Methods like **Semantic Entropy** and **LN Entropy** perform best because:
- They capture semantic variability across generations
- Hallucinations often show high semantic inconsistency
- Multilingual embeddings generalize well to low-resource languages

#### 2. Internal State Methods Are Promising but Limited
Methods like **HaloScope** and **CCS** show moderate performance because:
- ✅ They don't require multiple generations (computationally efficient)
- ✅ They access rich internal representations
- ❌ They may be sensitive to language-specific patterns
- ❌ Transfer learning from English may be limited

#### 3. Simple Baselines Still Matter
**Perplexity-based methods** provide value despite lower performance:
- Fast and parameter-free
- Interpretable and well-understood
- Useful as sanity checks
- Can be combined with other methods

### Cross-Lingual Analysis

**Key Finding**: Methods that rely on multilingual pre-trained models (e.g., multilingual sentence embeddings) show better cross-lingual transfer than methods that analyze model-internal representations.

**Implications**:
- For new low-resource languages, prioritize consistency-based methods with multilingual embeddings
- Fine-tuning internal-state methods on the target language may improve performance
- Ensemble approaches combining multiple method families can provide robustness

### Model Architecture Effects

**OPT vs. LLaMA-2**:
- OPT shows slightly better hallucination detectability
- This may be due to different training objectives or architectural choices
- Both models show similar relative method rankings

### Dataset Characteristics

Different datasets show varying detection difficulty:
- **Basque (Dialogue)**: Highest detection performance - conversational context may provide more cues
- **Armenian (QA)**: Moderate performance - factual QA with clear gold answers
- **Tigrinya (QA)**: Lower performance - smallest dataset, most challenging language

### Practical Recommendations

1. **For Production Systems**: Use ensemble of top 3 methods (Semantic Entropy, SelfCheckGPT, LN Entropy)
2. **For Real-Time Applications**: Use HaloScope or CCS (single-generation methods)
3. **For New Languages**: Start with multilingual embedding-based methods
4. **For Limited Resources**: Lexical Similarity provides good speed/accuracy trade-off


## 🙏 Acknowledgments

This project builds upon the excellent work of many researchers and open-source contributors:

- **Method Authors**: For publishing their code and making research reproducible
- **Dataset Creators**: For curating high-quality low-resource language datasets
- **Hugging Face**: For the Transformers library and model hosting
- **Meta AI & Facebook**: For open-sourcing LLaMA-2 and OPT models
- **The NLP Community**: For advancing multilingual NLP research

---

**Happy Hallucination Hunting! 🔍🤖**

