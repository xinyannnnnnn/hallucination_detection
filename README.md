# Hallucination Detection in Low-Resource Languages

A systematic benchmark and evaluation of 11 state-of-the-art hallucination detection methods across 3 typologically diverse low-resource languages

## 🎯 Overview

This project provides a comprehensive framework for evaluating hallucination detection methods in large language models, with a **specific focus on low-resource languages** including Armenian, Basque, and Tigrinya. The framework implements 11 state-of-the-art hallucination detection methods and evaluates them on question-answering tasks across multiple languages and model architectures. Our findings reveal that current hallucination detection methods are not language-agnostic and face fundamental challenges when applied to low-resource languages.

**Key Features:**
- 🌍 Multi-language support: Armenian, Basque, Tigrinya
- 🤖 Multiple LLM architectures: LLaMA-2 7B, OPT 6.7B
- 📊 11 hallucination detection methods: Internal state, consistency-based, uncertainty quantification, confidence elicitation
- 🔬 Comprehensive evaluation: AUROC metrics across 66 experimental configurations
- 🚀 Reproducible framework: Complete codebase with standardized evaluation pipeline

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
| **SynDARin** | Armenian | QA | 1324 | Question Answering |
| **Elkarhizketak** | Basque | Dialogue | 1698 | Dialogue Understanding |
| **TigQA** | Tigrinya | QA | 176 | Question Answering |

### Evaluation Metrics

- **AUROC** (Area Under ROC Curve): Primary metric for discriminative performance
- **Ground Truth** BLEURT-based labeling with 0.5 threshold (answers > 0.5 = truthful, ≤ 0.5 = hallucinated)

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

### Critical Findings

1. **Complete Detection Failure on Tigrinya**: All 11 methods achieve exactly 50% AUROC (random chance), indicating fundamental challenges in hallucination detection for extremely low-resource languages.

2. **Surprising Performance of Internal State Methods**: HaloScope and Self-Evaluation outperform consistency-based approaches, contradicting assumptions about cross-lingual transfer effectiveness.

3. **Extreme Method Variability**: Performance varies dramatically across languages and models, with no universal best approach.

4. **Language Resource Hierarchy**: Armenian (moderate success) > Basque (moderate but variable) > Tigrinya (complete failure).

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

## 🙏 Acknowledgments

This project builds upon the excellent work of many researchers and open-source contributors:

- **Method Authors**: For publishing their code and making research reproducible
- **Dataset Creators**: For curating high-quality low-resource language datasets
- **Hugging Face**: For the Transformers library and model hosting
- **Meta AI & Facebook**: For open-sourcing LLaMA-2 and OPT models
- **The NLP Community**: For advancing multilingual NLP research

---

**Happy Hallucination Hunting! 🔍🤖**

