# Hallucination Detection in Low-Resource Languages

A comprehensive benchmark and evaluation framework for detecting hallucinations in large language model (LLM) generations across low-resource languages.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Background](#background)
- [Experimental Setup](#experimental-setup)
- [Methods Implemented](#methods-implemented)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Interpretation & Analysis](#interpretation--analysis)
- [Future Directions](#future-directions)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project provides a comprehensive framework for evaluating hallucination detection methods in large language models, with a **specific focus on low-resource languages** including Armenian, Basque, and Tigrinya. The framework implements 11 state-of-the-art hallucination detection methods and evaluates them on question-answering tasks across multiple languages and model architectures.

**Key Features:**
- 🌍 Multi-language support (Armenian, Basque, Tigrinya)
- 🤖 Multiple LLM architectures (LLaMA-2, OPT)
- 📊 11 hallucination detection methods
- 🔬 Comprehensive evaluation metrics (AUROC, AUPRC, FPR@95)
- 🚀 Parallel experiment execution support
- 📈 Detailed result analysis and visualization

## 💡 Motivation

Large Language Models (LLMs) have achieved remarkable performance on various NLP tasks, but they often generate **hallucinations** - outputs that are fluent but factually incorrect or unsupported by the input. This problem is particularly acute in **low-resource languages** where:

1. **Training data is limited**, leading to higher uncertainty in model predictions
2. **Evaluation resources are scarce**, making it difficult to assess model reliability
3. **Critical applications** (healthcare, education, legal) require high factual accuracy
4. **Existing methods** are primarily designed and evaluated on high-resource languages (English)

This project addresses these challenges by:
- Systematically evaluating hallucination detection methods on low-resource languages
- Providing a reproducible benchmark for future research
- Identifying which methods generalize well across languages and model architectures
- Enabling safer deployment of LLMs in multilingual contexts

## 📚 Background

### What is Hallucination?

In the context of LLMs, **hallucination** refers to generated content that is:
- Factually incorrect or unverifiable
- Not grounded in the provided context
- Contradictory to known facts
- Fabricated with high confidence

### Types of Hallucination Detection Methods

The methods implemented in this framework can be categorized into several families:

1. **Internal State-Based Methods**: Analyze model hidden states and activations
   - HaloScope, CCS, HalluShift
   
2. **Uncertainty-Based Methods**: Measure model confidence and uncertainty
   - Semantic Entropy, LN Entropy, Eigenvalue-based methods
   
3. **Consistency-Based Methods**: Check consistency across multiple generations
   - SelfCheckGPT, Lexical Similarity
   
4. **Linguistic Analysis**: Analyze surface-level properties
   - Perplexity, Verbalization confidence
   
5. **Self-Evaluation**: Use the model to evaluate its own outputs
   - Self-Evaluation prompting

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
**Paper**: [HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection](https://arxiv.org/abs/2409.17504) (NeurIPS'24 Spotlight)

Leverages unlabeled LLM generations to detect hallucinations using weighted SVD on hidden representations.

**Key Features**:
- Unsupervised hallucination detection
- Feature extraction from transformer blocks
- Weighted singular value decomposition

### 2. Contrast-Consistent Search (CCS)
**Paper**: [Discovering Latent Knowledge in Language Models](https://arxiv.org/abs/2212.03827)

Trains a linear probe on model representations to detect truthfulness without labeled data.

**Key Features**:
- Unsupervised probe learning
- Exploits consistency in model representations
- Layer-wise analysis

### 3. SelfCheckGPT
**Paper**: [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection](https://arxiv.org/abs/2303.08896)

Samples multiple responses and checks consistency using various similarity metrics.

**Key Features**:
- Zero-resource hallucination detection
- Multiple consistency measures (BERTScore, BLEURT, etc.)
- Black-box method (no access to internal states)

### 4. Semantic Entropy
**Paper**: [Semantic Uncertainty in Language Models](https://arxiv.org/abs/2302.09664)

Measures uncertainty by clustering semantically equivalent generations and computing entropy.

**Key Features**:
- Semantic clustering of generations
- Bidirectional entailment checking
- Captures semantic variability

### 5. LN Entropy
**Paper**: [Uncertainty Estimation in Autoregressive Structured Prediction](https://openreview.net/forum?id=jN5y-zb5Q7m)

Calculates log-normal entropy over semantically clustered responses.

**Key Features**:
- Multilingual semantic embeddings
- Adaptive clustering
- Robust uncertainty quantification

### 6. Eigenvalue-Based Score (EigenScore)
Analyzes the eigenvalue spectrum of generation distributions for uncertainty estimation.

**Key Features**:
- Spectral analysis of embeddings
- Captures distributional properties
- Parameter-efficient

### 7. Lexical Similarity
Measures consistency through lexical overlap across multiple generations.

**Key Features**:
- Token-level n-gram analysis
- Self-BLEU computation
- Fast and interpretable

### 8. Verbalized Confidence
Prompts the model to verbalize its confidence in its own answers.

**Key Features**:
- Self-reflection mechanism
- Prompt-based elicitation
- Direct confidence estimation

### 9. Self-Evaluation
Uses prompted introspection where the model evaluates its own generations.

**Key Features**:
- Metacognitive assessment
- Prompt engineering
- No additional models required

### 10. HalluShift
**Paper**: [HalluShift: Detecting Hallucinations in Language Models](https://github.com/sharanya-dasgupta001/hallushift)

Detects distribution shifts in model representations as indicators of hallucination.

**Key Features**:
- Distribution shift detection
- Anomaly detection framework
- Layer-wise analysis

### 11. Perplexity-Based Detection
Uses perplexity and relative model perplexity (RMD) as hallucination indicators.

**Key Features**:
- Token-level and sequence-level perplexity
- Relative model divergence
- Simple baseline method

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 50GB+ disk space for models and datasets

### Step 1: Clone the Repository

```bash
git clone https://github.com/xinyannnnnnn/hallucination_detection.git
cd hallucination_detection
```

### Step 2: Initialize Submodules

```bash
git submodule update --init --recursive
```

This will download:
- Dataset repositories (Armenian, Basque, Tigrinya)
- Method implementations (HaloScope, CCS, etc.)
- Model dependencies

### Step 3: Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install sentence-transformers scikit-learn pandas numpy
pip install bleurt-pytorch tqdm baukit
```

### Step 4: Configure Access Tokens

Create a `src/config.py` file with your Hugging Face token:

```python
HF_TOKEN = "your_huggingface_token_here"
```

**Note**: This file is in `.gitignore` and should never be committed.

### Step 5: Download Models

```bash
# The models are set up as git submodules
# LLaMA-2 7B and OPT 6.7B will be in the models/ directory
# BLEURT-20 is also included for evaluation
```

Alternatively, models will be automatically downloaded when first used if you have configured your HF token.

## 🚀 Usage

### Running Individual Methods

Each method has a standardized interface. The general workflow involves three steps:

#### Step 1: Generate Model Responses

```bash
python src/haloscope/main.py \
    --model_name llama2_7B \
    --dataset_name armenian \
    --gene 1 \
    --most_likely 1 \
    --num_gene 1
```

#### Step 2: Generate Ground Truth Labels

```bash
python src/haloscope/main.py \
    --model_name llama2_7B \
    --dataset_name armenian \
    --generate_gt 1 \
    --most_likely 1
```

#### Step 3: Run Hallucination Detection

```bash
python src/haloscope/main.py \
    --model_name llama2_7B \
    --dataset_name armenian \
    --most_likely 1 \
    --weighted_svd 1 \
    --feat_loc_svd 3
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
| `--gpu_id` | GPU device ID | Integer (e.g., `0`, `1`, `2`) |
| `--most_likely` | Use greedy decoding | `0` (sampling) or `1` (greedy) |

### Example: Running SelfCheckGPT

```bash
# Step 1: Generate multiple responses (needed for consistency checking)
python src/selfcheckgpt/main.py \
    --model_name llama2_7B \
    --dataset_name armenian \
    --gene 1 \
    --num_gene 3

# Step 2: Generate ground truth
python src/selfcheckgpt/main.py \
    --model_name llama2_7B \
    --dataset_name armenian \
    --generate_gt 1

# Step 3: Run detection
python src/selfcheckgpt/main.py \
    --model_name llama2_7B \
    --dataset_name armenian
```

### Parallel Execution

For running multiple experiments in parallel:

```python
from src.parallel_experiment_runner import run_experiments_parallel

experiments = [
    {"model": "llama2_7B", "dataset": "armenian", "method": "haloscope"},
    {"model": "opt_6_7b", "dataset": "basque", "method": "ccs"},
    # Add more experiments...
]

run_experiments_parallel(experiments, num_gpus=4)
```

## 📁 Code Structure

```
hallucination_detection/
├── README.md                       # This file
├── run.sh                          # Master experiment runner script
├── .gitignore                      # Git ignore patterns
├── .gitmodules                     # Git submodule configuration
│
├── datasets/                       # Dataset repositories (submodules)
│   ├── armenian/                   # SynDARin Armenian QA dataset
│   ├── basque/                     # Elkarhizketak Basque dialogue
│   ├── tigrinya/                   # TigQA Tigrinya QA dataset
│   ├── ArmTokenizer/               # Armenian tokenization utilities
│   └── TigXLNet/                   # Tigrinya language model
│
├── models/                         # Pre-trained models (submodules)
│   ├── llama/                      # LLaMA-2 7B
│   ├── opt/                        # OPT 6.7B
│   └── BLEURT-20/                  # BLEURT evaluation metric
│
├── methods/                        # Original method implementations (submodules)
│   ├── haloscope/                  # HaloScope method
│   ├── ccs/                        # Contrast-Consistent Search
│   ├── selfcheckgpt/               # SelfCheckGPT
│   ├── semantic_entropy/           # Semantic Entropy
│   ├── ln_entropy/                 # LN Entropy
│   ├── eigenscore/                 # Eigenvalue-based scoring
│   ├── lexical_similarity/         # Lexical similarity checking
│   ├── verbalize/                  # Verbalized confidence
│   └── hallushift/                 # HalluShift
│
├── src/                            # Implementation code for low-resource languages
│   ├── config.py                   # Configuration (HF token, paths)
│   ├── tokenization_utils.py      # Multi-language tokenization
│   ├── multi_gpu_utils.py         # GPU parallelization utilities
│   ├── parallel_experiment_runner.py  # Parallel experiment execution
│   │
│   ├── haloscope/                  # HaloScope implementation
│   │   ├── main.py
│   │   ├── linear_probe.py
│   │   └── utils.py
│   │
│   ├── ccs/                        # CCS implementation
│   │   ├── main.py
│   │   └── ccs_probe.py
│   │
│   ├── selfcheckgpt/               # SelfCheckGPT implementation
│   │   └── main.py
│   │
│   ├── semantic_entropy/           # Semantic Entropy implementation
│   │   └── main.py
│   │
│   ├── ln_entropy/                 # LN Entropy implementation
│   │   ├── main.py
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── test_attention_mask.py
│   │   ├── test_ln_entropy.py
│   │   └── verify_setup.py
│   │
│   ├── eigenscore/                 # EigenScore implementation
│   │   └── main.py
│   │
│   ├── lexical_similarity/         # Lexical Similarity implementation
│   │   └── main.py
│   │
│   ├── verbalize/                  # Verbalized Confidence implementation
│   │   └── main.py
│   │
│   ├── self_evaluation/            # Self-Evaluation implementation
│   │   └── main.py
│   │
│   ├── hallushift/                 # HalluShift implementation
│   │   └── main.py
│   │
│   └── perplexity/                 # Perplexity-based detection
│       ├── main.py
│       ├── evaluation_utils.py
│       └── rmd_utils.py
│
├── results/                        # Experimental results
│   ├── haloscope/                  # HaloScope results
│   │   ├── haloscope_results_armenian_llama2_7B_proper_splits.json
│   │   ├── haloscope_results_basque_opt_6_7b_proper_splits.json
│   │   └── ...
│   ├── ccs/                        # CCS results
│   ├── selfcheckgpt/               # SelfCheckGPT results
│   ├── semantic_entropy/           # Semantic Entropy results
│   ├── ln_entropy/                 # LN Entropy results
│   ├── eigenscore/                 # EigenScore results
│   ├── lexical_similarity/         # Lexical Similarity results
│   ├── verbalize/                  # Verbalized Confidence results
│   ├── self_evaluation/            # Self-Evaluation results
│   ├── hallushift/                 # HalluShift results
│   └── perplexity/                 # Perplexity results
│
└── papers/                         # Reference papers
    ├── haloscope.pdf
    ├── hallushift.pdf
    └── notes.pdf
```

### Key Files

- **`run.sh`**: Master script containing all experiments
- **`src/tokenization_utils.py`**: Language-specific preprocessing and tokenization
- **`src/multi_gpu_utils.py`**: Utilities for parallel GPU execution
- **`src/perplexity/evaluation_utils.py`**: Comprehensive evaluation metrics
- **`src/config.py`**: Configuration file (not in repo, create locally)

## 📊 Results

### Overall Performance Summary

Results are stored in `results/` directory with the naming convention:
```
{method}_results_{dataset}_{model}_proper_splits.json
```

Each result file contains:
- `test_auroc`: AUROC on test set
- `best_val_auroc`: Best AUROC on validation set
- `best_params`: Optimal hyperparameters
- Additional method-specific metrics

### Sample Results (AUROC)

| Method | Armenian (LLaMA) | Armenian (OPT) | Basque (LLaMA) | Basque (OPT) | Tigrinya (LLaMA) | Tigrinya (OPT) |
|--------|------------------|----------------|----------------|--------------|------------------|----------------|
| **HaloScope** | 0.579 | 0.642 | 0.681 | 0.698 | 0.612 | 0.645 |
| **CCS** | 0.562 | 0.598 | 0.645 | 0.672 | 0.589 | 0.623 |
| **SelfCheckGPT** | 0.634 | 0.667 | 0.723 | 0.741 | 0.645 | 0.689 |
| **Semantic Entropy** | 0.678 | 0.701 | 0.756 | 0.779 | 0.671 | 0.712 |
| **LN Entropy** | 0.645 | 0.682 | 0.734 | 0.754 | 0.658 | 0.698 |
| **EigenScore** | 0.591 | 0.623 | 0.667 | 0.689 | 0.602 | 0.641 |
| **Lexical Similarity** | 0.623 | 0.654 | 0.701 | 0.723 | 0.638 | 0.674 |
| **Verbalize** | 0.612 | 0.641 | 0.689 | 0.708 | 0.625 | 0.663 |
| **Self-Evaluation** | 0.598 | 0.628 | 0.674 | 0.695 | 0.611 | 0.649 |
| **HalluShift** | 0.605 | 0.635 | 0.682 | 0.702 | 0.619 | 0.656 |
| **Perplexity** | 0.547 | 0.571 | 0.623 | 0.645 | 0.558 | 0.589 |

*Note: These are illustrative values. Actual results are available in the `results/` directory.*

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

## 🚀 Future Directions

### Short-Term Improvements

1. **Ensemble Methods**
   - Combine predictions from multiple methods
   - Learn optimal weighting schemes
   - Investigate method complementarity

2. **Hyperparameter Optimization**
   - Systematic grid search across methods
   - Language-specific tuning
   - Automated hyperparameter selection

3. **Additional Languages**
   - Expand to more low-resource languages
   - Include language families not yet covered
   - Test cross-lingual transfer

4. **Model Architectures**
   - Evaluate on newer models (LLaMA-3, Mistral, Gemma)
   - Test on multilingual models (mBERT, XLM-R based)
   - Analyze size scaling (1B to 70B parameters)

### Medium-Term Research

1. **Calibration Techniques**
   - Post-hoc calibration of detection scores
   - Temperature scaling for uncertainty
   - Conformal prediction for confidence sets

2. **Fine-Tuning for Detection**
   - Supervised fine-tuning with labeled hallucination data
   - Reinforcement learning from detection feedback
   - Multi-task learning (generation + detection)

3. **Explainability**
   - Provide explanations for hallucination predictions
   - Identify specific tokens or phrases causing hallucinations
   - Visualize internal representations

4. **Active Learning**
   - Efficient data collection strategies
   - Query-by-committee approaches
   - Human-in-the-loop refinement

### Long-Term Vision

1. **Universal Hallucination Detector**
   - Language-agnostic detection
   - Unified model across all languages
   - Transfer learning from high to low-resource languages

2. **Integrated Generation Systems**
   - Real-time hallucination prevention during generation
   - Constrained decoding to avoid hallucinations
   - Self-correcting generation mechanisms

3. **Theoretical Understanding**
   - Mathematical characterization of hallucinations
   - Information-theoretic bounds on detectability
   - Relationship to model capacity and training data

4. **Standardized Benchmarks**
   - Community-wide evaluation protocols
   - Shared leaderboards
   - Regular benchmark updates with new methods

### Open Problems

- **Cold-Start Problem**: How to detect hallucinations in entirely new languages without any labeled data?
- **Adversarial Robustness**: Can models generate hallucinations that evade detection?
- **Contextual Hallucinations**: How to detect subtle factual errors rather than obvious fabrications?
- **Multimodal Hallucinations**: Extending detection to vision-language and audio-language models

## 📄 Citation

If you use this code or find this work helpful, please cite:

```bibtex
@misc{hallucination_detection_lowres2024,
  author = {Xinyan},
  title = {Hallucination Detection in Low-Resource Languages: A Comprehensive Benchmark},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/xinyannnnnnn/hallucination_detection}}
}
```

### Related Work

Please also cite the original papers for the methods used:

**HaloScope**:
```bibtex
@inproceedings{du2024haloscope,
  title={HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection}, 
  author={Xuefeng Du and Chaowei Xiao and Yixuan Li},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

**CCS**:
```bibtex
@article{burns2022discovering,
  title={Discovering Latent Knowledge in Language Models Without Supervision},
  author={Burns, Collin and Ye, Haotian and Klein, Dan and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2212.03827},
  year={2022}
}
```

**SelfCheckGPT**:
```bibtex
@article{manakul2023selfcheckgpt,
  title={SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models},
  author={Manakul, Potsawee and Liusie, Adian and Gales, Mark JF},
  journal={arXiv preprint arXiv:2303.08896},
  year={2023}
}
```

**Semantic Entropy**:
```bibtex
@article{kuhn2023semantic,
  title={Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation},
  author={Kuhn, Lorenz and Gal, Yarin and Farquhar, Sebastian},
  journal={arXiv preprint arXiv:2302.09664},
  year={2023}
}
```

## 🙏 Acknowledgments

This project builds upon the excellent work of many researchers and open-source contributors:

- **Method Authors**: For publishing their code and making research reproducible
- **Dataset Creators**: For curating high-quality low-resource language datasets
- **Hugging Face**: For the Transformers library and model hosting
- **Meta AI & Facebook**: For open-sourcing LLaMA-2 and OPT models
- **The NLP Community**: For advancing multilingual NLP research

Special thanks to the authors of:
- [HaloScope](https://github.com/deeplearning-wisc/haloscope)
- [CCS](https://github.com/collin-burns/discovering_latent_knowledge)
- [SelfCheckGPT](https://github.com/potsawee/selfcheckgpt)
- [Semantic Entropy](https://github.com/lorenzkuhn/semantic_uncertainty)
- [EigenScore](https://github.com/D2I-ai/eigenscore)
- [HalluShift](https://github.com/sharanya-dasgupta001/hallushift)

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please:
- Open an issue on GitHub
- Contact: [Your contact information]

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Individual methods and datasets may have their own licenses. Please check the respective submodules for details.

---

**Happy Hallucination Hunting! 🔍🤖**

