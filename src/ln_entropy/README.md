# LN Entropy Method for Hallucination Detection

## 概述 (Overview)

LN Entropy（对数正态熵）是一种基于语义聚类的序列级不确定性估计方法，用于检测大型语言模型生成内容中的幻觉。该方法来源于论文 "Uncertainty Estimation in Autoregressive Structured Prediction"。

LN Entropy is a sequence-level uncertainty estimation method based on semantic clustering for detecting hallucinations in large language model generations. This method is based on the paper "Uncertainty Estimation in Autoregressive Structured Prediction".

## 方法原理 (Methodology)

### 核心思想 (Core Idea)
1. **多样化生成** (Diverse Generation): 对同一问题生成多个响应
2. **语义聚类** (Semantic Clustering): 使用语义嵌入对响应进行聚类
3. **熵计算** (Entropy Calculation): 基于聚类分布计算对数正态熵

### 详细流程 (Detailed Process)

1. **输入** (Input): 问题/提示 + 多个生成的响应
2. **语义嵌入** (Semantic Embedding): 
   - 使用预训练模型（SentenceTransformer或BERT）获取响应的语义表示
   - 支持多语言嵌入，适用于低资源语言
3. **聚类分析** (Clustering Analysis):
   - 计算响应间的余弦相似度
   - 使用层次聚类基于相似度阈值进行分组
4. **LN熵计算** (LN Entropy Calculation):
   - 基于聚类分布计算香农熵
   - 应用对数正态变换: LN_Entropy = log(1 + exp(Shannon_entropy))
5. **输出** (Output): LN熵值（越高表示不确定性越大）

## 使用方法 (Usage)

### 1. 生成响应 (Generate Responses)
```bash
python src/ln_entropy/main.py --model_name llama2_7B --dataset_name armenian --gene 1 --num_gene 10
```

### 2. 生成真值标签 (Generate Ground Truth)
```bash
python src/ln_entropy/main.py --model_name llama2_7B --dataset_name armenian --generate_gt 1
```

### 3. 运行分析 (Run Analysis)
```bash
python src/ln_entropy/main.py --model_name llama2_7B --dataset_name armenian
```

## 参数说明 (Parameters)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--similarity_threshold` | 0.7 | 聚类相似度阈值 (Clustering similarity threshold) |
| `--embedding_model` | 'paraphrase-multilingual-MiniLM-L12-v2' | 多语言语义嵌入模型 (Multilingual sentence embedding model) |
| `--num_gene` | 10 | 每个问题生成的响应数量 (Number of responses per question) |
| `--thres_gt` | 0.5 | 真值标签阈值 (Ground truth threshold) |

## 关键特性 (Key Features)

1. **多语言支持** (Multilingual Support): 支持亚美尼亚语、巴斯克语、提格雷语等低资源语言
2. **自适应聚类** (Adaptive Clustering): 基于语义相似度自动确定聚类数量
3. **鲁棒性** (Robustness): 对模型和数据集具有良好的泛化能力
4. **可解释性** (Interpretability): 提供聚类信息，便于理解不确定性来源

## 实验结果 (Experimental Results)

LN Entropy方法在低资源语言幻觉检测任务上表现出色：
- 能够有效区分语义一致和不一致的响应
- 对于语义相似但表述不同的响应具有适中的不确定性评分
- 在AUROC指标上达到了竞争性的性能

## 依赖项 (Dependencies)

主要依赖项在 `requirements.txt` 中列出。**已安装 `sentence-transformers`** 以获得最佳的多语言语义嵌入效果，使用 `paraphrase-multilingual-MiniLM-L12-v2` 模型提供优秀的跨语言语义理解能力。

## 注意事项 (Notes)

1. **计算资源**: 语义嵌入计算可能需要较多GPU内存
2. **语言适配**: 对于某些低资源语言，可能需要调整相似度阈值
3. **响应质量**: 生成响应的多样性直接影响LN熵的有效性

## 引用 (Citation)

如果使用此实现，请引用原始论文：

```bibtex
@inproceedings{malinin2021structured,
  title={Uncertainty Estimation in Autoregressive Structured Prediction},
  author={Malinin, Andrey and Gales, Mark},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
