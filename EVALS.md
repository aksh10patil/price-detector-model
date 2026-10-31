# 📈 Evaluation Methodology & Benchmark Results

This document provides the complete empirical evaluation records, benchmark methodology, error distribution analyses, and qualitative findings for the **Price Prediction** project.

---

## 🏆 Full Benchmark Leaderboard

Evaluated on a standardized holdout test split of 200 Amazon products across consumer electronics, home appliances, and accessories.

| Rank | Model Architecture | Paradigm | RMSE ($) | MAE ($) | R² (%) | Error Within 20% | Relative Cost / Query |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Llama 3.2 3B (QLoRA Full)** | **Open-Source Fine-Tuning** | **$39.85** | **$21.40** | **85.9%** | **58.5%** | **$0.00 (Self-hosted)** |
| 🥈 | **GPT-6** | Frontier Zero-Shot | $44.74 | $25.10 | 82.2% | 51.0% | $$$$ (High) |
| 🥉 | **Deep Residual Network (PyTorch)** | Custom 10-Layer ResNet | $46.49 | $27.35 | 80.8% | 49.0% | $0.00 (CPU / Local) |
| 4 | Claude Sonnet 5 | Frontier Zero-Shot | $47.10 | $26.80 | 80.3% | 48.5% | $$$ (Moderate) |
| 5 | Gemini 4 Pro | Frontier Zero-Shot | $50.54 | $29.15 | 77.3% | 45.0% | $$$ (Moderate) |
| 6 | Grok 4.5 Fast | Frontier Zero-Shot | $57.62 | $33.40 | 70.6% | 41.0% | $$ (Low) |
| 7 | GPT-5 Nano | Frontier Zero-Shot | $62.51 | $36.20 | 65.4% | 36.5% | $ (Very Low) |
| 8 | Neural Network Baseline | Classical MLP | $63.97 | $37.50 | 63.8% | 35.0% | $0.00 |
| 9 | **Llama 3.2 3B (QLoRA Lite)** | **Open-Source Fine-Tuning** | **$65.40** | **$38.10** | **62.2%** | **34.0%** | **$0.00 (Free T4 Colab)** |
| 10 | XGBoost Regressor | Gradient Boosted Trees | $68.23 | $40.50 | 58.9% | 31.5% | $0.00 |
| 11 | Random Forest Regressor | Ensemble Trees | $72.28 | $43.20 | 53.8% | 28.0% | $0.00 |
| 12 | GPT-5 Nano (OpenAI Fine-Tuned) | Proprietary Fine-Tuning | $75.91 | $44.90 | 49.0% | 26.0% | $$ |
| 13 | NLP + Ridge Regression | Bag-of-Words TF-IDF | $76.81 | $46.10 | 47.8% | 24.5% | $0.00 |
| 14 | 🧑‍💼 **Human Expert (Ed)** | **Human Intuition Baseline** | **$87.62** | **$54.30** | **32.1%** | **19.0%** | ⏳ High Human Effort |
| 15 | Linear Regression | Tabular Metadata Features | $101.56 | $64.80 | 9.4% | 12.0% | $0.00 |
| 16 | Constant Mean Baseline | Naive Training Mean ($68.42) | $106.18 | $69.20 | 0.0% | 8.5% | $0.00 |
| 17 | Base Llama 3.2 3B (Untrained) | Base 4-bit Foundation | $110.72 | $74.50 | -8.8% | 6.0% | $0.00 |

---

## 🔬 Key Empirical Insights

### 1. Why QLoRA on a 3B Model Beat Trillion-Parameter Frontier LLMs
- **Domain Specialization vs Generic Priors**: Frontier models (GPT-6, Claude Sonnet 5) possess vast world knowledge, but their training loss is broad and unaligned to current Amazon pricing structures. They frequently overestimate or round to arbitrary psychological anchors ($99, $199).
- **Exact Number Token Distribution**: Fine-tuning Llama 3.2 3B with 4-bit QLoRA on thousands of direct prompt-completion pairs allowed the adapter matrices to calibrate output logits directly to realistic commercial distributions.

### 2. The Failure of Naive Proprietary Fine-Tuning (OpenAI)
- `gpt-5-nano` fine-tuned via the OpenAI Chat Fine-Tuning API achieved **$75.91 RMSE**, substantially worse than its zero-shot counterpart ($62.51 RMSE).
- **Explanation**:
  - The chat completion objective minimizes cross-entropy on character tokens rather than numerical regression loss (MSE/Huber).
  - Fine-tuning with small datasets without strict format guidance resulted in mode collapse around typical price ranges, penalizing extreme values.

### 3. Deep Residual Networks as a Strong Offline Alternative
- The PyTorch 10-layer ResNet utilizing `HashingVectorizer` (100,000 sparse text features) reached **$46.49 RMSE**, outperforming Claude Sonnet 5 ($47.10) and Gemini 4 Pro ($50.54).
- **Advantage**: Zero GPU requirement during inference, ~2ms CPU prediction latency, and 0 external API dependencies.

---

## 📊 Evaluation Metrics Definitions

1. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
   Penalizes large outlier misses heavily; the primary competition metric.

2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   Measures the average dollar magnitude of errors.

3. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   Percentage of price variance explained by the model compared to predicting the dataset mean.

4. **Error Tolerance Bands (Within 10% / 20%)**:
   Fraction of predictions satisfying:
   $$\frac{|\hat{y}_i - y_i|}{\max(y_i, 1.0)} \le 0.10 \quad \text{or} \quad 0.20$$
