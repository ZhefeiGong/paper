# Large Language Model


**📋Catalogue**
* [Review](#review)


**🔬Resources**
* [Standard](#standard)

**📚Knowledge**
* 🪐**LoRA(Low-Rank Adaptation)**
	LoRA, short for "Low-Rank Adaptation," is a technique designed to fine-tune large pre-trained models efficiently by introducing **low-rank matrix decomposition**. It significantly reduces the number of parameters that need to be updated during the fine-tuning process, thereby decreasing computational and storage costs while maintaining model performance.
* 🪐**Encoder** $\Rightarrow$ **BERT**(Bidirectional Encoder Representations from Transformers)
	* Masked Language Model, MLM
	* Next Sentence Prediction, NSP
* 🪐**Decoder** $\Rightarrow$ **GPT**(Generative Pre-trained Transformer)
	* Auto-regressive Training Process
* 🪐**Components of Transformer**
	a. **Input Embeddings**
	- **Token Embeddings** 
		Converts input tokens into dense vectors.
	- **Positional Encodings** 
		Adds positional information to the token embeddings since the model does not inherently capture word order. 
	b. **Encoder-Decoder Structure**
	- The Transformer consists of an encoder and a decoder stack, each comprising multiple identical layers.
	c. **Encoder**
	- **Self-Attention Mechanism**
		Computes the attention scores between each pair of input tokens to capture dependencies regardless of their distance.
    - **Scaled Dot-Product Attention**
	    Calculates the attention weights using dot products, scaling by the square root of the dimension.
    - **Multi-Head Attention**
	    Enhances the model's ability to focus on different positions by using multiple attention heads.
    - **Feed-Forward Neural Networks (FFNN)**
	    Applies two linear transformations with a ReLU activation in between.
	- **Layer Normalization and Residual Connections**
		Stabilizes and speeds up training.
	d. **Decoder**
	- **Masked Self-Attention** 
		Ensures that each position can only attend to earlier positions in the sequence, maintaining the autoregressive property necessary for text generation.
	* **Multi-Head Attention**
		Splits the attention mechanism into multiple heads to capture different features and dependencies in the text.
	- **Feed-Forward Neural Networks (FFNN)** 
		Each layer contains a position-wise feed-forward network, consisting of two linear transformations with a ReLU activation in between.

## Benchmark

| Date            | Title                                                                                        | Summary                                                                                                                                                                                                                                                                                                            | Links                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NeurIPS<br>2022 | **Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering** | <sub>Provide **ScienceQA** benchmark, a new benchmark that consists of **∼21k** multimodal multiple choice questions with diverse science topics and annotations of their answers with corresponding lectures and explanations, for learning the **chain of thought (CoT)** of LLM. 💫\|🌷\|❤️‍🔥\|👍🏻\|😉 </sub> | <div style='width:150px;'>[![arXiv](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2209.09513)</div><div style='width:150px;'>[![Blog](https://img.shields.io/badge/Blog-Website-yellow?logo=rss)](https://scienceqa.github.io/)</div><div style='width:150px;'>[![GitHub](https://img.shields.io/badge/GitHub-View-brightgreen?logo=github)](https://github.com/lupantech/ScienceQA)</div> |
|                 |                                                                                              |                                                                                                                                                                                                                                                                                                                    |                                                                                                                                                                                                                                                                                                                                                                                                                                    |


## LLM
| Date | Title                                                    | Summary                                                           | Links                                                                                                                                                                                                                                                                                         |
| ---- | -------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | **LLaMA: Open and Efficient Foundation Language Models** | <sub>Proposed **LLaMA**, an open-source LLM(GPT framework).</sub> | <div style='width:150px;'>[![arXiv](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2302.13971)</div><div style='width:150px;'>[![GitHub](https://img.shields.io/badge/GitHub-View-brightgreen?logo=github)](https://github.com/meta-llama/llama)</div> |
|      |                                                          |                                                                   |                                                                                                                                                                                                                                                                                               |


## Review

| Date     | Title                                                                           | Summary                                                                                                                                                                                | Links                                                                                                                                       |
| -------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Mar/2023 | **Foundation Models for Decision Making: Problems, Methods, and Opportunities** | <sub>(1) **Foundation Models as Conditional Generative Models**. (2) **Foundation Models as Representation Learners**. (3) **Large Language Models as Agent and Environments**. </sub> | <div style='width:150px;'>[![arXiv](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://arxiv.org/abs/2303.04129)</div> |
|          |                                                                                 |                                                                                                                                                                                        |                                                                                                                                             |
|          |                                                                                 |                                                                                                                                                                                        |                                                                                                                                             |




## Standard

| Date             | Title                                                                                                          | Summary                                                                                                                                                                       | Links                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7/4/23<br>ICML23 | **Rejection Improves Reliability: Training LLMs to Refuse Unknown Questions Using RL from Knowledge Feedback** | <sub> Unifying Count-Based Exploration and Intrinsic MotivationUnifying Count-Based Exploration and Intrinsic MotivationUnifying Count-Based Exploration and Intrinsic </sub> | <div style='width:150px;'>[![arXiv](https://img.shields.io/badge/arXiv-Paper-%23D2691E?logo=arxiv)](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf)</div><div style='width:150px;'>[![GitHub](https://img.shields.io/badge/GitHub-View-brightgreen?logo=github)](https://github.com/openai/weak-to-strong)</div><div style='width:150px;'>[![Blog](https://img.shields.io/badge/Blog-Posts-yellow?logo=rss)](https://mp.weixin.qq.com/s/f6YW-CxnLhnfMWTLg4M4Cw)</div><div style='width:150px;'>[![Note](https://img.shields.io/badge/Note-Read-blue?logo=dependabot)](summary/2024-03/2403.18349.md)</div> |
|                  |                                                                                                                |                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
