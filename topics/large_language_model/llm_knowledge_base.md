

## 📚 Random Keywords

* 🪐**LoRA(Low-Rank Adaptation)**
	
	LoRA, short for "Low-Rank Adaptation," is a technique designed to fine-tune large pre-trained models efficiently by introducing **low-rank matrix decomposition**. It significantly reduces the number of parameters that need to be updated during the fine-tuning process, thereby decreasing computational and storage costs while maintaining model performance.

* 🪐**Task-Agnostic Web-Scale Pre-Training** in NLP
	
	**Task-agnostic web-scale pre-training** is a technique in NLP where large language models are trained on vast amounts of diverse text data from the internet, without targeting any specific task. 
	
	- **Task-Agnostic**: The model is not pre-trained for any specific application, allowing it to be versatile and adaptable to various tasks.
	- **Web-Scale**: Training involves massive datasets sourced from the web, encompassing a wide range of language uses and domains.
	- **Pre-Training**: The model learns general language features during this phase, which are later fine-tuned on smaller, task-specific datasets for optimal performance.
	
	This approach enhances the model's generalization, efficiency, and overall performance across multiple NLP tasks.

* 🪐ZeRO (Zero Redundancy Optimizer) 
	ZeRO is a technology designed to optimize memory usage in training large-scale deep learning models. It does this by **partitioning model states, gradients, and optimizer states** across multiple devices (GPUs). 
	ZeRO is divided into different stages:
	1. **ZeRO-1**: **Partitions optimizer states** (e.g., Adam's moment estimates) across devices to reduce memory usage.
	2. **ZeRO-2**: In addition to optimizer states, it **partitions gradients**, further reducing memory overhead.
	3. **ZeRO-3**: Extends memory optimization by **partitioning model parameters**, achieving near-linear memory savings with the number of devices used.
	These optimizations enable training larger models that would otherwise be impossible to fit into the memory of individual GPUs.

* 🪐Deepseed
	

* 🪐MoE (Mixture of Expert)
	



## 🦾Transformer

* 🪐**Encoder** $\Rightarrow$ **BERT**(Bidirectional Encoder Representations from Transformers)
	
	* Masked Language Model, MLM
	* Next Sentence Prediction, NSP

* 🪐**Decoder** $\Rightarrow$ **GPT**(Generative Pre-trained Transformer)
	
	* Auto-regressive Training Process

* 🪐**Components of Transformer**🪐
	
	[Youtube](https://www.youtube.com/watch?v=wjZofJX0v4M) | 
	
	a. **Input Embeddings**
	
	- **Token Embeddings** 
		
		Converts input tokens into dense vectors.
		
	- **Positional Encodings** 
		
		Adds positional information to the token embeddings since the model does not inherently capture word order. 
	
	b. **Encoder-Decoder Structure**
	
	- The Transformer consists of an encoder and a decoder stack, each comprising multiple identical layers.
	
	c. **😊Encoder😊**
	
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
	
	d. **😊Decoder😊**
	
	- **Masked Self-Attention** 
		
		Ensures that each position can only attend to earlier positions in the sequence, maintaining the autoregressive property necessary for text generation.
		
	* **Multi-Head Attention** 
		
		Splits the attention mechanism into multiple heads to capture different features and dependencies in the text.
		
		* 🔥Multi-Head🔥
			
	- **Feed-Forward Neural Networks (FFNN)** 
		
		Each layer contains a position-wise feed-forward network, consisting of two linear transformations with a ReLU activation in between.

* 🪐 - 🔥**All of the Matrices in Transformer** | 8 Categories🔥	
	
	[3b1b](https://www.youtube.com/watch?v=wjZofJX0v4M&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5)
	
	------------------------------------
	
	* | Embedding |
		* one column represents one word (or one token)
		* words $\Rightarrow$ vectors
		* $W_E = d_{embed} \times n_{vocab}$
	
	------------------------------------
	
	* | Query |
		* Query : Any adjectives in front of me?
		* $W_Q \times \vec{E} = \vec{Q}$
		* $W_Q = d_{qk} \times d_{embed}$
	* | Key |
		* Key : I'm an adjective! I'm here! 
		* $W_K \times \vec{E}=\vec{K}$
		* $W_K = d_{qk} \times d_{embed}$
		* 🔥**Key-Query Pair** | **Attention Pattern**🔥
			* try to map the $\vec{K}$ into the same direction and length in the Query/Key space
				* "key is attended to query"
			* act like a weight
			* 🔥**Masking**🔥
				* the Query only concerns the **Keys** before its position
	* | Value | 
		* $W_V = d_{embed} \times d_{embed}$ 
			* $\rightarrow$ "Low Rank" Transformation 
			* $\rightarrow$ $d_{embed}\times d_{qk}$ + $d_{qk} \times d_{embed}$
			* $=Value_{\uparrow} \times Value_{\downarrow}$
		* 🔥$V(K^TQ)$🔥
		* Multi-Head 
			* Practically, inside each head, use only the value-down matrix($Value_{\downarrow}$)
				* and put all of the value-up matrices($Value_{\uparrow}$) together into Output Matrix($W_O$), in order to compress things into a single matrix multiplication
				* so truly, $W_V = d_{qk} \times d_{embed}$
			* $n_{head}$ times ~~ $V(K^TQ)$ process
	* | Output |
		* $W_O=d_{embed}\times(d_{qk}\times n_{head})$
	
	------------------------------------
	
	* | Up-projection | 
		* 
	* | Down-projection | 
		* 
	
	------------------------------------
	
	* | Unembedding |
		* $W_U = n_{vocab} \times d_{embed}$
		* procedure 
			* first, utilize the unembedding matrix to map the last vector into a list of vocabularies probabilities
			* then use Softmax layer(with temperature) to choose the one with highest probability
				* Softmax
					* **Logits** $\Rightarrow$ **Probabilities**
	
------------------------------------

