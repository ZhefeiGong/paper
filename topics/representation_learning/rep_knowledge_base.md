


## 🧀 Random Keywords

* **InfoNCE** / Information Noise-Contrastive Estimation
	It is a crucial objective function used in contrastive learning. It aims to maximize the mutual information between paired data samples, such as an image and its corresponding text description, while minimizing the mutual information between non-paired samples. 
	Mathematically, given a set of positive samples $(x, x^+)$ and a set of negative samples $(x, x^-)$, the InfoNCE loss is defined as: $L_{InfoNCE}​=−log\frac{exp(sim(x,x^+)/τ)}{\sum_{i=1}^{N}exp(sim(x,x^i)/τ)}$

