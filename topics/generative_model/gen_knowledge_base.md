

## 🧀 Mathematical Theory


### 🍰 VAE

* #### ✨**Variational Auto-Encoder(VAE)**
	* 🌟The **neural net** perspective 
		* Encoder : $q_{\theta}(z|x)$
			* a hidden representation $𝑧$
			* the encoder $q_{\theta}​(z|x)$ outputs  is a Gaussian probability density. we can sample from this distribution to get noisy values of the representations $z$.
		* Decoder : $p_{\Phi}(x|z)$
			* we use the reconstruction log-likelihood $log⁡\ 𝑝_{\Phi}(𝑥|𝑧)$ whose units are nats. 
			* this measure tells us how effectively the decoder has learned to reconstruct an input image $x$ given its latent representation $z$.
		* Loss : $l_i(\theta,\phi)=-E_{z\sim q_{\theta}(z|x_i)}[log \ p_{\phi}(x_i|z)]+KL(q_{\theta}(z|x_i)||p(z))$
			* the **first** term encourages the decoder to learn to **reconstruct** the data.
			* the **second** term is a regularizer, which measures how much information is lost when using $𝑞$ to represent $𝑝$. It is a measure of how close $q$ is to $𝑝$.
			* $p$ is specified as **a standard Normal distribution** with mean zero and variance one, or $𝑝(𝑧)=\text{𝑁𝑜𝑟𝑚𝑎𝑙}(0,1)$
		* Gradient descent to optimize : $\theta \leftarrow \theta - \rho \frac{\partial l }{\partial \theta}$
			
	
	* 🌟The **probability model** perspective
		* the joint probability : $p(x,z)=p(x|z)p(z)$
			model
		* latent variables : $z_i\sim p(z)$
			prior
		* datapoint : $x_i \sim p(x|z)$
			likelihood
		* the goal is to infer good values of **the latent variables** given **observed data**, or to calculate the posterior $𝑝(𝑧|𝑥)$. Bayes says : $p(z|x)=\frac{p(x|z)p(z)}{p(x)}$
		* $p(x)$ is called the evidence, and we can calculate it by marginalizing out the latent variables : $p(x)=\int p(x|z)p(z)dz$
			* our **target** is to make $p(x)$ as **large** as possible
			* this integral requires exponential time to compute. we therefore need to approximate this posterior distribution.
		* **variational inference** approximates the posterior $p(z|x)$ with a family of distributions $𝑞_{\lambda}(𝑧|𝑥)$.
			* use **the Kullback-Leibler divergence**, which measures the information lost when using $𝑞$ to approximate $𝑝$. $KL(q_{\lambda}(z|x)||p(z|x))=𝐸_𝑞[log\ ⁡𝑞_{\lambda}(𝑧|𝑥)]−𝐸_𝑞[log⁡\ 𝑝(𝑥,𝑧)]+log\ p(x)$
			* our **target** is to find the variational parameters $𝜆$ that **minimize** this divergence.
			* due to the pesky evidence $p(x)$ which is intractable, we consider : $\text{ELBO}(\lambda)=E_{q}[log\ p(x,z)]-E_{q}[log \ q_{\lambda}(z|x)]$
			* combining the KL divergence, we can rewrite the evidence : $log\ p(x) = \text{ELBO}(\lambda)+ KL(q_{\lambda}(z|x)||p(z|x))$
			* try to **minimize** **the Kullback-Leibler divergence** between the approximate and exact posteriors. Instead, we can **maximize** the **ELBO** which is equivalent
		* we can rewrite the ELBO, through $p(x,z)=p(x|z)p(z)$. $\text{ELBO}(\lambda)=E_{q}[log\ p(x,z)]-E_{q}[log \ q_{\lambda}(z|x)]$ $= E_q[log\ p(x|z)]+E_q[log\ p(z)]-E_{q}[log \ q_{\lambda}(z|x)$ $=E_q[log\ p(x|z)]-(E_{q}[log \ q_{\lambda}(z|x)-E_q[log\ p(z)])$ $=E_q[log\ p(x|z)]-KL(q_{\lambda}(z|x)||p(z))$
		* **Maximizing** the $\text{ELBO}$ is equivalent to **Maximizing** the log-likelihood function of the observed data $log\ p(x)$
		* approximate posterior $𝑞_{\theta}(𝑧|𝑥,\lambda)$ with an _inference network_ (or encoder). parametrize the likelihood $𝑝_{\phi}(𝑥|𝑧)$ with a _generative network_ (or decoder)
			* we can rewrite the $\text{ELBO}$ as : $\text{ELBO}(\theta,\phi)=E_{q_{\theta}(z|x)}[log\ p_{\phi}(x|z)]-KL(q_{\theta}(z|x)||p(z))$
			* $\text{ELBO}​(\theta,\phi)=−l_i​(\theta,\phi)$, and we try to **maximize** the $\text{ELBO}$
			* gradient ascent on the $\text{ELBO}$
		* the term **variational inference** usually refers to **maximizing** the $\text{ELBO}$ with respect to the variational parameters $\lambda$.
			* This technique is called **variational EM (expectation maximization)**, because we are maximizing the expected log-likelihood of the data with respect to the model parameters.
		* We have followed the recipe for **variational inference**. We’ve defined:
			- **a probability model** $p_{\phi}(x|z)$ of latent variables and data
			- **a variational family** $q_{\theta}(z|x)$ for the latent variables to approximate our posterior
		- Representation Trick
			* The expectation term in the loss function invokes generating samples from $z\sim q_{\phi}(z|x)$. Sampling is a stochastic process and therefore we cannot back-propagate the gradient. 
			* To make it trainable, the reparameterization trick is introduced: It is often possible to express the random variable $z$ as a deterministic variable $z=T_{\phi}(x,\epsilon)$, where $\epsilon$ is an auxiliary independent random variable, and the transformation function $T_{\phi}$ parameterized by $\phi$ converts $\epsilon$ to $z$.
				* $z\sim q_{\phi}(z|x^{(i)}=N(z;\mu^{(i)},\sigma^{2(i)}I)$
				* $\Rightarrow$ $z=\mu+\sigma \odot \epsilon$, where $\epsilon \sim N(0,I)$
	
	* [reference1](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) | [reference2](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
	



* #### ✨ VQ-VAE
	* Vector Quantisation Variational AutoEncoder model learns a discrete latent variable by the encoder. 
	* Let $e\in R^{K\times D},i=1,…,K$ be the latent embedding space (also known as “**codebook**”) in VQ-VAE, where K is the number of latent variable categories and D is the embedding size. An individual embedding vector is $e^i\in R^D,i=1,…,K$.
	* The encoder output $E(x)=z_e$ goes through a nearest-neighbor lookup to match to one of $K$ embedding vectors and then this matched code vector becomes the input for the decoder $D$ : $z_q(x)=Quantize(E(x))=e_k$, where $k=arg \min_i ||E(x)-e_i||_2$
	* Because $arg\min()$ is non-differentiable on a discrete space, the gradients $\nabla_zL$ from decoder input $z_q$ is copied to the encoder output $z_e$.
	* **Loss**
		* **reconstruction loss** : $||x-D(e_k)||_2^2$
		* **vq loss**: The L2 error between the embedding space and the encoder outputs : $||sg[E(x)-e_k]||^2_2$
		* **commitment loss**: A measure to encourage the encoder output to stay close to the embedding space and to prevent it from fluctuating too frequently from one code vector to another : $\beta||E(x)-sg[e_k]||^2_2$
		
		1. the decoder $D$ optimizes the reconstruction loss term only
		2. the encoder $E$ optimizes the reconstruction and the commitment loss terms
		3. the embedding $e$ is optimized by the vq loss
	
	* Why don't we have a KL divergence like traditional VAE?
		* $\begin{equation}q(z = k|x)= \begin{cases} 1& \text{for}\ k = arg\min_j||z_e(x) − e_j||^2 \\ 0& \text{otherwise} \end{cases}\end{equation}$
			* the proposal distribution $q(z = k|x)$ is deterministic
		* $p(z=k)=\frac{1}{K}​,\text{for all} k\in{1,2,…,K}$
			* the prior is uniform
		* so the the KL divergence : $KL(q(z|x)|p(z))$$=\sum^K_{k=1}q(z=k|x)log\frac{q(z=k|x)}{p(x)}$$=log\frac{1}{\frac{1}{K}} = log\ K$
			, which is a constant
	
	* one can alternatively also update the **dictionary** items as function of **EMA (exponential moving average)**
		* Given a code vector $e_i$, say we have $n_i$ encoder output vectors, $\{z_{i,j}\}_{j=1}^{n_i}$, that are quantized to $e_i$:
			* $N_{i}^{(t)}=\gamma N_i^{t-1}+(1-\gamma)n_i^{(t)}$
			* $m_i^{(t)}=\gamma m_i^{(t-1)}+(1-\gamma)\sum^{n_i^{(t)}}_{j=1}z_{i,j}^{(t)}$
			* $e^{(t)}_i=m^{(t)}_{i}/N_i^{(t)}$
			where $(t)$ refers to batch sequence in time. $N_i$ and $m_i$ are accumulated vector count and volume, respectively.
	


### 🍰 Diffusion

* #### ✨ Diffusion Model
	* 🌟 The Forward Process
		* Combing the fact with **Markov assumption**
			* $q(x_{1:T}|x_0):= \prod^T_{t=1}q(x_t|x_{t-1})$
		* Sample from a Gaussian distribution whose **mean** is the previous value
			* $X_t \sim N(X_{t-1},1) \Longleftrightarrow X_t=X_{t-1}+N(0,1)$
				Prove it : 
				Apply the Law of Total Probability : $p_{x_t}(x_t)=\int p(x_t|x_{t-1})p(x_{t-1})dx_{t-1}$
				Use the notation : $N(x;\mu,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{1}{2}(\frac{x-\mu}{\sigma})^2)$
				So, replace the conditional distribution : $p_{x_t}(x_t)=\int N(x_t;x_{t-1},1)p(x_{t-1})dx_{t-1}$
				Due to the notation, we can rearrange : $p_{x_t}(x_t)=\int N(x_t-x_{t-1};0,1)p(x_{t-1})dx_{t-1}$
				Use the definition of convolution : $(f∗g)(t)=\int^{\infty}_{\infty}​f(\tau)g(t−\tau)d\tau$
				So, we get $p_{x_t}(x_t)=(N(0,1)*p_{x_{t-1}})(x_{t})$
				Due to the independence of the two distribution, we can rewrite the above formation as $X_{t}=N(0,1) + X_{t-1}$
		* Utilizing the above equivalent and using the [reparameterization trick](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick): 
			* $q(x_{1:T}|x_0):= \prod^T_{t=1}q(x_t|x_{t-1}) := \prod^T_{t=1}N(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)$
			* Sample $x_t$ from $N(x_{t-1},1)$
		* $\beta_1,...,\beta_T$ is a variance schedule (either learned or fixed)
			If well-behaved, it ensures that $x_T$ is nearly an isotropic Gaussian for sufficiently large $T$.
	
	* 🌟 The Reverse Process
		Starting with the pure Gaussian noise $p(x_T):=N(x_T;0,I)$. The model learns the joint distribution $p_{\theta}(x_{0:T})$ as $p_{\theta}(x_{0:T}):=p(x_T)\prod^T_{t=1}p_{\theta}(x_{t-1}|x_t):=p(x_T)\prod^T_{t=1}N(x_{t-1};\mu_{\theta}(x_t,t),\sum_{\theta}(x_t,t)))$
		* Due to **the Markov Formulation** that a given reverse diffusion transition distribution depends only on the **previous** time step: $p_{\theta}(x_{t-1}|x_t):=N(x_{t-1};\mu_{\theta}(x_t,t),\sum_{\theta}(x_t,t)))$
	
	* 🌟 Training
		* A Diffusion Model is trained by finding **the reverse Markov transitions** that **maximize** the likelihood of the training data. 
			* **Minimize** Cross-Entropy Loss ： 
				* $E_{q(x_0)}[-log\ p_{\theta}(x_0)]$
		* In practice, we try to **minimize** the **variational upper bound** on the negative log likelihood. 
			* $L_{CE}=-E_{q(x_0)}[log\ p_{\theta}(x_0)]=-E_{q(x_0)}[log (\int p_{\theta}(x_{0:T})d_{0:T})]$
			* Introduce **forward** probability $q$ : $L_{CE}=-E_{q(x_0)}[log (\int q(x_{1:T}|x_0) \frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}d_{0:T})]$ $L_{CE}=-E_{q(x_0)}[log (E_{q(x_{1:T}|x_0)} \frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)})]$
			* Use **Jensen inequality** : $f(E[X])\leq E[f(X)]$, so we have $L_{CE}\leq -E_{q_(x_{0:T})}log\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}$
			* So we get the the variational lower bound(VLB) : $E_{q_(x_{0:T})}[log\frac{q(x_{1:T}|x_0)}{p_{\theta}(x_{0:T})}]=L_{VLB}$
		* To convert each term in the equation to be analytically computable, the objective can be further rewritten to be a combination of several **KL-divergence** and **entropy terms**.
			* $L_{VLB}=E_{q(x_{0:T})}[log\frac{q(x_{1:T}|x_0)}{p_{\theta}(x_{0:T})}]$ $= E_{q}[log\frac{\prod^T_{t=1}q(x_t|x_{t-1})}{p_{\theta}(x_T)\prod^T_{t=1}p_{\theta}(x_{t-1}|x_t)}]$ $=E_q[-logp_{\theta}(x_T)+\sum^T_{t=1} log \frac{q(x_t|x_{t-1})}{p_{\theta}(x_{t-1}|x_t)}]$ $=E_q[-logp_{\theta}(x_T)+\sum^T_{t=2}log \frac{q(x_t|x_{t-1})}{p_{\theta}(x_{t-1}|x_t)}+log\frac{q(x_1|x_{0})}{p_{\theta}(x_{0}|x_1)}]$ $=E_q[-logp_{\theta}(x_T)+\sum^T_{t=2}log ( \frac{q(x_t|x_{t-1}, x_0)}{p_{\theta}(x_{t-1}|x_t)} ) +log\frac{q(x_1|x_{0})}{p_{\theta}(x_{0}|x_1)}]$ $=E_q[-logp_{\theta}(x_T)+\sum^T_{t=2}log (\frac{q(x_{t-1}|x_{t}, x_0) \cdot \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)} }{p_{\theta}(x_{t-1}|x_t)} ) +log\frac{q(x_1|x_{0})}{p_{\theta}(x_{0}|x_1)}]$  $=E_q[-logp_{\theta}(x_T)+\sum^T_{t=2}log (\frac{q(x_{t-1}|x_{t}, x_0)}{p_{\theta}(x_{t-1}|x_t)} ) + \sum^T_{t=2}log\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}  +log\frac{q(x_1|x_{0})}{p_{\theta}(x_{0}|x_1)}]$ $=E_q[-logp_{\theta}(x_T)+\sum^T_{t=2}log (\frac{q(x_{t-1}|x_{t}, x_0)}{p_{\theta}(x_{t-1}|x_t)} )+log\frac{q(x_T|x_0)}{q(x_1|x_0)}  +log\frac{q(x_1|x_{0})}{p_{\theta}(x_{0}|x_1)}]$ $=E_q[log \frac{q(x_T|x_0)}{p_{\theta}(x_T)}+\sum^T_{t=2}log (\frac{q(x_{t-1}|x_{t}, x_0)}{p_{\theta}(x_{t-1}|x_t)} )-log p_{\theta}(x_{0}|x_1)]$ $=E_q(D_{KL}(q(x_T|x_0)||p_{\theta}(x_T))+\sum^T_{t=2}D_{KL}(q(x_{t-1}|x_t,x_0)||p_{\theta}(x_{t-1}|x_t)-log p_{\theta}(x_0|x_1))$
			* Let’s label each component in the variational lower bound loss separately:
				* $L_{VLB}=L_T+L_{T-1}+...+L_0$
				* $L_{T}=D_{KL}(q(x_T|x_0)||p_{\theta}(x_T))$
				* $L_{t}=D_{KL}(q(x_{t}|x_{t+1},x_0)||p_{\theta}(x_{t}|x_{t+1})$, $1 \leq x \leq T-1$
				* $L_0 = -log p_{\theta}(x_0|x_1)$
	
	* 🌟 Parameterization
		* $L_t$ | $\mu_{\theta}$
			* $L_T$ is constant and can be ignored during training because $q$ has no learnable parameters and $x_T$ is a Gaussian noise.
			* [Ho et al. 2020](https://arxiv.org/abs/2006.11239) models $L_0$ using a separate discrete decoder derived from $N(x_0;\mu_{\theta}(x_1,1),\sum_{\theta}(x_1,1))$.
			* $p_{\theta}(x_{t-1}|x_t):=N(x_{t-1};\mu_{\theta}(x_t,t),\sum_{\theta}(x_t,t)))$ | [Reverse Process](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process)
				* we would like to train $\mu_{\theta}$ to predict $\widetilde{\mu}_{t}=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_{\theta}(x_t,t))$
			* The loss $L_t$ is parameterized to minimize the difference from $\widetilde{\mu}_{t}$ : $L_t=E_{x_0, \epsilon}[\frac{1}{2||\sum_{\theta}(x_t,t)||^2_2} ||\widetilde{\mu}_{t}(x_t,x_0)-\mu_{\theta}(x_t,t)||^2]$
				* [Close Form](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions) of KL Divergence
		* $\beta_t$
			* The forward variances are set to be a sequence of linearly increasing constants in [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), from $\beta_1=10^{−4}$ to $\beta_T=0.02$.
		* $\sum_{\theta}$
			* [Ho et al. (2020)](https://arxiv.org/abs/2006.11239) chose to fix $\beta_t$ as constants instead of making them learnable and set $\sum_{\theta}(x_t,t)=\sigma^2_t I$ , where $\sigma_t$ is not learned but set to $\beta_t$ or $\widetilde{\beta}_t=\frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_{t}}$.
	
	* 🌟 Model Architecture
		* U-Net
			U-Net ([Ronneberger, et al. 2015](https://arxiv.org/abs/1505.04597)) consists of a **downsampling** stack and an **upsampling** stack.
		* Transformer
			Diffusion Transformer (DiT; [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)) for diffusion modeling operates on **latent patches**, using the same design space of [Latent Diffusion Model](https://arxiv.org/abs/2112.10752).
	
	* 🌟 Brief
		- **Pros**: **Tractability** and **flexibility** are two **conflicting** objectives in generative modeling. 
			- **Tractable** models can be analytically evaluated and cheaply fit data (e.g. via a Gaussian or Laplace), but they cannot easily describe the structure in rich datasets. 
			- **Flexible** models can fit arbitrary structures in data, but evaluating, training, or sampling from these models is usually expensive. 
			- Diffusion models are **both** analytically tractable and flexible.
		- **Cons**: 
			- Diffusion models rely on a long Markov chain of diffusion steps to generate samples, so it can be quite **expensive** in terms of **time** and compute. 
			- New methods have been proposed to make the process much faster, but the sampling is still slower than GAN.
	
	* [reference](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
	


* #### ✨ Gaussian Mixture Model (GMM)
	


* #### ✨ Flow-Based Model
	

