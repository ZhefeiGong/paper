
## 📚 Random Keywords


* ##### ✨Feature-wise Linear Modulation (FiLM)
	$\text{FiLM}(x_i​,\gamma,\beta)=\gamma_i​⋅x_i​+\beta_i​$
	* $\text{FiLM}$ dynamically adjusts neural network feature representations using scaling and shifting parameters generated from conditional inputs.


* ##### ✨Logistic Cumulative Distribution Function (Logistic CDF)
	A sigmoid-shaped curve commonly used in binary classification tasks, especially in logistic regression, to estimate the probability of an event occurring.


* ##### ✨**Bayesian Inference**
	$$
	p(z|x)=\frac{p(x|z)\cdot p(z)}{p(x)}
	$$
	* $x$ is the **data**
	* $z$ is the **variable**
	* $p(x∣z)$ is the **likelihood** function, representing the probability of the data $x$ given the latent variable $z$
	* $p(z)$ is the **prior** distribution
	* $p(z|x)$ is the **posterior** distribution
	* $p(x)$ is the **marginal likelihood** of the observed data, usually obtained by integrating over all possible values of $z$ : $p(x) = \int p(x|z)p(z)dz$


* ##### ✨Normal Distribution | Gaussian Distribution
	When we refer to a Normal distribution with a mean of 0 and a standard deviation of 1, we are specifically talking about the standard normal distribution.
	
	* A random variable $X$ is said to be normally distributed with mean $\mu$ and standard deviation $\sigma$ if its **probability dense function(PDF)** is given by : $f(x|\mu,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}​​exp(−\frac{(x−\mu)^2}{2\sigma^2})$
	
	* The cumulative distribution function (CDF) is : $\Phi(x)=P(X\leq x)=\int{x}{-\infty}\frac{1}{\sqrt{2\pi}}exp(-\frac{t^2}{2})dt$


* ##### ✨Bernoulli Distribution
	It describes the outcome of a single trial experiment that results in a binary outcome: success (usually coded as 1) or failure (usually coded as 0). This distribution is named after the Swiss mathematician Jacob Bernoulli.
	
	* A random variable $X$ follows a Bernoulli distribution if it takes the value $1$ with probability $p$ (success) and the value $0$ with probability $1−p$ (failure). The **probability mass function (PMF)** of a Bernoulli distributed random variable is given by : $f(X=x)=\begin{cases} p &\text{if}\ x=1 \\ 1-p &\text{if}\ x=0 \end{cases}$
	
	* Mean(Expected Value) : $E[X]=p$
	
	* Variance : $Var[X]=p(1-p)$
	
	* Standard Deviation : $\sigma_{X}=\sqrt{p(1-p)}$


* ##### ✨Uniform Distribution
	It describes a continuous probability distribution where every possible outcome has an equal likelihood of occurring within a given range.
	
	* A random variable $X$ is uniformly distributed over the interval $[a,b]$ if its **probability density function (PDF)** is given by : $f(x|a,b)=\begin{cases} \frac{1}{b-a} &\text{if}\ a\leq x \leq b \\ 0 &\text{otherwise}\  \end{cases}$
	
	* The **cumulative distribution function (CDF)** $F(x)$ of a uniformly distributed random variable $X$ over $[a,b]$ is : $F(x)=\begin{cases} 0 &\text{for}\ x <a \\ \frac{x-a}{b-a} &\text{for}\ a\leq x \leq b \\ 1 &\text{for}\ a > b \end{cases}$
	
	* The **mean (expected value)** of a uniform distribution over $[a, b]$ is : $E[X]=\frac{a+b}{2}$
	
	* The **variance** of $X$ is : $Var(X)=\frac{(b-a)^2}{12}$


* ##### ✨**Variational Auto-Encoder(VAE)**
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
			* this integral requires exponential time to compute. we therefore need to approximate this posterior distribution.
		* **variational inference** approximates the posterior $p(z|x)$ with a family of distributions $𝑞_{\lambda}(𝑧|𝑥)$.
			* use **the Kullback-Leibler divergence**, which measures the information lost when using $𝑞$ to approximate $𝑝$. $KL(q_{\lambda}(z|x)||p(z|x))=𝐸_𝑞[log\ ⁡𝑞_{\lambda}(𝑧|𝑥)]−𝐸_𝑞[log⁡\ 𝑝(𝑥,𝑧)]+log\ p(x)$
			* Our goal is to find the variational parameters $𝜆$ that minimize this divergence.
			* due to the pesky evidence $p(x)$ which is intractable, we consider : $\text{ELBO}(\lambda)=E_{q}[log\ p(x,z)]-E_{q}[log \ q_{\lambda}(z|x)]$
			* combining the KL divergence, we can rewrite the evidence : $log\ p(x) = \text{ELBO}(\lambda)+ KL(q_{\lambda}(z|x)||p(z|x))$
			* try to **minimize** **the Kullback-Leibler divergence** between the approximate and exact posteriors. Instead, we can **maximize** the **ELBO** which is equivalent
		* we can rewrite the ELBO, through $p(x,z)=p(x|z)p(z)$. $\text{ELBO}(\lambda)=E_{q}[log\ p(x,z)]-E_{q}[log \ q_{\lambda}(z|x)]$ $= E_q[log\ p(x|z)]+E_q[log\ p(z)]-E_{q}[log \ q_{\lambda}(z|x)$ $=E_q[log\ p(x|z)]-(E_{q}[log \ q_{\lambda}(z|x)-E_q[log\ p(z)])$ $=E_q[log\ p(x|z)]-KL(q_{\lambda}(z|x)||p(z))$
		* approximate posterior $𝑞_{\theta}(𝑧|𝑥,\lambda)$ with an _inference network_ (or encoder). parametrize the likelihood $𝑝_{\phi}(𝑥|𝑧)$ with a _generative network_ (or decoder)
			* we cab rewrite the $\text{ELBO}$ as : $\text{ELBO}(\theta,\phi)=E_{q_{\theta}(z|x)}[log\ p_{\phi}(x|z)]-KL(q_{\theta}(z|x)||p(z))$
			* $\text{ELBO}​(\theta,\phi)=−l_i​(\theta,\phi)$, and we try to **maximize** the $\text{ELBO}$
			* gradient ascent on the $\text{ELBO}$
		* the term **variational inference** usually refers to **maximizing** the $\text{ELBO}$ with respect to the variational parameters $\lambda$.
			* This technique is called **variational EM (expectation maximization)**, because we are maximizing the expected log-likelihood of the data with respect to the model parameters.
		* We have followed the recipe for **variational inference**. We’ve defined:
			- **a probability model** $p_{\phi}(x|z)$ of latent variables and data
			- **a variational family** $q_{\theta}(z|x)$ for the latent variables to approximate our posterior
	* [reference1](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) | [reference2](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

