
## 📚 Random Keywords

* ##### ✨Feature-wise Linear Modulation (FiLM)
	$\text{FiLM}(x_i​,\gamma,\beta)=\gamma_i​⋅x_i​+\beta_i​$
	* $\text{FiLM}$ dynamically adjusts neural network feature representations using scaling and shifting parameters generated from conditional inputs.


* ##### ✨Logistic Cumulative Distribution Function (Logistic CDF)
	* A sigmoid-shaped curve commonly used in binary classification tasks, especially in logistic regression, to estimate the probability of an event occurring.


## 🧮 Probability

* ##### ✨**Bayesian Inference**
	
	$p(z|x)=\frac{p(x|z)\cdot p(z)}{p(x)}$
	
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


* ✨ MAP Inference
	**MAP Inference (Maximum A Posteriori Inference)** is a fundamental concept in probabilistic models, particularly in Bayesian statistics and machine learning. It involves finding the most likely parameters or states given observed data, under a probabilistic model : 
	$z_{MAP}​=arg \max_{z} \ ​p(z|\mathbf{X})$
	$=arg \max_{z} \frac{p(\mathbf{X}|z)p(z)}{p(\mathbf{X})}$
	$= arg \max_{z} \ p(\mathbf{X}|z)p(z)$


* ✨ Maximum Likelihood Estimation (MLE)
	**MLE** focuses solely on maximizing the likelihood $p(\mathbf{X}|z)$, assuming no prior knowledge about $z$ : $z_{MLE}​=arg \max_{z} \ ​p(\mathbf{X}|z)$


* ✨ Markov Chain (MC)
	MCMC is based on the theory of Markov chains. A Markov chain is a stochastic process with the "memoryless" property, meaning that the next state depends only on the current state and not on the sequence of states that preceded it. Formally, if $x_0,x_1,x_2,…$ is a Markov chain, then the transition probability satisfies: $P(x_{t+1}​|x_{t}​,x_{t−1}​,…,x_0​)=P(x_{t+1}​|x_{t}​)$


* ✨ Markov Chain Monte Carlo (MCMC)
	* The core idea of MCMC is to construct a Markov chain that has the desired target distribution as its stationary distribution. 
	* By running this chain for a sufficient number of steps, we can generate samples that approximate the target distribution. 
	* The general steps of an MCMC algorithm are as follows:
		1. **Initialization**: Choose an initial state $x_0$​.
		2. **Generate a Candidate Sample**: Based on the current state $x_t$​, generate a candidate state $x_{t+1}$​.
		3. **Accept-Reject Rule**: Use a rule to decide whether to accept the candidate sample $x_{t+1}$​ as the new state. If accepted, move to $x_{t+1}$​; otherwise, remain at $x_t$​.
		4. **Repeat**: Repeat steps 2 and 3 until a sufficient number of samples are generated.

