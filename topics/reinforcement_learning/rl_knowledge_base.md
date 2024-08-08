
### 📖 Keywords

* Monte Carlo
* Dynamic Programming
* Bellman Equation
* Policy Gradient
* Policy Iteration
* Value Iteration
* Discrete
* Continuous
* Stochastic


### 📖 Category

**⚖️Policy-Based Methods $\Rightarrow$ Policy Gradient**

* 🌟**Algorithm :** 
	$$
	\nabla_{\theta}​J(\theta)=E_{\tau∼\pi_{\theta}}​​[\sum_{t=0}^T​\nabla_{\theta}​logπ_{\theta}​(a_t​∣s_t​) \cdot G_t​]
	$$

	* REINFORCE
	* Proximal Policy Optimization(PPO)
	* Trust Region Policy Optimization(TRPO)
	* ...

* 💊Problem : 
	
	* **sample inefficiency**



**⚖️Value-Based Methods $\Rightarrow$ Q-Learning**

* 🌟**Update Rule :** 
	$$
	Q(s_t​,a_t​)\Leftarrow Q(s_t​,a_t​)+\alpha[r_{t+1}​+\gamma max_{a}​Q(s_{t+1}​,a)−Q(s_t​,a_t​)]
	$$
	
	* Deep Q-Network (DQN)
	* Double DQN
	* Deep Deterministic Policy Gradient(DDPG)
	* Actor-Critic
	* ...


