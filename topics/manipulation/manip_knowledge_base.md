

## 🧀 Knowledge

* SO(3) : 3D rotation group - 纯旋转

* SE(3) : Special Euclidean Group - 刚体运动

* Cartesian Pose | 笛卡尔坐标
	* Position : $x$, $y$, $z$
	* Orientation
		* 🔥 | Euler Angles | 欧拉角 : $rx$, $ry$, $rz$
			
			* Roll | 横滚 | $rx$ | $\alpha$
			* Pitch | 俯仰 | $ry$ | $\beta$
			* Yaw | 偏航 | $rz$ | $\gamma$
			
			💦 Rotation Matrix | 旋转矩阵 : 
			* $R_z(\gamma) = \begin{pmatrix} \cos \gamma & -\sin \gamma & 0 \\ \sin \gamma & \cos \gamma & 0 \\ 0 & 0 & 1 \end{pmatrix}$
			* $R_y(\beta) = \begin{pmatrix} \cos \beta & 0 & \sin \beta \\ 0 & 1 & 0 \\ -\sin \beta & 0 & \cos \beta \end{pmatrix}$
			* $R_x(\alpha) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos \alpha & -\sin \alpha \\ 0 & \sin \alpha & \cos \alpha \end{pmatrix}$
			* $R=R_z​(\gamma)R_y​(\beta)R_x​(\alpha)$
			
			💦 Gimbal Lock | 万向节死锁
			* When we rotate **90** degree through Y-axis
			* $R=R_z​(\gamma)R_y​(\frac{\pi}{2})R_x​(\alpha)= \begin{pmatrix} 0 & 0 & 1 \\ \sin(\alpha+\gamma) & \cos(\alpha+\gamma) & 0 \\ -\cos(\alpha+\gamma) & \sin(\alpha+\gamma) & 0 \end{pmatrix}$
			* We cannot calculate the $\alpha$ and $\gamma$ **respectively**
			* The **Gimbal Lock** is due to the possibility that two axes coincide, which leads to the **uncertainty** of **the final reverse resolution result** (when **Roll** and **Yaw** coincide, it is impossible to determine whether the rotation is Roll or Yaw from the final result).
			
		* 🔥 | Quaternions | 四元数 : $qx,qy,qz,qw$
			
			💦 **Multiple rotations** about **a coordinate axis** can be equivalent to **A Certain Angle** $w$ of rotation about **A Certain Vector** $\vec{K}=[x,y,z]$.
			* $q=w+xi+yj+zk = ((x,y,z)sin\frac{\theta}{2}, cos\frac{\theta}{2})$ 
				* $x=\vec{K}_x \cdot sin\frac{\theta}{2}$
				* $y=\vec{K}_y \cdot sin\frac{\theta}{2}$
				* $z=\vec{K}_z \cdot sin\frac{\theta}{2}$
				* $w=cos\frac{\theta}{2}$
				* $x^2+y^2+z^2+w^2=1$
			
			💦 LINK : [3b1b](https://www.youtube.com/watch?v=d4EgbgTm0Bg)
			

* Task and Motion Planning (TAMP)
	* PDDL (Planning Domain Definition Language)
		
	* PDDLStream (Planning Domain Definition Language Stream)
		

* **Sim2Real**
	* Domain Randomization
		* Introduce parameter randomization during simulation
		* Randomizing parameters during simulation training covers a wide range of conditions, potentially encompassing variations that might occur in real-world settings
	* Domain Adaptation
		* Utilize **Generative Adversarial Networks (GANs)** to map images from one distribution into another
	* Real2Sim2real
		* construct a “digital twin” simulation environment
	* TRANSIC
		* Enable real-time human intervention to correct robot behaviors in real-world scenes. 
		* The data collected from these interventions are used to train a residual policy
		* Integrating both foundational and residual policies ensures smoother trajectories in real-world applications following sim-to-real transfer
	* System Identification
		* Construct an accurate mathematical model of physical scenes in **real-world** environments, encompassing parameters such as dynamics and visual rendering
	* Lang4sim2real
		* Use natural language as a bridge to address the sim-to-real gap by using textual descriptions of images as a cross-domain unified signal.
	

* Proportional-Derivative (PD) controller
	* Proportional (P)
	* Derivative (D)

* Rapidly-exploring Random Tree (RRT)
	* It works by incrementally building **a tree** rooted at **the start position**, with branches that grow toward unexplored areas of the space. 
	* The tree expands by **randomly sampling** the state space and connecting **the sampled points** to **the nearest existing point** in the tree, gradually covering the space and searching for a path to the goal.

* Residual Policy
	* $\pi_{total}​(s)=\pi_{baseline}​(s)+\pi_{residual}​(s)$

## 📊 Dataset

1. **CALVIN**
	* [LeaderBoard](http://calvin.cs.uni-freiburg.de/)
	* [Code](https://github.com/mees/calvin)
	* [Paper](https://arxiv.org/abs/2112.03227)
2. RoboSet
	* [web](https://robopen.github.io/roboset/)
3. Bridge Data
	* [web](https://sites.google.com/view/bridgedata)
4. Bridge V2
	* [web](https://rail-berkeley.github.io/bridgedata/)
	* [huggingface](https://huggingface.co/datasets/jdvakil/RoboSet-Teleoperation/tree/main)
5. Berkeley UR5
	* [web](https://sites.google.com/view/berkeley-ur5/home)
	* [google-drive](https://drive.google.com/drive/folders/1u5AV7maR3AJ8x5abmpDegRDUJEhFFEmd)
