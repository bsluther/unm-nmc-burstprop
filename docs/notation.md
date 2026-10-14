AI DISCLOSURE: this file was not written by AI and should not be without express consent of a human and modification of this statement. The purpose of this file is to (a) act as a central reference point, (b) standardize notation across the project, and (c) build the understanding of the authors. AI was used to assist during research and to review this file.

# Notation

A compilation of the notation used in this project. Most notation is taken from [Stuck 2025](./papers/stuck2025.pdf) and [Payeur 2021](./papers/payeur2021.pdf) but some modifications have been made for consistency, clarity, or to resolve apparent ambiguities or mistakes.

##### Spikes, bursts, and events
$\overline E_j:$ the rate estimate of neuron $j$. Equivalent to the eligibility trace $\tilde E_j$. (Stuck 2025, p5)
<br/><br/>


$S_i(t) = \displaystyle\sum_k \delta (t-t_k^{(i)}):$ the event train of neuron $i$, a function of continuous time $t$. The event train tells us the times at which neuron $i$ emitted an event. 
- Here $t_k^{(i)}$ is the *time* of the $k$-th event from neuron $i$. Payeur calls this $t_{i,event}$.
- We use the event train to count the number of events in some continuous time interval $[t_0, t_1]$ by integrating. Say there were 4 events in the interval, then $\displaystyle\int_{t_0}^{t_1} S_i(t)dt = 4$.
- Another use is to move from a continuous integral to a discrete summation (the *sifting property*): $\displaystyle\int_{-\infty}^{\infty} f(t)S_i(t)dt = \displaystyle\sum_k f(t_k^{(i)})$.
- $\delta (x) = \begin{cases}\infty & x=0 \\ 0 & x\ne 0\end{cases} \quad$ is (the heuristic definition of) the [Dirac delta function](https://en.wikipedia.org/wiki/Dirac_delta_function), which is useful because $\displaystyle\int_{-\infty}^{\infty} \delta(x)dx = 1$.
- Note that this is the mathematical formulation in continuous time. In the implementation we have tensors with 0/1 entries at discrete time steps $\Delta t$.
- Payeur calls the event train $E_i(t)$ and uses $S_i(t)$ for the spike train. I like that notation better (E is for event, lines up with the eligibility trace and rate estimate) but I'm hesitant to diverge from Stuck's notation.
<br/><br/>

##### Time constants
$\tau_{\tilde E}:$ the eligibility trace time constant. (Stuck 2025, p5)
- Stuck 2025 uses 10ms.
<br/><br/>

$\tau_d:$ dendritic membrane time constant. (Stuck 2025, p5)
- Stuck 2025 uses 10ms.
<br/><br/>

##### Membrane potentials
$V_s(t):$ somatic membrane potential. (Stuck 2025)
<br/><br/>

$\tilde V_s(t) = V_s - \dfrac{E_L}{V_{\text{threshold}} - E_L}:$ dimensionless somatic membrane potential. (Stuck 2025)
<br/><br/>


##### Burst probabilities

$P_B:$

