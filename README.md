# COPproject2
Quantum Variational Monte Carlo

Simple harmonic oscillator in 1D
-
Solvable toy example in 1D

Hamiltonian 

<img src="http://latex.codecogs.com/gif.latex?H=-\frac{1}{2}\frac{d^2}{dx^2}+\frac{1}{2}x^2" border="0"/>

Trial wavefunction 

<img src="http://latex.codecogs.com/gif.latex?\psi_T(x,\alpha)=\exp\left(-\alpha\hspace{0}x^2\right)" border="0"/>

Local energy

<img src="http://latex.codecogs.com/gif.latex?E_L(x,\alpha)=\alpha+x^2\left(\frac{1}{2}-2\alpha^2\right)" border="0"/>


Helium atom
-
One version with brute force variation of parameters <img src="http://latex.codecogs.com/gif.latex?\alpha" border="0"/>

One version with stochastic gradient approximation (SGA)

Hamiltonian 

<img src="http://latex.codecogs.com/gif.latex?H=-\frac{1}{2}\nabla_1^2-\frac{1}{2}\nabla_2^2-\frac{2}{r_1}-\frac{2}{r_2}+\frac{1}{r_{12}}" border="0"/>

where

<img src="http://latex.codecogs.com/gif.latex?r_{12}=|\vec{r}_1-\vec{r}_2|" border="0"/>


Trial wavefunction 

<img src="http://latex.codecogs.com/gif.latex?\psi_T(\vec{r}_1,\vec{r}_2,\alpha)=\exp\left(-2(r_1+r_2)+\frac{r_{12}}{2(1+\alpha\;r_{12})}\right)" border="0"/>

Local energy

<img src="http://latex.codecogs.com/gif.latex?E_L(\vec{r}_1,\vec{r}_2,\alpha)=-4+\left(\frac{\vec{r}_1}{r_1}-\frac{\vec{r}_2}{r_2}\right)\cdot\left(\vec{r}_1-\vec{r}_2\right)\frac{1}{r_{12}(1+\alpha\;r_{12})^2}-\frac{1}{r_{12}(1+\alpha\;r_{12})^3}-\frac{1}{4(1+\alpha\;r_{12})^4}+\frac{1}{r_{12}}" border="0"/>
