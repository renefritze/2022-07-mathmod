---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
rise:
  autolaunch: true
---

+++ {"slideshow": {"slide_type": "slide"}}

<center><img src="img/pymor_logo.png" width="70%"></center>

# pyMOR -- Model Order Reduction with python

## 10th Vienna International Conference on Mathematical Modelling

## René Fritze

+++ {"slideshow": {"slide_type": "slide"}}

<div class="container">

<div>

# Get the slides

## <https://rene.fritze.me/22-mathmod>

README has a link to execute the presentation on mybinder.org

</div>

<div>
<img src="qr_self.png" />
</div>
</div>


# Documentation

<https://docs.pymor.org>

- (interactive) [Tutorials](https://docs.pymor.org/2021-2-0/tutorials.html)
- [Design](https://docs.pymor.org/2021-2-0/technical_overview.html) overview
- [API](https://docs.pymor.org/2021-2-0/autoapi/index.html) reference

</div>

<div>
<img src="qr_docs.png" />
</div>
</div>

+++ {"slideshow": {"slide_type": "slide"}}

# What is pyMOR?

+++ {"slideshow": {"slide_type": "fragment"}}

- a software library for writing **M**odel **O**rder **R**eduction applications

+++ {"slideshow": {"slide_type": "fragment"}}

- in the **py**thon programming language.

+++ {"slideshow": {"slide_type": "fragment"}}

- BSD-licensed, fork us on [Github](https://github.com/pymor/pymor).

+++ {"slideshow": {"slide_type": "fragment"}}

- Started 2012, 22k lines of code, 7k commits.

+++ {"slideshow": {"slide_type": "slide"}}

### Design Goal 1

## One library for algorithm development *and* large-scale applications

+++ {"slideshow": {"slide_type": "fragment"}}

- Small NumPy/SciPy-based discretization toolkit for easy prototyping.
- `VectorArray`, `Operator`, `Model` interfaces for seamless integration with high-performance PDE solvers.

+++ {"slideshow": {"slide_type": "slide"}}

### Design Goal 2

## Unified view on MOR

+++ {"slideshow": {"slide_type": "fragment"}}

- Implement RB and system-theoretic methods in one common language.

+++ {"slideshow": {"slide_type": "slide"}}

## Implemented Algorithms

- Gram-Schmidt, POD, HAPOD.
- Greedy basis generation with different extension algorithms.
- Automatic (Petrov-)Galerkin projection of arbitrarily nested affine combinations of operators.
- Interpolation of arbitrary (nonlinear) operators, EI-Greedy, DEIM.
- A posteriori error estimation.

+++ {"slideshow": {"slide_type": "slide"}}

## Implemented Algorithms

- System theory methods: balanced truncation, IRKA, ...
- Iterative linear solvers, eigenvalue computation, Newton algorithm, time-stepping algorithms.
- Non-intrusive MOR using artificial neural networks.

+++ {"slideshow": {"slide_type": "slide"}}

## PDE Solvers: Official Support

- [deal.II](https://dealii.org) [Example Code](https://github.com/DavidSCN/mor-coupling)
- [FEniCS](https://fenicsproject.org) [Demo](https://github.com/pymor/pymor/blob/main/src/pymordemos/neural_networks_fenics.py)
- [NGSolve](https://ngsolve.org) [Demo](https://github.com/pymor/pymor/blob/main/src/pymordemos/thermalblock_simple.py#L202)

+++ {"slideshow": {"slide_type": "slide"}}

## PDE Solvers: Used with

- [DUNE](https://dune-project.org)
- [preCICE](https://precice.org) [Example Code](https://github.com/DavidSCN/mor-coupling)
- [BEST](https://www.itwm.fraunhofer.de/en/departments/sms/products-services/best-battery-electrochemistry-simulation-tool.html)
- [GridLOD](https://github.com/fredrikhellman/gridlod)
- file I/O, e.g. [COMSOL](https://comsol.com)
- ...

+++ {"slideshow": {"slide_type": "slide"}}

## Parallelisation/Async support

- [Automatic MPI Models](https://github.com/pymor/pymor/blob/main/src/pymor/models/mpi.py#L82)
- [Worker Pools](https://github.com/pymor/pymor/blob/main/src/pymor/parallel/interface.py#L8)
- [Use of `concurrent.futures`](https://github.com/pymor/pymor/blob/main/src/pymor/algorithms/hapod.py#L133)

+++ {"slideshow": {"slide_type": "slide"}}

## Open Development

- over 25 contributors
- everyone can/should(!) contribute
- everyone can become main developer
- not just code contributions are valued!

+++ {"slideshow": {"slide_type": "slide"}}

## pyMOR 2022.1 Highlights

- Support for discrete-time systems
- Structure-preserving MOR for symplectic systems
- new features for the neural network based reductors
- FEniCS discretizer

+++ {"slideshow": {"slide_type": "slide"}}

## Tutorials

- [unstable LTI systems](https://docs.pymor.org/2022-1-0/tutorial_unstable_lti_systems.html)
- [Model order reduction with artificial neural networks](https://docs.pymor.org/2022-1-0/tutorial_mor_with_anns.html)
- [Model order reduction for PDE-constrained optimization problems](https://docs.pymor.org/2022-1-0/tutorial_optimization.html)
- [Binding an external PDE solver to pyMOR](https://docs.pymor.org/2022-1-0/tutorial_external_solver.html)

+++ {"slideshow": {"slide_type": "slide"}}

## Minimal pyMOR installation

  ```
  python -m pip install pymor
  ```

+++ {"slideshow": {"slide_type": "slide"}}

## Minimal pyMOR installation

  ```
  conda install -c conda-forge pymor
  ```

+++ {"slideshow": {"slide_type": "slide"}}

## Hello pyMOR

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import pymor
pymor.config
```

+++ {"slideshow": {"slide_type": "slide"}}

# Reduced Basis Methods with pyMOR

+++ {"slideshow": {"slide_type": "slide"}}

## Building the Full Order Model (FOM)

+++ {"slideshow": {"slide_type": "slide"}}

### The Thermal Block Problem

Solve:

\begin{align}
- \nabla \cdot [d(x, \mu) \nabla u(x, \mu)] &= f(x),  & x &\in \Omega,\\
                                  u(x, \mu) &= 0,     & x &\in \partial\Omega,
\end{align}

where

\begin{align}
d(x, \mu) &= \sum_{q=1}^Q \mathbb{1}_{\Omega_q}, \\
f(x)      &= 1.
\end{align}

satisfying $\overline{\Omega} = \overline{\dot{\bigcup}_{i=1}^{Q} \Omega_q}$.

+++ {"slideshow": {"slide_type": "slide"}}

### Setting up an analytical description of the thermal block problem

The thermal block problem already comes with pyMOR:

```{code-cell} ipython3
from pymor.basic import *
p = thermal_block_problem([2,2])
```

+++ {"slideshow": {"slide_type": "fragment"}}

Our problem is parameterized:

```{code-cell} ipython3
p.parameters
```

+++ {"slideshow": {"slide_type": "slide"}}

### Looking at the definition

We can easily look at the definition of `p` by printing its `repr`:

```{code-cell} ipython3
p
```

+++ {"slideshow": {"slide_type": "slide"}}

### Building a discrete model

We use the builtin discretizer `discretize_stationary_cg` to compute a finite-element discretization of the problem:

```{code-cell} ipython3
from pymor.basic import *
fom, data = discretize_stationary_cg(p, diameter=1/100)
```

+++ {"slideshow": {"slide_type": "fragment"}}

`fom` is a `Model`. It has the same `Parameters` as `p`:

```{code-cell} ipython3
fom.parameters
```

+++ {"slideshow": {"slide_type": "slide"}}

### Looking at the model

`fom` inherits its structure from `p`:

```{code-cell} ipython3
fom
```

+++ {"slideshow": {"slide_type": "slide"}}

### Note

> Using an `analyticalproblem` and a `discretizer` is just one way
  to build the FOM.

> Everything that follows works the same for a FOM built using an external PDE solver.

+++ {"slideshow": {"slide_type": "slide"}}

### Solving the FOM

Remember the FOM's parameters:

```{code-cell} ipython3
fom.parameters
```

+++ {"slideshow": {"slide_type": "fragment"}}

To `solve` the FOM, we need to specify values for those parameters:

```{code-cell} ipython3
U = fom.solve({'diffusion': [1., 0.01, 0.1, 1]})
fom.visualize(U)
```

+++ {"slideshow": {"slide_type": "slide"}}

`FEniCS` discretizer as drop-in replacement

```{code-cell} ipython3
from pymor.discretizers.fenics import discretize_stationary_cg as fenics_discretizer
fom_fenics, data = fenics_discretizer(p, diameter=1/100)
U = fom_fenics.solve({'diffusion': [1., 0.01, 0.1, 1]})
fom_fenics.visualize(U)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Reducing the FOM

+++ {"slideshow": {"slide_type": "slide"}}

### Building an approximation space

As before, we compute some random solution **snapshots** of the FOM, which will
span our **reduced** approximation space:

```{code-cell} ipython3
snapshots = fom.solution_space.empty()
for mu in p.parameter_space.sample_randomly(10):
    snapshots.append(fom.solve(mu))
basis = gram_schmidt(snapshots)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Projecting the Model

In pyMOR, ROMs are built using a `Reductor`. Let's pick the most basic `Reductor`
available for a `StationaryModel`:

```{code-cell} ipython3
reductor = StationaryRBReductor(fom, basis)
rom = reductor.reduce()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Comparing ROM and FOM

```{code-cell} ipython3
fom
```

+++ {"slideshow": {"slide_type": "slide"}}

### Comparing ROM and FOM

```{code-cell} ipython3
rom
```

+++ {"slideshow": {"slide_type": "slide"}}

### Solving the ROM

To solve the ROM, we just use `solve` again,

```{code-cell} ipython3
mu = fom.parameters.parse([1., 0.01, 0.1, 1])
u_rom = rom.solve(mu)
```

+++ {"slideshow": {"slide_type": "fragment"}}

to get the reduced coefficients:

```{code-cell} ipython3
u_rom
```

+++ {"slideshow": {"slide_type": "fragment"}}

A high-dimensional representation is obtained from the `reductor`:

```{code-cell} ipython3
U_rom = reductor.reconstruct(u_rom)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Computing the MOR error

Let's compute the error:

```{code-cell} ipython3
U = fom.solve(mu)
ERR = U - U_rom
ERR.norm() / U.norm()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Computing the MOR error

and look at it:

```{code-cell} ipython3
fom.visualize(ERR)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Certified Reduced Basis Method

Let's use use a more sophisticated `reductor` which assembles an efficient
upper bound for the MOR error:

```{code-cell} ipython3
reductor = CoerciveRBReductor(
   fom,
   product=fom.h1_0_semi_product,
   coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Certified Reduced Basis Method

and build a basis using a greedy search over the parameter space:

```{code-cell} ipython3
training_set = p.parameter_space.sample_uniformly(4)
print(training_set[0])
```

+++ {"slideshow": {"slide_type": "slide"}}

### Certified Reduced Basis Method

```{code-cell} ipython3
greedy_data = rb_greedy(fom, reductor, training_set, max_extensions=20)
print(greedy_data.keys())
rom = greedy_data['rom']
```

+++ {"slideshow": {"slide_type": "slide"}}

### Testing the ROM

Let's compute the error again:

```{code-cell} ipython3
mu = p.parameter_space.sample_randomly()
U = fom.solve(mu)
u_rom = rom.solve(mu)
ERR = U - reductor.reconstruct(u_rom)
ERR.norm(fom.h1_0_semi_product)
```

+++ {"slideshow": {"slide_type": "fragment"}}

and compare it with the estimated error:

```{code-cell} ipython3
rom.estimate_error(mu)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Is it faster?

Finally, we compute some timings:

```{code-cell} ipython3
from time import perf_counter
mus = p.parameter_space.sample_randomly(10)
tic = perf_counter()
for mu in mus:
    fom.solve(mu)
t_fom = perf_counter() - tic
tic = perf_counter()
for mu in mus:
    rom.solve(mu)
t_rom = perf_counter() - tic
```

+++ {"slideshow": {"slide_type": "slide"}}

### Yes, it is

```{code-cell} ipython3
print(f'Speedup: {t_fom/t_rom}')
```
