# Generalized Path/Tile-Sum Algorithm for Computing Operator Correlations 

### Problem Setting

We consider a tensor product of single-site $q$-dimensional Hilbert spaces $\mathcal{H} \simeq \mathbb{C}^q$ over a spatial lattice of length $2L$ where $x \in \{-\frac{(L-1)}{2},...,-\frac{1}{2},0,\frac{1}{2},...,\frac{L}{2}\}$. By convention, time evolution is given by layering rows of local unitary gates along the vertical axis of the tensor network. Each local gate $U$ evolves a combined 2-site Hilbert space $\mathbb{C}^q \otimes \mathbb{C}^q$ for half a full time step (N.B. another half step is needed to couple sites to both their neighbours within the full step). At $t=0$ we place a local charge operator on the central integer site $x=0$. 

We compute infinite-temperature correlation functions in the Heisenberg picture, which are nothing more than the Hilbert-Schmidt inner product $\langle \mathbb{U}^{\dagger}(t) Z(0) \mathbb{U}(t), Z(x) \rangle$ between our time-evolved origin charge and each local charge operator throughout space. This amounts to contracting a particular tensor network.

Following Bertini, Prosen and Kos (https://arxiv.org/abs/2006.07304) we contract the tensor network via a tile-sum decomposition of its $x_{+} \times x_{-}$ causal block - that is, the intersection of the forward light-cone from $(0,0)$ and the reverse light-cone from $(x,t)$ - generalized to cater for tiles of arbitrary dimension $d$. The principal idea is to slice the block into a grid of sub-blocks and merge the internal indices along each face-cut. Then, you insert resolutions of the identity within every internal face-cut to project that face into complementary trivial and non-trivial subspaces. The result is a sum over all tile configurations where the contracted tile tensors are either projected to the trivial (1D) or non-trivial subspace on each of their 4 faces (N.B. connected faces get projected to the same subspace so their contraction is still valid). Moreover, only configurations with an unbroken path of non-blank tiles between the bottom-left and top-right corners contribute to the sum. This motivates a path-sum interpretation, where each path connects the 2 corners and may fork into branches as long as they rejoin later.

In contrast to Bertini, Prosen and Kos who use trivial $1\times 1$ sub-blocks ($W$), a perfectly regular partition of the causal block is not possible when either $x_{+}$ or $x_{-}$ are indivisible by $d$. To account for this we must include "trimmings" on the leftmost and bottom edges of the causal block where the tiles may be rectangular with 4 varieties of shape. We denote these varieties as "Main", "Low-Trim", "Left-Trim" and "Corner-Trim" respectively. Under this prescription we partition the causal block and merge the indices as described, resulting in a contraction of fewer albeit larger tensors $T_{M_{a}M_{b}}^{M_{c}M_{d}}$ (w. potentially uneven dimensions).

Referring to Bertini Prosen and Kos for a rigorous justification, we work in the "skeleton approximation" and neglect paths which have any branching - i.e. they can only make upward or rightward turns on the tile grid. Had we used $1\times1$ tiles as opposed to $d\times d$ then this would be equivalent to keeping paths whose branching is supported up to a maximum width of $d$ tiles at any slice along the path. As a consequence, each skeleton path is a sequence of matrix-vector multiplications starting with the input vector.

### The Code

This repository contains our package for computing operator correlations in perturbed dual-unitary circuits (i.e. all the necessary modules & functions therein). There are 3 main modules:

- tensors.py (for generating the "constituent" tensors needed to build a (uniformly-) perturbed dual-unitary circuit - sampled from their property-constrained subsets)
- circuit.py (for performing the tile decomposition, generating the tile zoo, computing skeleton paths, running the path-sum, and actually getting a full light-cone of operator correlations)
- random_unitaries.py (for computations on dual unitary circuits with random anisotropic perturbations)

