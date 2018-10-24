# birl
A Python Implementation of Bayesian Inverse Reinforcement Learning (BIRL)

## Directories
- tabular - BIRL implementation that uses a tabular form to represent rewards (as in the original BIRL paper)
- reward - BIRL implementation that calculates reward as a function of state (BIRL learns the weights of this function)
- tp - BIRL implementation that calculates reward as a functions of state AND TP as function of state, action and subsequent state

## Prerequisites

- Python 2.7
- SciPy
- pymdptoolbox (https://media.readthedocs.org/pdf/pymdptoolbox/latest/pymdptoolbox.pdf)
- CFFI
- Python Boost

## Resources

- http://www.boost.org/doc/libs/1_64_0/more/getting_started/unix-variants.html
- http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
- http://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
- https://media.readthedocs.org/pdf/cython-docs2/stable/cython-docs2.pdf

## Compilation

	python setup.py build_ext --inplace