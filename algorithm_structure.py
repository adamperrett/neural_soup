"""
Evolving at the neural level with GD:
1. initialise the population (generate random neurons and weights)
2. present a batch and measure the performance (per neuron)
3. perform gradient descent as determined
4. continue step 2-3 until end criteria
5. present bootstrap (possibly weighted) of training data
6. collect performance of individual neurons and assign a fitness
7. evolve the population and keep the best x%
8. repeat steps 2-7 until end criteria

------------------------------------------------------------------------------

Building architecture:
(Have a randomly connected architecture and use dropout to extract useful architectures?)
1. Initialise with shit loads of connections
2. Train using dropout
3. Combine dropout mask with error to approximate neuron fitness
    Keep track of mask used to extract important connections?
4. Mate and mutate



------------------------------------------------------------------------------

Smoothing activation functions:

activation a =
    max(0, m_0(x - v) + 1) if x < v
    max(0, m_1(x - v) + 1) else
v is the stored value (init data value)
m is a multiplicative factor (init m_0=1 and m_1=-1)

1. sample data to create a seed network, like EDN
2.
"""