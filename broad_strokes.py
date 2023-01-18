'''
This file will aim to provide a template that the rest of the code can be built from.

It is an idea encompassing a spectrum of different areas: ensembles, evolution, gradient descent and network construction.

Basic idea:
- initialise a 'neural soup' that changes its architecture and component neurons in response to input
- the connections (and network size) change as learning progresses

Functional aims:
- strengthen/make useful connections
- weaken/prune unuseful ones
- develop architecture in response to data
- developing high level structures
- sharing of discovered useful properties

Relevant literature:
- Ensembles // each neuron sucks individually but collectively they form an impressive whole
    - remain aware that it is their independence that makes their averaging beneficial
    - the better they perform the better the whole, as long as bias remains distributed
- Evolution // neurons will compete and cooperate with the best surviving
    - a metric of fitness needs to be determined for each neuron
    - genes need to be passed on from most fit individuals
    - worst ones need to die without catastrophic effect on performance
    Things that can be evolved:
        - gd learning params
        - projection prob
        - activation function (make them smooth continuations between ReLU and hat?)
- Gradient descent // model parameters will require fine-tuning to reach high levels of performance
    - can be a neuron specific learning rule
    - cannot require high-level error signals requiring global knowledge
    - parameterised by evolution

Applications:
- unpredictable environments or tasks
- discovering new architectures
- removing need for expert knowledge
- combination of different neuron types

Challenges:
- determining rules of emergence
- allowing quick/efficient development of learning
- constructing layers of neurons
- allowing neuron deletion to be non-catastrophic
- determining an useful fitness function for neurons
- appropriately leveraging the benefits of ensembles
- stopping gradient descent from causing catastrophic forgetting
- recurrent connections
'''