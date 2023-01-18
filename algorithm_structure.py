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
2.

class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, X):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            return X * binomial.sample(X.size()) * (1.0/(1-self.p))
        return X

class MyLinear(nn.Linear):
    def __init__(self, in_feats, out_feats, drop_p, bias=True):
        super(MyLinear, self).__init__(in_feats, out_feats, bias=bias)
        self.custom_dropout = Dropout(p=drop_p)

    def forward(self, input):
        dropout_value = self.custom_dropout(self.weight)
        return F.linear(input, dropout_value, self.bias)

loss_fn = nn.MSELoss(reduction='none')
input = torch.randn(10, 1, requires_grad=True)
target = torch.randn(10, 1)
loss_each = loss_fn(input, target)
loss_all =  torch.mean(loss_each)
loss_all.backward()
other_computation(loss_each.detach())

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