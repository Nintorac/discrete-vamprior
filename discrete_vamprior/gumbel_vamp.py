#%%
import torch
from torch.distributions import Normal
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch import nn

from agoge import AbstractModel

class FeedForward(nn.Module):


    def __init__(self, in_features, hidden_features, out_features):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):

        return self.net(x)


class VampAutoEncoder(AbstractModel):


    def __init__(self, n_priors=10, n_params=784, latent_dim=2, t=0.5, gumbel=True):
        
        super().__init__()

        self.U = nn.Parameter(torch.randn(n_priors, n_params))

        self.t = t
        self.gumbel = gumbel

        self.enc = FeedForward(n_params, n_params*3, latent_dim*2)
        self.dec = FeedForward(latent_dim, n_params*3, n_params)

    def forward(self, x):
        
        if self.gumbel:
            prior_exemplars = RelaxedBernoulli(self.t, logits=self.U).rsample()
        else:
            prior_exemplars = torch.sigmoid(self.U)
        
        prior_mu, prior_var = self.enc(prior_exemplars).chunk(2, -1)
        prior = Normal(prior_mu, (prior_var*0.5).clamp(-5, 4).exp())

        posterior_mu, posterior_var = self.enc(x).chunk(2, -1)
        posterior = Normal(posterior_mu, (posterior_var*0.5).clamp(-5, 4).exp())

        z = posterior.rsample()

        x_hat = self.dec(z)

        return {
            'prior': prior,
            'posterior': posterior,
            'z': z,
            'x_hat': x_hat
        }

if __name__=='__main__':

    net_g = AutoEncoder()
    net = AutoEncoder(gumbel=False)
    x = torch.rand(10, 784)

    net_g(x)
    net(x)
    

# %%
