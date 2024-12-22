import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import gpytorch

#-------------------------------------------------------------
# As in example1, a simple multidimensional cubic

def cubic(x: torch.Tensor) -> torch.Tensor:
    # input x is (..., d)
    # output is (..., 1)
    return (x**3).sum(dim=-1, keepdim=True) / x.shape[-1]

# simple MLP

class MLPLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        '''
        MLP Layer with ReLU activations
        '''
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias = False)
        self.norm = nn.BatchNorm1d(out_features)
        self.act = nn.ReLU()

    def forward(self, x: torch.tensor):
        return self.act(self.norm(self.linear(x)))

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, num_hidden: int):
        '''
        MLP with GeLU activations
        '''
        super(MLP, self).__init__()
        layers = [MLPLayer(in_features, hidden_features)] \
              + [MLPLayer(hidden_features, hidden_features) for ii in range(num_hidden)] \
              + [nn.Linear(hidden_features, out_features, bias = True)] 
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)



# GP model with LogNormal Prior

class GPModelDefault(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelDefault, self).__init__(train_x, train_y, likelihood)
        self.num_dims = train_x.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=self.num_dims)
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# GP model with LogNormal Prior

class GPModelWithPrior(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithPrior, self).__init__(train_x, train_y, likelihood)
        self.num_dims = train_x.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean()
        lengthscale_prior = gpytorch.priors.LogNormalPrior(np.log(self.num_dims)/2, 1.0)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=self.num_dims,
                                       lengthscale_prior=lengthscale_prior,
                                       )
            )
        
        # Initialize lengthscale to its prior mean
        self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
#-----------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setting up training data
aspect_ratio = 5.0 # num samples n / parameter dimension d
d = 400
n = int(aspect_ratio * d)

train_x = 2*torch.rand((n,d))-1
train_y = cubic(train_x)

test_x = 2*torch.rand((2*n,d))-1
test_y = cubic(test_x)

train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)

# mlp specification and training

hidden_features = 128
num_hidden = 5
mlp_model = MLP(in_features=d, hidden_features=hidden_features, out_features=1, num_hidden=num_hidden)
mlp_model.to(device)

weight_decay = 1e-1
learning_rate = 1e-2

# candidate parameters
param_dict = {pn: p for pn, p in mlp_model.named_parameters()}
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

# weight decay applied only to tensor weights of order >= 2 (i.e., not biases)
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {"params": decay_params, 'weight_decay': weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0}
        ]

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
criterion = torch.nn.MSELoss(reduction='mean')

max_steps = 2000
losses = {"train": [], "test": []}

print("Training MLP \n")
for step in range(max_steps):
    mlp_model.train()
    optimizer.zero_grad()

    pred = mlp_model(train_x)
    loss = criterion(pred, train_y)
    loss.backward()
    optimizer.step()

    losses["train"].append(loss.item())

    mlp_model.eval()
    with torch.no_grad():
        pred = mlp_model(test_x)
        loss = criterion(pred, test_y)
    losses["test"].append(loss.item())

    if step % 100 == 0:
        train_loss = losses["train"][-1]
        test_loss = losses["test"][-1]
        print(f"step {step} | train loss: {train_loss} | test_loss = {test_loss}")


# Default GP specification and training
learning_rate = 1e-1
gp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp_model = GPModelDefault(train_x, train_y.squeeze(), gp_likelihood)

gp_model.train()
gp_likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(gp_model.parameters(), lr=learning_rate)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_likelihood, gp_model)

print("Training GP \n")
for step in range(max_steps):
    optimizer.zero_grad()
    pred = gp_model(train_x)
    loss = -mll(pred, train_y.squeeze())
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step} | train loss (-mll): {loss.item()}")

# GP with LogNormal prior on lengthscale

gplnp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
gplnp_model = GPModelWithPrior(train_x, train_y.squeeze(), gplnp_likelihood)

gplnp_model.train()
gplnp_likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(gplnp_model.parameters(), lr=learning_rate)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(gplnp_likelihood, gplnp_model)

print("Training GP \n")
for step in range(max_steps):
    optimizer.zero_grad()
    pred = gplnp_model(train_x)
    loss = -mll(pred, train_y.squeeze())
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step} | train loss (-mll): {loss.item()}")

# Plotting

mlp_model.eval()
with torch.no_grad():
    mlp_pred_train = mlp_model(train_x)
    mlp_pred_test = mlp_model(test_x)

gp_model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    gp_preds_train = gp_likelihood(gp_model(train_x))
    gp_preds_test = gp_likelihood(gp_model(test_x))

mean_gp_pred_train = gp_preds_train.mean
std_gp_pred_train = torch.sqrt(gp_preds_train.variance)
mean_gp_pred_test = gp_preds_test.mean
std_gp_pred_test = torch.sqrt(gp_preds_test.variance)

gplnp_model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    gplnp_preds_train = gplnp_likelihood(gplnp_model(train_x))
    gplnp_preds_test = gplnp_likelihood(gplnp_model(test_x))

mean_gplnp_pred_train = gplnp_preds_train.mean
std_gplnp_pred_train = torch.sqrt(gplnp_preds_train.variance)
mean_gplnp_pred_test = gplnp_preds_test.mean
std_gplnp_pred_test = torch.sqrt(gplnp_preds_test.variance)

fig = plt.figure(figsize=(17,5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

ax1.scatter(train_y.squeeze(), mlp_pred_train.squeeze(), c='r', s=1, alpha = 0.4, label='train')
ax1.scatter(test_y.squeeze(), mlp_pred_test.squeeze(), c='b', s=1, alpha = 0.4, label='test')
lims = [min(*ax1.get_xlim(), *ax1.get_ylim()), max(*ax1.get_xlim(), *ax1.get_ylim())]
ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax1.set_aspect('equal')
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_title("Parity plot")
ax1.set_xlabel("Data")
ax1.set_ylabel("Prediction")
ax1.set_title("MLP")
ax1.legend(loc="upper right")

ax2.scatter(train_y.squeeze(), mean_gp_pred_train, c='r', s=5, alpha = 0.4, label='train')
ax2.errorbar(train_y.squeeze(), mean_gp_pred_train, yerr=2*std_gp_pred_train, c='r', fmt='o', alpha = 0.01)
ax2.scatter(test_y.squeeze(), mean_gp_pred_test, c='b', s=5, alpha = 0.4, label='test')
ax2.errorbar(test_y.squeeze(), mean_gp_pred_test, yerr=2*std_gp_pred_test, c='b', fmt='o', alpha = 0.01)
lims = [min(*ax2.get_xlim(), *ax2.get_ylim()), max(*ax2.get_xlim(), *ax2.get_ylim())]
ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax2.set_aspect('equal')
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set_title("Parity plot")
ax2.set_xlabel("Data")
ax2.set_ylabel("Prediction")
ax2.set_title("Default GP (mean +/- 2$\sigma$)")
ax2.legend(loc="upper right")

ax3.scatter(train_y.squeeze(), mean_gplnp_pred_train, c='r', s=5, alpha = 0.4, label='train')
ax3.errorbar(train_y.squeeze(), mean_gplnp_pred_train, yerr=2*std_gplnp_pred_train, c='r', fmt='o', alpha = 0.01)
ax3.scatter(test_y.squeeze(), mean_gplnp_pred_test, c='b', s=5, alpha = 0.4, label='test')
ax3.errorbar(test_y.squeeze(), mean_gplnp_pred_test, yerr=2*std_gplnp_pred_test, c='b', fmt='o', alpha = 0.01)
lims = [min(*ax3.get_xlim(), *ax3.get_ylim()), max(*ax3.get_xlim(), *ax3.get_ylim())]
ax3.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax3.set_aspect('equal')
ax3.set_xlim(lims)
ax3.set_ylim(lims)
ax3.set_title("Parity plot")
ax3.set_xlabel("Data")
ax3.set_ylabel("Prediction")
ax3.set_title("GP with lengthscale prior (mean +/- 2$\sigma$)")
ax3.legend(loc="upper right")

fig.suptitle(f'Fitting a cubic in {d}-dimensions with {n} samples', fontweight="bold")
plt.savefig(f"parity-{n}_samples-{d}_dimensions.png")
plt.close()
