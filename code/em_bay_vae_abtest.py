
###########Code Experiment for Masked Campaign Treatment#############
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind

## read data
data_abtest = pd.read_csv('ab_sales.csv')

# Initialize Parameters
mu_C = np.random.uniform(40, 60)  # Initial guess for Control mean
sigma_C = np.random.uniform(5, 15)  # Initial guess for Control STD
mu_T = np.random.uniform(50, 70)  # Initial guess for Treatment mean
sigma_T = np.random.uniform(10, 20)  # Initial guess for Treatment STD
pi_C = 0.5  # Initial guess for mixing proportion of Control group
pi_T = 0.5  # Initial guess for mixing proportion of Treatment group

# Define the log-likelihood function
def log_likelihood(data, mu_C, sigma_C, mu_T, sigma_T, pi_C, pi_T):
    likelihood_control = pi_C * norm.pdf(data, mu_C, sigma_C)
    likelihood_treatment = pi_T * norm.pdf(data, mu_T, sigma_T)
    return np.sum(np.log(likelihood_control + likelihood_treatment))

# EM Algorithm
max_iter = 100  # Maximum number of iterations
tolerance = 1e-6  # Convergence criterion

log_likelihoods = []
data = data_abtest["sales"]

for iteration in range(max_iter):
    # E-Step: Calculate responsibilities (posterior probabilities)
    resp_control = pi_C * norm.pdf(data, mu_C, sigma_C)
    resp_treatment = pi_T * norm.pdf(data, mu_T, sigma_T)
    total_resp = resp_control + resp_treatment

    gamma_control = resp_control / total_resp  # Responsibility for Control group
    gamma_treatment = resp_treatment / total_resp  # Responsibility for Treatment group

    # M-Step: Update parameters
    N_C = np.sum(gamma_control)  # Effective number of points in Control group
    N_T = np.sum(gamma_treatment)  # Effective number of points in Treatment group

    # Update means
    mu_C = np.sum(gamma_control * data) / N_C
    mu_T = np.sum(gamma_treatment * data) / N_T

    # Update standard deviations
    sigma_C = np.sqrt(np.sum(gamma_control * (data - mu_C)**2) / N_C)
    sigma_T = np.sqrt(np.sum(gamma_treatment * (data - mu_T)**2) / N_T)

    # Update mixing proportions
    pi_C = N_C / n_samples
    pi_T = N_T / n_samples

    # Compute log-likelihood and check for convergence
    current_log_likelihood = log_likelihood(data, mu_C, sigma_C, mu_T, sigma_T, pi_C, pi_T)
    log_likelihoods.append(current_log_likelihood)

    if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tolerance:
        print(f"Converged at iteration {iteration}")
        break

# Output the results
print(f"Final parameters:")
print(f"Control Mean (mu_C): {mu_C:.2f}, Control STD (sigma_C): {sigma_C:.2f}, Mixing Proportion (pi_C): {pi_C:.2f}")
print(f"Treatment Mean (mu_T): {mu_T:.2f}, Treatment STD (sigma_T): {sigma_T:.2f}, Mixing Proportion (pi_T): {pi_T:.2f}")

# Compare the means for inference
if mu_T > mu_C:
    print("The Treatment group outperforms the Control group.")
else:
    print("The Control group outperforms the Treatment group.")

# Step 4: Perform a T-test for comparison
labeled_data = data_with_labels.dropna()
control_group = labeled_data[labeled_data["group"] == 0]["sales"]
treatment_group = labeled_data[labeled_data["group"] == 1]["sales"]

# Perform two-sample T-test
t_stat, p_value = ttest_ind(control_group, treatment_group, equal_var=False)
print(f"T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.4f}")

if p_value < 0.05:
    print("T-test: The difference between groups is statistically significant.")
else:
    print("T-test: The difference between groups is not statistically significant.")
    
    

############Code Experiment for Observed Campaign Treatment####################################    

import numpy as np
import pandas as pd
from scipy.stats import norm

## use the same data
data = data_abtest.copy()

# Initialize Parameters
mu_C = np.random.uniform(40, 60)  # Initial guess for Control mean
sigma_C = np.random.uniform(5, 15)  # Initial guess for Control STD
mu_T = np.random.uniform(50, 70)  # Initial guess for Treatment mean
sigma_T = np.random.uniform(10, 20)  # Initial guess for Treatment STD

# Define the log-likelihood function
def log_likelihood(data, mu_C, sigma_C, mu_T, sigma_T):
    likelihood_control = norm.pdf(data[data["group"] == 0]["sales"], mu_C, sigma_C)
    likelihood_treatment = norm.pdf(data[data["group"] == 1]["sales"], mu_T, sigma_T)
    return np.sum(np.log(likelihood_control)) + np.sum(np.log(likelihood_treatment))

# EM Algorithm (simplified due to observed group labels)
max_iter = 100  # Maximum number of iterations
tolerance = 1e-6  # Convergence criterion

log_likelihoods = []

for iteration in range(max_iter):
    # E-Step: No need for responsibilities as group labels are observed

    # M-Step: Update parameters
    control_data = data[data["group"] == 0]["sales"]
    treatment_data = data[data["group"] == 1]["sales"]

    # Update means
    mu_C = np.mean(control_data)
    mu_T = np.mean(treatment_data)

    # Update standard deviations
    sigma_C = np.std(control_data, ddof=1)
    sigma_T = np.std(treatment_data, ddof=1)

    # Compute log-likelihood and check for convergence
    current_log_likelihood = log_likelihood(data, mu_C, sigma_C, mu_T, sigma_T)
    log_likelihoods.append(current_log_likelihood)

    if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tolerance:
        print(f"Converged at iteration {iteration}")
        break

# Output the results
print(f"Final parameters:")
print(f"Control Mean (mu_C): {mu_C:.2f}, Control STD (sigma_C): {sigma_C:.2f}")
print(f"Treatment Mean (mu_T): {mu_T:.2f}, Treatment STD (sigma_T): {sigma_T:.2f}")

# Compare the means for inference
if mu_T > mu_C:
    print("The Treatment group outperforms the Control group.")
else:
    print("The Control group outperforms the Treatment group.")
    
    
################Code Experiment for Bayesian A/B Testing#####################    

import numpy as np
import pandas as pd
from scipy.stats import invgamma, norm
import matplotlib.pyplot as plt

# Define Priors
mu_0 = 0  # Prior mean for group means
tau_2 = 10  # Prior variance for group means
alpha = 2  # Shape parameter for inverse-gamma prior
beta = 2  # Scale parameter for inverse-gamma prior

# Bayesian Posterior Estimation
# Known variance for simplicity
sigma_C = 10
sigma_T = 12

# Compute posterior parameters for Control group
control_data = data[data["group"] == 0]["sales"]
n_C = len(control_data)
x_bar_C = np.mean(control_data)

post_mu_C = (mu_0 / tau_2 + n_C * x_bar_C / sigma_C**2) / (1 / tau_2 + n_C / sigma_C**2)
post_var_C = 1 / (1 / tau_2 + n_C / sigma_C**2)

# Compute posterior parameters for Treatment group
treatment_data = data[data["group"] == 1]["sales"]
n_T = len(treatment_data)
x_bar_T = np.mean(treatment_data)

post_mu_T = (mu_0 / tau_2 + n_T * x_bar_T / sigma_T**2) / (1 / tau_2 + n_T / sigma_T**2)
post_var_T = 1 / (1 / tau_2 + n_T / sigma_T**2)

# Monte Carlo Sampling
num_samples = 10000
samples_mu_C = np.random.normal(post_mu_C, np.sqrt(post_var_C), num_samples)
samples_mu_T = np.random.normal(post_mu_T, np.sqrt(post_var_T), num_samples)

# Compute probability that Treatment mean > Control mean
prob_treatment_better = np.mean(samples_mu_T > samples_mu_C)

# Step 5: Results
print(f"Posterior mean for Control group: {post_mu_C:.2f}, Variance: {post_var_C:.2f}")
print(f"Posterior mean for Treatment group: {post_mu_T:.2f}, Variance: {post_var_T:.2f}")
print(f"Probability that Treatment mean > Control mean: {prob_treatment_better:.4f}")

# Step 6: Visualization
plt.figure(figsize=(10, 6))
plt.hist(samples_mu_C, bins=50, alpha=0.6, label='Posterior of Control Mean')
plt.hist(samples_mu_T, bins=50, alpha=0.6, label='Posterior of Treatment Mean')
plt.axvline(x=post_mu_C, color='blue', linestyle='--', label='Control Mean')
plt.axvline(x=post_mu_T, color='orange', linestyle='--', label='Treatment Mean')
plt.title('Posterior Distributions of Group Means')
plt.xlabel('Mean Sales')
plt.ylabel('Frequency')
plt.legend()
plt.show()

############Code Experiment: VAE for A/B Testing with Masked Treatment############

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

data = data_abtest.copy()
data["group"] = np.nan  # Mask the group labels

# Normalize sales data
sales_data = (data["sales"] - np.mean(data["sales"])) / np.std(data["sales"])
sales_data = sales_data.values.astype(np.float32).reshape(-1, 1)

# Define VAE Components
latent_dim = 2
input_dim = 1
hidden_dim = 64

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.z_mean = nn.Linear(hidden_dim // 2, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        h = self.encoder(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_log_var

# Initialize the VAE model
vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Loss Function
def loss_function(x, x_reconstructed, z_mean, z_log_var):
    reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x.size(0)
    return reconstruction_loss + kl_loss

# Train VAE Model
sales_tensor = torch.tensor(sales_data)
epochs = 50
batch_size = 32
vae.train()

for epoch in range(epochs):
    permutation = torch.randperm(sales_tensor.size(0))
    epoch_loss = 0
    for i in range(0, sales_tensor.size(0), batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        batch = sales_tensor[indices]
        x_reconstructed, z_mean, z_log_var = vae(batch)
        loss = loss_function(batch, x_reconstructed, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / (sales_tensor.size(0) // batch_size):.4f}")

# Step 4: Inference and Analysis
vae.eval()
with torch.no_grad():
    x_reconstructed, z_mean, z_log_var = vae(sales_tensor)
    z_mean = z_mean.numpy()

control_latents = z_mean[:control_size]
treatment_latents = z_mean[control_size:]

# Visualize latent space
plt.figure(figsize=(8, 6))
plt.scatter(control_latents[:, 0], control_latents[:, 1], alpha=0.6, label="Control")
plt.scatter(treatment_latents[:, 0], treatment_latents[:, 1], alpha=0.6, label="Treatment")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Latent Space Representation")
plt.legend()
plt.show()

# Generate new samples for A/B testing analysis
latent_samples = torch.randn(1000, latent_dim)
with torch.no_grad():
    generated_samples = vae.decoder(latent_samples).numpy()

plt.hist(generated_samples, bins=50, alpha=0.7, label="Generated Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Generated Sales Data from Latent Space")
plt.legend()
plt.show()

# Additional Analysis
# Compute latent means and variances for control and treatment groups
control_mean = np.mean(control_latents, axis=0)
control_var = np.var(control_latents, axis=0)
treatment_mean = np.mean(treatment_latents, axis=0)
treatment_var = np.var(treatment_latents, axis=0)

print("Latent Space Statistics:")
print(f"Control Group Mean: {control_mean}, Variance: {control_var}")
print(f"Treatment Group Mean: {treatment_mean}, Variance: {treatment_var}")

# Compute the difference in latent means
diff_mean = treatment_mean - control_mean
print(f"Difference in Latent Means (Treatment - Control): {diff_mean}")

# Compute the Euclidean distance between latent group centers
euclidean_distance = np.linalg.norm(treatment_mean - control_mean)
print(f"Euclidean Distance Between Latent Group Centers: {euclidean_distance:.4f}")

###########Code Experiment: VAE for A/B Testing with Observed Treatment############

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# use the same data
data = data_abtest.copy()

# Normalize sales data
sales_data = (data["sales"] - np.mean(data["sales"])) / np.std(data["sales"])
sales_data = sales_data.values.astype(np.float32).reshape(-1, 1)

group_data = data["group"].values.astype(np.float32).reshape(-1, 1)
input_data = np.hstack((sales_data, group_data))

# Define VAE Components
latent_dim = 2
input_dim = 2
hidden_dim = 64

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.z_mean = nn.Linear(hidden_dim // 2, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        h = self.encoder(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_log_var

# Initialize the VAE model
vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Loss Function
def loss_function(x, x_reconstructed, z_mean, z_log_var):
    reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x.size(0)
    return reconstruction_loss + kl_loss

# Train VAE Model
input_tensor = torch.tensor(input_data)
epochs = 50
batch_size = 32
vae.train()

for epoch in range(epochs):
    permutation = torch.randperm(input_tensor.size(0))
    epoch_loss = 0
    for i in range(0, input_tensor.size(0), batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        batch = input_tensor[indices]
        x_reconstructed, z_mean, z_log_var = vae(batch)
        loss = loss_function(batch, x_reconstructed, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / (input_tensor.size(0) // batch_size):.4f}")

# Inference and Analysis
vae.eval()
with torch.no_grad():
    x_reconstructed, z_mean, z_log_var = vae(input_tensor)
    z_mean = z_mean.numpy()

control_latents = z_mean[:control_size]
treatment_latents = z_mean[control_size:]

# Visualize latent space
plt.figure(figsize=(8, 6))
plt.scatter(control_latents[:, 0], control_latents[:, 1], alpha=0.6, label="Control")
plt.scatter(treatment_latents[:, 0], treatment_latents[:, 1], alpha=0.6, label="Treatment")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Latent Space Representation")
plt.legend()
plt.show()

# Generate new samples for A/B testing analysis
latent_samples = torch.randn(1000, latent_dim)
with torch.no_grad():
    generated_samples = vae.decoder(latent_samples).numpy()

plt.hist(generated_samples[:, 0], bins=50, alpha=0.7, label="Generated Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Generated Sales Data from Latent Space")
plt.legend()
plt.show()

# Additional Analysis
# Compute latent means and variances for control and treatment groups
control_mean = np.mean(control_latents, axis=0)
control_var = np.var(control_latents, axis=0)
treatment_mean = np.mean(treatment_latents, axis=0)
treatment_var = np.var(treatment_latents, axis=0)

print("Latent Space Statistics:")
print(f"Control Group Mean: {control_mean}, Variance: {control_var}")
print(f"Treatment Group Mean: {treatment_mean}, Variance: {treatment_var}")

# Compute the difference in latent means
diff_mean = treatment_mean - control_mean
print(f"Difference in Latent Means (Treatment - Control): {diff_mean}")

# Compute the Euclidean distance between latent group centers
euclidean_distance = np.linalg.norm(treatment_mean - control_mean)
print(f"Euclidean Distance Between Latent Group Centers: {euclidean_distance:.4f}")
