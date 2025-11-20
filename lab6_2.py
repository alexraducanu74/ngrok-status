import numpy as np
import scipy.stats as stats
import arviz as az
import matplotlib.pyplot as plt

# ----------------------------
# Observed data
# ----------------------------
total_calls = 180  # total calls observed
hours_observed = 10
observed_mean = total_calls / hours_observed  # 18 calls per hour

# ----------------------------
# Prior: Gamma distribution
# ----------------------------
# Non-informative prior: alpha=1, beta=0 (flat prior)
alpha_prior = 1
beta_prior = 0

# ----------------------------
# Posterior parameters
# ----------------------------
alpha_post = alpha_prior + total_calls
beta_post = beta_prior + hours_observed

print(f"Posterior parameters: alpha={alpha_post}, beta={beta_post}")

# Posterior distribution is Gamma(alpha_post, beta_post)
posterior_mean = alpha_post / beta_post
posterior_mode = (alpha_post - 1) / beta_post if alpha_post > 1 else 0
posterior_variance = alpha_post / (beta_post ** 2)

print(f"Posterior mean: {posterior_mean:.2f}")
print(f"Posterior mode: {posterior_mode:.2f}")
print(f"Posterior variance: {posterior_variance:.2f}")

# ----------------------------
# Sample from posterior
# ----------------------------
posterior_samples = np.random.gamma(shape=alpha_post, scale=1/beta_post, size=100000)

# ----------------------------
# Compute 94% HDI
# ----------------------------
hdi_94 = az.hdi(posterior_samples, hdi_prob=0.94)
print(f"94% HDI: [{hdi_94[0]:.2f}, {hdi_94[1]:.2f}]")

# ----------------------------
# Plot posterior using ArviZ
# ----------------------------
az.plot_posterior(posterior_samples, hdi_prob=0.94)
plt.title("Posterior distribution of λ (call rate per hour)")
plt.xlabel("λ (calls per hour)")
plt.show()
