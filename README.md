# Generative Models Lab

<p align="right">
English | <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  A curated collection of small, readable generative-model implementations,<br>
  moving from latent-variable models to learned probability flows.
</p>

<p align="center">
  <a href="https://github.com/CHU-ZP/Modular-Vae"><strong>VAE</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/CHU-ZP/Modular-Diffusion"><strong>Diffusion</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/CHU-ZP/Discrete-Diffusion"><strong>Discrete Diffusion</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/CHU-ZP/FlowMatching"><strong>Flow Matching</strong></a>
</p>

Generative Models Lab is a hands-on study of four generative-model families.
Each project starts from the probabilistic idea, follows it into a modular
PyTorch implementation, and closes the loop with trained examples and generated
results.

## Start Here

Follow the projects from `01` to `04` for a path from latent-variable modeling
to iterative denoising and continuous-time transport. Each repository is also
self-contained, so readers can jump directly to the method they want to study.

| Step | Project | Data | Learned object | Generation |
| :---: | --- | --- | --- | --- |
| 01 | [**VAE**](https://github.com/CHU-ZP/Modular-Vae) | MNIST | posterior, prior, and decoder | sample a latent and decode it |
| 02 | [**Diffusion**](https://github.com/CHU-ZP/Modular-Diffusion) | CIFAR10 pixels and latents | noise, data, or velocity prediction | reverse a Gaussian corruption path |
| 03 | [**Discrete Diffusion**](https://github.com/CHU-ZP/Discrete-Diffusion) | quantized MNIST and binary ModelNet10 voxels | clean-token probabilities | reverse a categorical transition chain |
| 04 | [**Flow Matching**](https://github.com/CHU-ZP/FlowMatching) | CIFAR10 pixels | continuous velocity field | integrate an ODE from noise to data |

## Explore the Projects

<table>
  <tr>
    <td width="50%" valign="top">
      <h3 align="center">01 · VAE</h3>
      <p align="center"><code>latent-variable model</code></p>
      <a href="https://github.com/CHU-ZP/Modular-Vae">
        <img src="https://raw.githubusercontent.com/CHU-ZP/Modular-Vae/main/assets/figures/cnn_reconstructions.png" alt="VAE reconstructions on MNIST" width="100%">
      </a>
      <p>Learns a compact latent distribution and a decoder that turns latent samples into data.</p>
      <p align="center"><a href="https://github.com/CHU-ZP/Modular-Vae"><strong>Explore VAE →</strong></a></p>
    </td>
    <td width="50%" valign="top">
      <h3 align="center">02 · Diffusion</h3>
      <p align="center"><code>continuous-state denoising</code></p>
      <a href="https://github.com/CHU-ZP/Modular-Diffusion">
        <img src="https://raw.githubusercontent.com/CHU-ZP/Modular-Diffusion/main/results/cifar10_unet_cosine.cond.png" alt="Conditional CIFAR10 samples from modular diffusion" width="100%">
      </a>
      <p>Learns to reverse a gradual Gaussian noising process, turning noise into data through repeated denoising.</p>
      <p align="center"><a href="https://github.com/CHU-ZP/Modular-Diffusion"><strong>Explore Diffusion →</strong></a></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3 align="center">03 · Discrete Diffusion</h3>
      <p align="center"><code>categorical denoising</code></p>
      <a href="https://github.com/CHU-ZP/Discrete-Diffusion">
        <img src="https://raw.githubusercontent.com/CHU-ZP/Discrete-Diffusion/main/results/mnist/generated_samples.png" alt="Quantized MNIST samples from discrete diffusion" width="100%">
      </a>
      <p>Learns to reverse a categorical corruption process while keeping every state discrete.</p>
      <p align="center"><a href="https://github.com/CHU-ZP/Discrete-Diffusion"><strong>Explore Discrete Diffusion →</strong></a></p>
    </td>
    <td width="50%" valign="top">
      <h3 align="center">04 · Flow Matching</h3>
      <p align="center"><code>probability flow</code></p>
      <a href="https://github.com/CHU-ZP/FlowMatching">
        <img src="https://raw.githubusercontent.com/CHU-ZP/FlowMatching/main/assets/results/class_grid_euler_050.png" alt="Class-conditional CIFAR10 samples from Flow Matching" width="100%">
      </a>
      <p>Learns a continuous velocity field that transports a simple noise distribution into the data distribution.</p>
      <p align="center"><a href="https://github.com/CHU-ZP/FlowMatching"><strong>Explore Flow Matching →</strong></a></p>
    </td>
  </tr>
</table>

## How the Methods Differ

The four projects differ in what they treat as the state, what the neural
network predicts, and how a new sample is produced.

### 01 · VAE — Latent-Variable Modeling

A VAE encodes an observation into an approximate latent posterior and uses the
reparameterization trick to draw a differentiable sample:

```math
q_\phi(z\mid x)
=
\mathcal{N}\!\left(
z;\mu_\phi(x),\mathrm{diag}(\sigma_\phi^2(x))
\right),
\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,
\quad
\epsilon\sim\mathcal{N}(0,I).
```

The evidence lower bound balances reconstruction with a regularized latent
space:

```math
\mathcal{L}_{\mathrm{VAE}}
=
\mathbb{E}_{q_\phi(z\mid x)}[-\log p_\theta(x\mid z)]
+
\beta\,\mathrm{KL}\!\left(q_\phi(z\mid x)\Vert p(z)\right).
```

Generation is a single latent sample followed by one decoder pass:

```math
z\sim p(z),
\qquad
x\sim p_\theta(x\mid z).
```

### 02 · Diffusion — Gaussian Denoising

Diffusion defines a fixed forward process that mixes clean data with Gaussian
noise at a chosen timestep:

```math
x_t
=
\sqrt{\bar\alpha_t}\,x_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
```

A denoiser learns noise, clean data, or velocity; the classic noise-prediction
objective is:

```math
\mathcal{L}_{\mathrm{diff}}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|\epsilon-\epsilon_\theta(x_t,t,c)\right\|^2
\right].
```

Sampling starts from Gaussian noise and repeatedly applies the learned reverse
updates. DDPM makes these updates stochastic, while DDIM can follow a
deterministic trajectory with the same denoiser.

### 03 · Discrete Diffusion — Categorical Denoising

Discrete diffusion replaces Gaussian noise with a categorical transition. At
each step, a token is preserved or replaced through a transition matrix:

```math
Q_t=(1-\beta_t)I+\beta_t U,
\qquad
\bar Q_t=Q_1Q_2\cdots Q_t,
\qquad
q(x_t\mid x_0)=\mathrm{Cat}(x_0\bar Q_t).
```

The network predicts the clean-token distribution directly:

```math
p_\theta(x_0\mid x_t,t,y)
=
\mathrm{softmax}\!\left(f_\theta(x_t,t,y)\right).
```

Generation starts from categorical noise and combines this prediction with the
exact discrete posterior at every reverse step. Pixels remain quantized tokens
and voxels remain binary throughout the entire process.

### 04 · Flow Matching — Continuous Probability Flow

Flow Matching chooses a probability path between Gaussian noise `z` and a data
sample `x`. For the straight Rectified Flow path, its target velocity is known
analytically:

```math
x_t=(1-t)z+tx,
\qquad
u_t=\frac{d x_t}{dt}=x-z.
```

The model regresses this velocity field:

```math
\mathcal{L}_{\mathrm{FM}}
=
\mathbb{E}_{x,z,t}
\left[
\left\|v_\theta(x_t,t)-(x-z)\right\|^2
\right].
```

Generation then solves an ODE from noise to data, using a numerical method such
as Euler or Heun:

```math
x_0\sim\mathcal{N}(0,I),
\qquad
\frac{d x_t}{dt}=v_\theta(x_t,t),
\qquad
t:0\rightarrow1.
```

## What You Will Find

Each repository includes:

- a concise explanation of the underlying probability model;
- small modules that correspond directly to the equations;
- configuration-driven experiments and reproducible commands;
- qualitative samples, quantitative results, and implementation notes;
- English and Chinese documentation.

Choose a project above to continue from the mathematical overview to the code
and completed experiments.
