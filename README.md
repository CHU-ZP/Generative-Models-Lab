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
      <p>Five MNIST experiments compare MLP, CNN, Transformer, beta-VAE, and a learned flow prior behind one probabilistic interface.</p>
      <p align="center"><a href="https://github.com/CHU-ZP/Modular-Vae"><strong>Explore VAE →</strong></a></p>
    </td>
    <td width="50%" valign="top">
      <h3 align="center">02 · Diffusion</h3>
      <p align="center"><code>continuous-state denoising</code></p>
      <a href="https://github.com/CHU-ZP/Modular-Diffusion">
        <img src="https://raw.githubusercontent.com/CHU-ZP/Modular-Diffusion/main/results/cifar10_unet_cosine.cond.png" alt="Conditional CIFAR10 samples from modular diffusion" width="100%">
      </a>
      <p>Composable pixel- and latent-space diffusion on CIFAR10, with replaceable backbones, schedules, prediction targets, and samplers.</p>
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
      <p>A categorical diffusion engine that stays in the discrete state space for both quantized MNIST pixels and binary ModelNet10 voxels.</p>
      <p align="center"><a href="https://github.com/CHU-ZP/Discrete-Diffusion"><strong>Explore Discrete Diffusion →</strong></a></p>
    </td>
    <td width="50%" valign="top">
      <h3 align="center">04 · Flow Matching</h3>
      <p align="center"><code>probability flow</code></p>
      <a href="https://github.com/CHU-ZP/FlowMatching">
        <img src="https://raw.githubusercontent.com/CHU-ZP/FlowMatching/main/assets/results/class_grid_euler_050.png" alt="Class-conditional CIFAR10 samples from Flow Matching" width="100%">
      </a>
      <p>Rectified Flow on CIFAR10: learn a continuous velocity field and integrate it from Gaussian noise with Euler or Heun.</p>
      <p align="center"><a href="https://github.com/CHU-ZP/FlowMatching"><strong>Explore Flow Matching →</strong></a></p>
    </td>
  </tr>
</table>

## How the Methods Differ

```text
VAE                 latent distribution  -> decoder
Diffusion           Gaussian noise path  -> reverse denoising
Discrete Diffusion  categorical path     -> exact discrete reverse chain
Flow Matching       probability path     -> ODE integration
```

The comparison is not only architectural. Each method defines a different
state space, training target, and route from a simple distribution to a sample.
Reading the projects together makes those modeling choices explicit.

## What You Will Find

Each repository includes:

- a concise explanation of the underlying probability model;
- small modules that correspond directly to the equations;
- configuration-driven experiments and reproducible commands;
- qualitative samples, quantitative results, and implementation notes;
- English and Chinese documentation.

Choose a project above to continue from the mathematical overview to the code
and completed experiments.
