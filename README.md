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

This repository is the front door to four independent projects. It does not
duplicate their source code: each project keeps its own history, environment,
documentation, and experiments.

## The Collection

The display names are normalized below even when the underlying GitHub
repository uses a different slug.

| Order | Display name | Repository | State space | Generation route |
| :---: | --- | --- | --- | --- |
| 01 | **VAE** | [`CHU-ZP/Modular-Vae`](https://github.com/CHU-ZP/Modular-Vae) | MNIST latent space | sample a latent, then decode |
| 02 | **Diffusion** | [`CHU-ZP/Modular-Diffusion`](https://github.com/CHU-ZP/Modular-Diffusion) | CIFAR10 pixels and latents | reverse a Gaussian corruption path |
| 03 | **Discrete Diffusion** | [`CHU-ZP/Discrete-Diffusion`](https://github.com/CHU-ZP/Discrete-Diffusion) | quantized pixels and binary voxels | reverse a categorical transition chain |
| 04 | **Flow Matching** | [`CHU-ZP/FlowMatching`](https://github.com/CHU-ZP/FlowMatching) | CIFAR10 pixel space | integrate a learned velocity field |

## Projects

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

## One Lab, Four Views

```text
VAE                 latent distribution  -> decoder
Diffusion           Gaussian noise path  -> reverse denoising
Discrete Diffusion  categorical path     -> exact discrete reverse chain
Flow Matching       probability path     -> ODE integration
```

Across the collection, mathematical components remain visible in the code and
experiments change one part of the system at a time. The goal is to make the
relationship between model, objective, and sampler easy to inspect.

## Website

[`index.html`](index.html) is a responsive version of this overview and can be
served directly with GitHub Pages. After pushing the repository, select
**Settings → Pages → Deploy from a branch → `main` / root**.

For a local preview:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.
