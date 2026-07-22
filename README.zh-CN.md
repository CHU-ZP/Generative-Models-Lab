# Generative Models Lab

<p align="right">
<a href="README.md">English</a> | 中文
</p>

<p align="center">
  一组小而清晰的生成模型实践，<br>
  从 latent-variable model 一直延伸到 learned probability flow。
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

这个仓库是四个独立项目的统一入口，不会复制它们的源代码。每个项目继续保留自己的 Git 历史、运行环境、文档和实验结果。

## 项目索引

下面统一使用整洁的展示名称；实际 GitHub 仓库名可以保留原来的 slug。

| 顺序 | 展示名称 | 仓库 | 状态空间 | 生成路径 |
| :---: | --- | --- | --- | --- |
| 01 | **VAE** | [`CHU-ZP/Modular-Vae`](https://github.com/CHU-ZP/Modular-Vae) | MNIST latent space | 采样 latent 后解码 |
| 02 | **Diffusion** | [`CHU-ZP/Modular-Diffusion`](https://github.com/CHU-ZP/Modular-Diffusion) | CIFAR10 pixel 和 latent | 反转高斯加噪路径 |
| 03 | **Discrete Diffusion** | [`CHU-ZP/Discrete-Diffusion`](https://github.com/CHU-ZP/Discrete-Diffusion) | 量化像素和二值 voxel | 反转类别转移链 |
| 04 | **Flow Matching** | [`CHU-ZP/FlowMatching`](https://github.com/CHU-ZP/FlowMatching) | CIFAR10 pixel space | 积分学习到的速度场 |

## 四个项目

<table>
  <tr>
    <td width="50%" valign="top">
      <h3 align="center">01 · VAE</h3>
      <p align="center"><code>latent-variable model</code></p>
      <a href="https://github.com/CHU-ZP/Modular-Vae">
        <img src="https://raw.githubusercontent.com/CHU-ZP/Modular-Vae/main/assets/figures/cnn_reconstructions.png" alt="VAE 在 MNIST 上的重建结果" width="100%">
      </a>
      <p>五组 MNIST 实验在同一概率接口下比较 MLP、CNN、Transformer、beta-VAE 和学习得到的 flow prior。</p>
      <p align="center"><a href="https://github.com/CHU-ZP/Modular-Vae"><strong>查看 VAE →</strong></a></p>
    </td>
    <td width="50%" valign="top">
      <h3 align="center">02 · Diffusion</h3>
      <p align="center"><code>continuous-state denoising</code></p>
      <a href="https://github.com/CHU-ZP/Modular-Diffusion">
        <img src="https://raw.githubusercontent.com/CHU-ZP/Modular-Diffusion/main/results/cifar10_unet_cosine.cond.png" alt="模块化 Diffusion 的 CIFAR10 条件生成结果" width="100%">
      </a>
      <p>在 CIFAR10 pixel 和 latent space 中组合不同骨干网络、schedule、预测目标与 DDPM/DDIM sampler。</p>
      <p align="center"><a href="https://github.com/CHU-ZP/Modular-Diffusion"><strong>查看 Diffusion →</strong></a></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3 align="center">03 · Discrete Diffusion</h3>
      <p align="center"><code>categorical denoising</code></p>
      <a href="https://github.com/CHU-ZP/Discrete-Diffusion">
        <img src="https://raw.githubusercontent.com/CHU-ZP/Discrete-Diffusion/main/results/mnist/generated_samples.png" alt="Discrete Diffusion 的量化 MNIST 生成结果" width="100%">
      </a>
      <p>始终留在离散状态空间中的 categorical diffusion，同时覆盖量化 MNIST 像素和二值 ModelNet10 voxel。</p>
      <p align="center"><a href="https://github.com/CHU-ZP/Discrete-Diffusion"><strong>查看 Discrete Diffusion →</strong></a></p>
    </td>
    <td width="50%" valign="top">
      <h3 align="center">04 · Flow Matching</h3>
      <p align="center"><code>probability flow</code></p>
      <a href="https://github.com/CHU-ZP/FlowMatching">
        <img src="https://raw.githubusercontent.com/CHU-ZP/FlowMatching/main/assets/results/class_grid_euler_050.png" alt="Flow Matching 的 CIFAR10 类别条件生成结果" width="100%">
      </a>
      <p>CIFAR10 上的 Rectified Flow：学习连续速度场，再使用 Euler 或 Heun 从高斯噪声积分到图像。</p>
      <p align="center"><a href="https://github.com/CHU-ZP/FlowMatching"><strong>查看 Flow Matching →</strong></a></p>
    </td>
  </tr>
</table>

## 同一个 Lab，四种视角

```text
VAE                 latent distribution  -> decoder
Diffusion           Gaussian noise path  -> reverse denoising
Discrete Diffusion  categorical path     -> exact discrete reverse chain
Flow Matching       probability path     -> ODE integration
```

四个项目都尽量让数学组件在代码中保持可见，并让每组实验一次只替换系统中的一个部分，方便观察模型、目标函数和 sampler 之间的关系。

## 展示网站

[`index.html`](index.html) 是这个概览的响应式网页版本，可以直接通过 GitHub Pages 发布。推送仓库后，进入 **Settings → Pages → Deploy from a branch → `main` / root**。

本地预览：

```bash
python -m http.server 8000
```

然后打开 `http://localhost:8000`。
