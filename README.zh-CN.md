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

Generative Models Lab 是对四类生成模型的实践性学习。每个项目都从概率建模思想出发，将公式对应到模块化 PyTorch 实现，并通过完成的训练实验和生成结果把整个过程串联起来。

## 从这里开始

按照 `01` 到 `04` 阅读，可以从 latent-variable model 逐步走向迭代去噪和连续时间分布搬运。每个仓库也可以独立阅读，因此也可以直接进入最感兴趣的方法。

| 顺序 | 项目 | 数据 | 学习对象 | 生成方式 |
| :---: | --- | --- | --- | --- |
| 01 | [**VAE**](https://github.com/CHU-ZP/Modular-Vae) | MNIST | 后验、先验和解码器 | 采样 latent 后解码 |
| 02 | [**Diffusion**](https://github.com/CHU-ZP/Modular-Diffusion) | CIFAR10 pixel 和 latent | noise、data 或 velocity prediction | 反转高斯加噪路径 |
| 03 | [**Discrete Diffusion**](https://github.com/CHU-ZP/Discrete-Diffusion) | 量化 MNIST 和二值 ModelNet10 voxel | clean-token 概率 | 反转类别转移链 |
| 04 | [**Flow Matching**](https://github.com/CHU-ZP/FlowMatching) | CIFAR10 pixel | 连续速度场 | 从噪声出发积分 ODE 到数据 |

## 探索四个项目

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

## 四种方法有何不同

```text
VAE                 latent distribution  -> decoder
Diffusion           Gaussian noise path  -> reverse denoising
Discrete Diffusion  categorical path     -> exact discrete reverse chain
Flow Matching       probability path     -> ODE integration
```

这些方法的差别不只是网络架构。它们选择了不同的状态空间、训练目标，以及从简单分布到生成样本的路径。把四个项目放在一起阅读，可以更清楚地看到这些建模选择。

## 每个项目包含什么

每个仓库都包含：

- 对底层概率模型的简明说明；
- 与公式直接对应的小型代码模块；
- 配置驱动的实验和可复现命令；
- 定性样本、定量结果与实现说明；
- 英文和中文文档。

选择上面的任意项目，可以继续从数学概览进入代码实现和完整实验。
