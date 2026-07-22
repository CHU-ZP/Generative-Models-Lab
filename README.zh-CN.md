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
        <img src="assets/diagrams/vae.png" alt="展示 latent 采样、解码和训练目标的 VAE 原理图" width="100%">
      </a>
      <p>学习一个紧凑的潜在分布，再用解码器把潜变量样本转换成数据。</p>
      <p align="center"><a href="https://github.com/CHU-ZP/Modular-Vae"><strong>查看 VAE →</strong></a></p>
    </td>
    <td width="50%" valign="top">
      <h3 align="center">02 · Diffusion</h3>
      <p align="center"><code>continuous-state denoising</code></p>
      <a href="https://github.com/CHU-ZP/Modular-Diffusion">
        <img src="assets/diagrams/diffusion.png" alt="展示迭代去噪和噪声预测训练的 Diffusion 原理图" width="100%">
      </a>
      <p>学习逐步高斯加噪过程的逆过程，通过反复去噪把噪声还原为数据。</p>
      <p align="center"><a href="https://github.com/CHU-ZP/Modular-Diffusion"><strong>查看 Diffusion →</strong></a></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3 align="center">03 · Discrete Diffusion</h3>
      <p align="center"><code>categorical denoising</code></p>
      <a href="https://github.com/CHU-ZP/Discrete-Diffusion">
        <img src="assets/diagrams/discrete-diffusion.png" alt="展示类别反向链和 clean-token prediction 的 Discrete Diffusion 原理图" width="100%">
      </a>
      <p>学习类别扰动过程的逆过程，并让所有中间状态始终保持离散。</p>
      <p align="center"><a href="https://github.com/CHU-ZP/Discrete-Diffusion"><strong>查看 Discrete Diffusion →</strong></a></p>
    </td>
    <td width="50%" valign="top">
      <h3 align="center">04 · Flow Matching</h3>
      <p align="center"><code>probability flow</code></p>
      <a href="https://github.com/CHU-ZP/FlowMatching">
        <img src="assets/diagrams/flow-matching.png" alt="展示速度场积分和训练路径的 Flow Matching 原理图" width="100%">
      </a>
      <p>学习一个连续速度场，把简单噪声分布持续搬运成数据分布。</p>
      <p align="center"><a href="https://github.com/CHU-ZP/FlowMatching"><strong>查看 Flow Matching →</strong></a></p>
    </td>
  </tr>
</table>

## 四种方法有何不同

四个项目的主要区别在于：把什么作为状态、让神经网络预测什么，以及如何从简单分布生成新样本。

### 01 · VAE——Latent-Variable Modeling

VAE 把观测编码为近似 latent posterior，并通过重参数化技巧得到可微的采样：

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

证据下界在重建质量和规则化的 latent space 之间取得平衡：

```math
\mathcal{L}_{\mathrm{VAE}}
=
\mathbb{E}_{q_\phi(z\mid x)}[-\log p_\theta(x\mid z)]
+
\beta\,\mathrm{KL}\!\left(q_\phi(z\mid x)\Vert p(z)\right).
```

生成时只需从先验采样一次，再经过一次 decoder：

```math
z\sim p(z),
\qquad
x\sim p_\theta(x\mid z).
```

### 02 · Diffusion——Gaussian Denoising

Diffusion 定义一个固定的前向过程，在指定时间步把干净数据与高斯噪声混合：

```math
x_t
=
\sqrt{\bar\alpha_t}\,x_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
```

去噪网络可以学习 noise、clean data 或 velocity；经典的 noise-prediction 目标是：

```math
\mathcal{L}_{\mathrm{diff}}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|\epsilon-\epsilon_\theta(x_t,t,c)\right\|^2
\right].
```

采样从高斯噪声开始，反复执行学习到的反向更新。DDPM 的更新具有随机性，DDIM 则可以使用同一个去噪网络沿确定性轨迹生成。

### 03 · Discrete Diffusion——Categorical Denoising

Discrete Diffusion 用类别转移替代高斯噪声。每一步通过转移矩阵保留或替换一个 token：

```math
Q_t=(1-\beta_t)I+\beta_t U,
\qquad
\bar Q_t=Q_1Q_2\cdots Q_t,
\qquad
q(x_t\mid x_0)=\mathrm{Cat}(x_0\bar Q_t).
```

网络直接预测 clean-token 分布：

```math
p_\theta(x_0\mid x_t,t,y)
=
\mathrm{softmax}\!\left(f_\theta(x_t,t,y)\right).
```

生成从 categorical noise 开始，每个反向步骤都把网络预测与精确的离散后验结合起来。在整个过程中，像素始终是量化 token，voxel 也始终保持二值状态。

### 04 · Flow Matching——Continuous Probability Flow

Flow Matching 在高斯噪声 `z` 和数据样本 `x` 之间选择一条概率路径。对于直线 Rectified Flow 路径，目标速度可以直接解析得到：

```math
x_t=(1-t)z+tx,
\qquad
u_t=\frac{d x_t}{dt}=x-z.
```

模型回归这个速度场：

```math
\mathcal{L}_{\mathrm{FM}}
=
\mathbb{E}_{x,z,t}
\left[
\left\|v_\theta(x_t,t)-(x-z)\right\|^2
\right].
```

生成时从噪声出发求解 ODE，并通过 Euler 或 Heun 等数值方法从 `t=0` 积分到 `t=1`：

```math
x_0\sim\mathcal{N}(0,I),
\qquad
\frac{d x_t}{dt}=v_\theta(x_t,t),
\qquad
t:0\rightarrow1.
```

## 每个项目包含什么

每个仓库都包含：

- 对底层概率模型的简明说明；
- 与公式直接对应的小型代码模块；
- 配置驱动的实验和可复现命令；
- 定性样本、定量结果与实现说明；
- 英文和中文文档。

选择上面的任意项目，可以继续从数学概览进入代码实现和完整实验。
