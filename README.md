# AttFisher: A Computation of Visual Attention with Fisher Information Matrix


# はじめに
近年，機械学習分野では人の注意機構をヒントとしたAttentionと呼ばれるモデルアーキテクチャが注目されている．
また，神経科学・生理学の分野では，特に視覚型注意に関する研究が盛んで，
top-down attentionとbottom-up attentionが知られている．

- Bottom-up attention: 目的に対して偏りに対する注意．
- Top-down attention: 周りとの違いが顕著な刺激に対する注意．

# Fisher情報量とKL-divergence

### Fisher情報量
未知母数$\theta \in \Theta \subset \mathbb{R}^d$に依存する$\mathbb{R}^m$上の確率変数$X \sim p(x; \theta)$の
対数尤度関数$\ell(\theta \vert x) = \log p(x \vert \theta)$に対して，Score関数

$$
    g(\theta; X) = \nabla_{θ} \ell(\theta; X)
$$

が定義される．
ここで，$g$の1次モーメントは$\mathbb{E}_X[g(\theta | X)] = {\bf 0}$となることが知られており，
2次モーメント${Var}_X[g(\theta | X)] = \mathbb{E}_{X}[{g(\theta | X)}^2]$をFisher情報量行列と呼び，
これは，$\theta$の対数尤度関数$\ell(\theta; X) = \log p(X \vert \theta)$に対するHessian行列の期待値として以下のように定義される．

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}\begin{align}&space;G(\theta)&space;&=&space;\mathbb{E}_{X}[&space;\nabla^2_{\theta}&space;\ell(\theta;&space;X)&space;]&space;\nonumber&space;\\&space;&=&space;\mathbb{E}_{X}&space;\left(&space;\begin{array}{cccc}&space;\frac{\partial^2&space;\ell(\theta;&space;X)}{\partial&space;\theta_1^2}&space;&&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_1&space;\partial&space;\theta_2}&space;&&space;\ldots&space;&&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_1&space;\partial&space;\theta_d}&space;\\&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_2&space;\partial&space;\theta_1}&space;&&space;\frac{\partial^2&space;\ell(\theta;&space;X)}{\partial&space;\theta_2^2}&space;&&space;\ldots&space;&&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_2&space;\partial&space;\theta_d}&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_d&space;\partial&space;\theta_1}&space;&&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_d&space;\partial&space;\theta_2}&space;&&space;\ldots&space;&&space;\frac{\partial^2&space;\ell(\theta;&space;X)}{\partial&space;\theta_d^2}&space;\end{array}&space;\right)&space;\nonumber&space;\end{align}" />

### KL-divergence

確率分布間の擬距離であるKL-divergence

$$
    D_{KL}(p_{\theta_1}, p_{\theta_2}) = \int p(x \vert \theta_1) \log \frac{p(x \vert \theta_1)}{p(x \vert \theta_2)} dx
$$

について，2次までのMaclaurin展開をすると，

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;D_{KL}(p_{\theta},&space;p_{\theta&space;&plus;&space;\Delta&space;\theta})&space;\approx&space;\frac{1}{2}&space;{\Delta&space;\theta}^T&space;G(\theta)&space;\Delta&space;\theta" />

となる．
