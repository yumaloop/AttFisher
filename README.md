# AttFisher: A Computation of Visual Attention with Fisher Information Matrix


# はじめに
近年，機械学習分野では人の注意機構をヒントとしたAttentionと呼ばれるモデルアーキテクチャが注目されている．
また，神経科学・生理学の分野では，特に視覚型注意に関する研究が盛んで，
top-down attentionとbottom-up attentionが知られている．

- Bottom-up attention: 目的に対して偏りに対する注意．
- Top-down attention: 周りとの違いが顕著な刺激に対する注意．

# Fisher情報量とKL-divergence

### Fisher情報量
未知母数
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\theta&space;\in&space;\mathbb{R}^d" />
に依存する
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\mathbb{R}^m" />
上の確率変数
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;X&space;\sim&space;p(x;\theta)" />
の
対数尤度関数
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\ell(\theta&space;\vert&space;x)&space;=&space;\log&space;p(x&space;\vert&space;\theta)" />
に対して，Score関数
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;g(\theta;&space;X)&space;=&space;\nabla_{\theta}&space;\ell(\theta;&space;X)" />
が定義される．
ここで，
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;g" />
の1次モーメントは
<imf src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\mathbb{E}_X[g(\theta&space;|&space;X)]&space;=&space;{\bf&space;0}" />
となることが知られており，
2次モーメント
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;{Var}_X[g(\theta&space;|&space;X)]&space;=&space;\mathbb{E}_{X}[{g(\theta&space;|&space;X)}^2]" />
をFisher情報量行列と呼び，
これは，
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\theta" />
の対数尤度関数
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\ell(\theta;&space;X)&space;=&space;\log&space;p(X&space;\vert&space;\theta)" />
に対するHessian行列の期待値として以下のように定義される．

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}\begin{align}&space;G(\theta)&space;&=&space;\mathbb{E}_{X}[&space;\nabla^2_{\theta}&space;\ell(\theta;&space;X)&space;]&space;\nonumber&space;\\&space;&=&space;\mathbb{E}_{X}&space;\left(&space;\begin{array}{cccc}&space;\frac{\partial^2&space;\ell(\theta;&space;X)}{\partial&space;\theta_1^2}&space;&&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_1&space;\partial&space;\theta_2}&space;&&space;\ldots&space;&&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_1&space;\partial&space;\theta_d}&space;\\&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_2&space;\partial&space;\theta_1}&space;&&space;\frac{\partial^2&space;\ell(\theta;&space;X)}{\partial&space;\theta_2^2}&space;&&space;\ldots&space;&&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_2&space;\partial&space;\theta_d}&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_d&space;\partial&space;\theta_1}&space;&&space;\frac{\partial&space;\ell(\theta;&space;X)}{\partial&space;\theta_d&space;\partial&space;\theta_2}&space;&&space;\ldots&space;&&space;\frac{\partial^2&space;\ell(\theta;&space;X)}{\partial&space;\theta_d^2}&space;\end{array}&space;\right)&space;\nonumber&space;\end{align}" />

### KL-divergence

確率分布間の擬距離であるKL-divergence

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;D_{KL}(p_{\theta_1},&space;p_{\theta_2})&space;=&space;\int&space;p(x&space;\vert&space;\theta_1)&space;\log&space;\frac{p(x&space;\vert&space;\theta_1)}{p(x&space;\vert&space;\theta_2)}&space;dx" />

について，2次までのMaclaurin展開をすると，

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;D_{KL}(p_{\theta},&space;p_{\theta&space;&plus;&space;\Delta&space;\theta})&space;\approx&space;\frac{1}{2}&space;{\Delta&space;\theta}^T&space;G(\theta)&space;\Delta&space;\theta" />

となる．
