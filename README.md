## Variational Graph Auto-Encoders
Paper : [VGAE-1.pdf](doc/VGAE-1.pdf)
### 1. Model Architecture
![img1.png](doc%2Fimg1.png)

The Model include Two Layers GCN:
1. The First Layer GCN use the Feature Matrix (Dimension: N * F) as Input, and Output the Hidden Matrix (Dimension: N * H).
2. The Second Layer GCN use the Hidden Matrix (Dimension: N * H) as Input, and Output the µ (N * L) and σ (N * L).

After the GCN Layers, the Model use the normal distribution to sample the Z (N * L) from the µ and σ.

$$ 
Z = \mu + \sigma \odot \theta
$$

In Decoder , We use the Matrix Z and $Z^T$ to compute the $A_hat$ (N * N) with the sigmoi function and the $A_hat$ is the predict Adjacent Matrix.
$$
A_hat = \sigma(Z * Z^T) 
$$

### 2. Loss Function
The Loss Function include two parts:
1. The Reconstruction Loss:  use the $A_hat$ and the Adjacent Matrix to compute the Reconstruction Loss.
2. The KL Divergence Loss:  use the µ and σ to compute the KL Divergence Loss.

$$
 Loss = E_{q(Z|X,A)}[logp(X|Z)] - KL(q(Z|X,A)||p(Z))
$$

### 3. Dataset
The Dataset is the Cora Dataset, CiteSeer Dataset and Pubmed Dataset. We used the Dataset from the library [Planetoid](Planetoid) in [PyG(torch_geometric)]().

### 4. Experiment
We use the Cora Dataset to train the Model, and use the CiteSeer Dataset and Pubmed Dataset to test the Model to recurrent the Paper's result.
Accoding the one times training, the Model can get the result as follow:

| Dataset | Cora | CiteSeer | Pubmed |
| :-----: |:----:|:--------:| :----: |
|  Paper  | 81.5 |   70.3   |  79.0  |

