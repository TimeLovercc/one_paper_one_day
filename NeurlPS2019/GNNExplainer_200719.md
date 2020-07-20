## GNNExplainer

#### 1. 本文结构

1. 基本信息
2. 读摘要
3. 读图、表分析思路和效果
4. 通读全文
7. 读后思考

#### 2. 基本信息

- 标题：GNNExplainer: Generating Explanations for Graph Neural Networks
- 作者：Rex Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, Jure Leskovec.
- 会议：NeurIPS 2019
- 代码链接：https://github.com/RexYing/gnn-model-explainer

#### 3. 摘要部分

图神经网络是在图上进行机器学习的强大的工具。GNNs通过在输入的图的边上递归地传递神经信息结合了节点属性信息和图的结构两方面内容。然而，合并图结构和属性信息模型复杂，GNNs的解释预测始终无法解决。这里，我们提出GNNExplainer，第一个对任何基于GNN的对任何基于图的机器学习任务的模型各异的通用解决方法。考虑一个实例，GNNExplainer确定一个紧凑的子图结构和一个小的在GNN的预测中有关键作用的节点属性的集合。然后，GNNExplainer能够对整个实例类生成一致的准确的解释。我们把GNNExplainer公式化成一个最大化GNN预测和可能子图结构分布的互信息优化任务。在综合的和真实图上的实验说明我们的方法能够识别重要的图结构和节点属性，并且表现得比其它baseline方法在解释准确度上好43%。GNNExplainer有很多好处，从可视化语义相关的结构到可解释性，再到深入了解GNNs的错误。

#### 4. 图、表分析

![image-20200719174521006](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200719174521006.png)

Figure 1：通过识别一个小的子图部分和一个小的节点属性集合，GNNExplainer完成预测。

![image-20200719174945773](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200719174945773.png)

Figure 2：GNNExplainer识别一个小的有影响力的属性集合和对预测关键的路径，排除无效信息。

![image-20200719175329187](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200719175329187.png)

Table 1：综合数据集下，GNNExplainer和其它baseline方法的比较。可以看到，效果提高了很多。

![image-20200719175616389](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200719175616389.png)

Figure 3: 单一实例解释性的评估。在四个综合数据集上对于节点分类任务的模范解释子图。发现找到了很好的对节点有影响的子图。

![image-20200719175916858](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200719175916858.png)

Figure 4：单实例的解释。展示了在两个数据集上的模范解释子图。

![image-20200719180058664](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200719180058664.png)

Figure 5：找出重要属性。GNNExplainer成功找出了重要的属性。

#### 5. 通读全文

##### 1. Introduction

GNNs在解决现实问题上很有价值。但是，不好让人理解，但理解起来又很有价值，既能提高对GNN的信任，又能提高在很多问题上的模型透明度，还能帮助从业者在部署之前更好理解网络特性。没有方法解释GNNs。解释神经网络有两条路线，一种解释简单化的模型，一种从属性上解释。这些方法都没有用到图上的联系信息。GNNExplainer先找小结构和属性，再预测。

##### 2. Related work

非图的神经网络可解释性方法分为两类。第一类使用简易模型代表整个神经网络，通常只是学得一个近似。第二类方法识别出计算时重要的层。但是，都不适合图神经网络。

图上的可解释很复杂，不容易做。最近通过注意力机制提高了可解释性。但是也有很多缺陷。

##### 3. Formulating explannations

将神经网络模型公式化，利用$G, E, V, X, f, C, \Phi$等表示一定含义。

三个重要公式概括一般的GNN：
$$
m_{ij}^l=Msg(h^{l-1}_i,h^{l-1}_j,r_{ij})
$$

$$
M_i^l=Agg({m_{ij}^l|v_j\in N_{v_i} })
$$


$$
h_i^l=Update\left( M_i^l,h_i^{l-1} \right)
$$
我们关键的insight在于节点的计算图：

- 计算图$G_c(v)$

- 关联邻接矩阵$A_c(v)\in \{0,1\}^{n\times n}$

- 关联属性集$X_c(v)=\{x_j|v_j\in G_c(v)\}$

- GNN学得一个条件分布$P_\Phi(Y|G_c,X_c)$

- GNN的预测$  \hat{y}=\phi \left( G_{c\left( v \right)},X_c\left( v \right) \right) $

- 为$ \hat{y}$生成的解释$\left( G_S,X^F_S \right) $

##### 4. GNNExplainer

优化框架：
$$
\mathop {\max} \limits_{G_S} MI\left( Y,\left( G_S,X_S \right) \right) =H( Y ) -H( Y|G=G_S,X=X_S )
$$
其中$H(Y)$是常数，而右式可以变换：
$$
H\left( Y|G=G_S,X=X_S \right) =-E_{Y|G_S,X_S}\left[ \log P_{\varPhi}\left( Y|G=G_S,X=X_S \right) \right] 
$$
然后变分近似，再近似，可以变成<font color=red>?</font>
$$
\underset{G}{\min}E_{G_S~G}H\left( Y|G=G_S,X=X_S \right) 
$$
由Jensen不等式可以得到上界：
$$
\underset{G}{\min}H\left( Y|G=E_G\left[ G_S \right] ,X=X_S \right) 
$$
用平均场近似，条件熵可以被取代<font color=red>?</font>,得到:
$$
\underset{M}{\min}-\sum_{c=1}^C{1\left[ y=c \right]}\log P_{\varPhi}\left( Y=y|G=A_c\odot \sigma \left( M \right) ,X=X_c \right) 
$$

###### 图结构和节点属性的联合学习

$$
\underset{G_S,F}{\max}MI\left( Y,\left( G_S,F \right) \right) =H\left( Y \right) -H\left( Y|G=G_S,X=X_S^F \right) 
$$

<font color=red>后面看不懂了，这里以后有时间就看，先跳过不耽误进度。</font>

##### 5. Experiments

四个人造数据集用于节点分类，两个真实世界数据集用于图分类。用了几个GNN的方法来作为baseline：GRAD是一个基于梯度的方法，类似于a saliency map approach. ATT是一个图注意GNN（GAT），它考虑图结构，但不解释节点属性。

实验结果可以直接看图，效果很好。

##### 6. Conclusion

文中方法提供了一个用GNN预测，调试GNN模型和识别系统错误模式的直接接口。

#### 6. 读后思考

这篇文章应该是图神经网络的开门之作，作者都是大佬。亮点在于将可解释量化了，用优化模型的优化过程寻找可解释。但是我感觉还是将模型当作黑箱，改变输出然后看箱子里面哪里变化了，找出变化的部分而已。

有时间继续看，深入理解。