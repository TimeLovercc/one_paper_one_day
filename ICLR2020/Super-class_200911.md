## Super-class

#### 1. 本文结构

1. 本文结构
2. 基本信息
3. 读摘要
4. 读图、表分析思路和效果
5. 通读全文
6. 代码分析
7. 读后思考

#### 2. 基本信息

- 标题：Few-shot learning on graphs via super-classes based on graph spectral measures
- 作者：Jatin Chauhan, Deepak Nathani, Manohar Kaul
- 会议：ICLR 2020
- 代码链接：https://github.com/RexYing/gnn-model-explainer

#### 3. 摘要部分

我们要利用GNN去识别few-shot图识别问题种没有见过的类别的问题，在图数目给定的情形下。尽管很多GNN的变种能解决图分类或者节点分类的任务，但是当遇到变迁数据稀少的情形下，这些GNN的表现都不是很好。这里我们提出一种新的方法，给每一个图依据它的图正则Laplacian分配一个概率测度。这让我们能够有依据地将图地基本标签聚类成超类，这样就能让$L^P$Wasserstein距离作为我们的内在距离量度。然后，一个基于这个超类的超图送进GNN框架处理去发现隐含的内在类别关系以达到更好地类别标签分类效果。我们做了很多测试并且展示出它比SOTA模型在few-shot场景下的效果好。并且，我们还将模型应用在了半监督和其它场景。

#### 4. 图、表分析

![image-20200911151752036](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200911151752036.png)

Figure 1: 这个图展示了GNN模型在训练和Fine-tuning阶段的不同。

![image-20200911152017304](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200911152017304.png)

Figure 2: Wasserstein超类聚类算法。

![image-20200911154705458](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200911154705458.png)

Table 1:  Letter-High 和 TRIANGLES数据集下不同few-shot时候结果。可以看到效果提高了很多个百分点。

![image-20200911155006289](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200911155006289.png)

Table 2: 在Reddit-12K和ENZYMES数据集上的变现。没有之前的好。可能这两个数据集有点大？

![image-20200911152131426](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200911152131426.png)

Figure 3: 可视化的embedding效果展示。可以看到embedding效果是最好的。

![image-20200911155142988](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200911155142988.png)

Table 3: 作者提出的模型，调节一个超参数，效果对比。可以看出有SC的情况下效果会好一些。

![image-20200911161808210](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200911161808210.png)

Table 4 & 5: 模型在20shot情形下对不同超类的分析。调节超参数k。

从上面看出，这似乎是将GNN与聚类结合了一些，构建超类。

#### 5. 通读全文

##### 1. Introduction

GNN需要更多轮聚集去影响大的邻居，但是实验说明层数增加之后效果没有显著变化。

作者模型首先用GIN，然后分别用MLP层和GAT处理，计算loss。

##### 2. Related work

Few-shot learning首先在2006年被李飞飞引进，很火，GNN也很火。比较了few-shot在图上和图像上的不同之处。

##### 3. Preliminaries

Base class和Novel class是不相交的。

Wasserstein Distance的定义。

##### 4. Our method

把所有图按标签分成几类，然后利用W距离计算最中间的图。

然后对prototype graph进行k-means聚类。

GIN进行特征提取。

然后一边GAT预测超类概率（associated class probabilities），一边走MLP学得超类标签。

fine-tune阶段就不一样了。都走，但是只用GAT进行预测。

##### 5. Experimental results

模型里面GAT换用GCN，整个模型换成GIN，结果提高了很多。

并且通过带不带super-class classifier的实验，说明带上之后提高了很多效果。

然后对超参数数据集和k进行了测试。

##### 6. Conclussion

感觉模型就是在后面加了个$C^{sup}$判断超类帮助进行分类。

#### 6. 代码分析

![image-20200912170443752](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200912170443752.png)

![image-20200912170321351](https://raw.githubusercontent.com/TimeLovercc/img/master/image-20200912170321351.png)

理解起来很容易，就是GIN之后用两种方式处理，然后loss加在一起算。

#### 7. 读后思考

这篇论文其实思路很简单，就是一个普通的GNN模型最后阶段换成了两种方法，训练方法用两种loss相加。fine-tune阶段两者不是。

注意里面巧妙地利用了聚类的思想。以后也可以思考在模型的某个环节叠加一个无监督或者非神经网络的方法，或许有点帮助。