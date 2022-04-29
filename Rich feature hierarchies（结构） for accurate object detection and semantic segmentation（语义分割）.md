# Rich feature hierarchies（结构） for accurate object detection and semantic segmentation（语义分割）

## Abstract

Object detection performance, as measured on the **canonical** PASCAL VOC dataset, has **plateaued** in the last few years. The best-performing methods are complex **ensemble systems** that **typically** combine multiple low-level image features with high-level context. In this paper, we propose a simple and **scalable** detection algorithm that improves mean average precision (mAP) by more than 30% relative to the previous best result on VOC 2012—achieving a mAP of 53.3%. Our approach **combines two key insights**: (1) one can **apply** **high-capacity** convolutional neural networks (CNNs) **to** **bottom-up region proposals** in order to **localize and segment objects** and (2) when labeled training data is **scarce**, supervised pre-training for an **auxiliary** task, followed by **domain-specific fine-tuning**, **yields a significant performance boost**. Since we combine **region proposals** with CNNs, we call our method R-CNN: **Regions with CNN features.** We also present experiments that provide insight into what the network learns, revealing a rich **hierarchy** of image features. Source code for the complete system is available at http://www.cs.berkeley.edu/rbg/rcnn. 

在过去的几年里，在典型的PASCAL VOC数据集上测量的物体检测性能已经趋于平稳。表现最好的方法是复杂的组合系统，通常将多个低层次的图像特征与高层次的背景相结合。在本文中，我们提出了一种简单的、可扩展的检测算法，相对于VOC 2012上的最佳结果，该算法的平均精度（mAP）提高了30%以上，达到了53.3%。我们的方法结合了两个关键的见解。**(1) 我们可以将大容量卷积神经网络（CNN）应用于自下而上的候选区域，以便对物体进行定位和分割；(2) 当标记的训练数据不足时，对辅助任务进行监督性的预训练，然后再进行特定领域的微调，可以产生显著的性能提升。由于我们将候选区域与CNN结合起来，我们将我们的方法称为R-CNN。**具有CNN特征的区域。我们还提出了一些实验，这些实验提供了对网络学习内容的洞察力，揭示了丰富的图像特征层次结构。整个系统的源代码可在http://www.cs.berkeley.edu/rbg/rcnn。

## 1. Introduction

 Features matter. The last decade of progress on various visual recognition tasks has been based considerably on the use of SIFT [26] and HOG [7]. But if we look at performance on the canonical visual recognition task, PASCAL VOC object detection [12], it is generally acknowledged that progress has been slow during 2010-2012, with small gains obtained by building ensemble systems and employing minor variants of successful methods. 

 特征很重要。在过去的十年中，各种视觉识别任务的进展主要是基于SIFT[26]和HOG[7]的使用。但是，如果我们看一下典型的视觉识别任务的表现，即PASCAL VOC物体检测[12]，人们普遍承认在2010-2012年期间进展缓慢，通过建立集合系统和采用成功方法的微小变体获得了小的收益。

SIFT and HOG are **blockwise orientation histograms**, a representation we could associate roughly with complex cells in V1, the first cortical area in the primate visual pathway. But we also know that recognition occurs several stages **downstream**, which suggests that there might be **hierarchical, multi-stage processes** for computing features that are even more informative for visual recognition. 

SIFT和HOG是顺时针方向的直方图，这种表示方法我们可以大致与V1的复杂细胞联系起来，V1是灵长类动物视觉通路的第一个皮质区域。但我们也知道，识别发生在下游的几个阶段，这表明可能有分层的、多阶段的过程来计算特征，这些特征对视觉识别来说甚至更有参考价值。

Fukushima’s “neocognitron” [16], a biologically inspired hierarchical and shift-invariant model for pattern recognition, was an early attempt at just such a process. The neocognitron, however, lacked a supervised training algorithm. LeCun et al. [23] provided the missing algorithm by showing that **stochastic gradient descent,** via **back propagation**, can train convolutional neural networks (CNNs), a class of models that extend the neocognitron. 

福岛的 "新认知器"[16]，一个受生物启发的分层和移位变量的模式识别模型，是对这样一个过程的早期尝试。然而，新认知器缺乏一个有监督的训练算法。LeCun等人[23]通过展示随机梯度下降，通过反向传播，可以训练卷积神经网络（CNN），一类扩展新认知模型的算法，提供了所缺的算法。

CNNs saw heavy use in the 1990s (e.g., [24]), but then **fell out of fashion**, particularly in computer vision, with the rise of support vector machines. In 2012, Krizhevsky et al. [22] rekindled interest in CNNs by showing substantially higher image classification accuracy on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) [9, 10]. Their success resulted from training a large CNN on 1.2 million labeled images, together with a few twists on LeCun’s CNN (e.g., max(x, 0) rectifying non-linearities and “dropout” regularization).

CNN在20世纪90年代被大量使用（例如[24]），但后来随着支持向量机的兴起，CNN逐渐淡出人们的视野，尤其是在计算机视觉领域。2012年，Krizhevsky等人[22]在ImageNet大规模视觉识别挑战赛（ILSVRC）[9, 10]上展示了大幅提高的图像分类精度，从而重新点燃了人们对CNN的兴趣。他们的成功来自于在120万张已标记的图像上训练一个大型的CNN，以及对LeCun的CNN的一些扭曲（例如，max(x, 0)纠正非线性和 "失活"正则化）。

 The significance of the ImageNet result was **vigorously** debated during the ILSVRC 2012 workshop. The central issue can be **distilled** to the following: To what extent do the CNN classification results on ImageNet generalize to object detection results on the PASCAL VOC Challenge? 

 ImageNet结果的意义在ILSVRC 2012研讨会上得到了激烈的讨论。核心问题可以提炼为以下几点。ImageNet上的CNN分类结果在多大程度上可以推广到PASCAL VOC挑战赛的物体检测结果？

We answer this question **decisively** by **bridging the chasm between image classification and object detection.** This paper is the first to show that a CNN can lead to **dramatically** higher object detection performance on PASCAL VOC as compared to systems based on simpler HOG-like features.1 Achieving this result required solving two problems: **localizing objects with a deep network and training a high-capacity model with only a small quantity of annotated detection data.**

我们通过弥合图像分类和物体检测之间的鸿沟来决定性地回答这个问题。本文首次表明，与基于更简单的类似HOG的特征的系统相比，CNN可以在PASCAL VOC上带来更高的物体检测性能。1 实现这一结果需要解决两个问题：用深度网络定位物体，以及仅用少量的注释检测数据训练一个高容量模型。

 Unlike image classification, detection requires localizing (likely many) objects within an image. **One approach frames localization as a regression problem.** However, work from Szegedy et al. [31], **concurrent with our own**, indicates that this strategy may not **fare well** in practice (they report a mAP of 30.5% on VOC 2007 compared to the 58.5% achieved by our method). An **alternative** is to build a **sliding-window detector**. CNNs have been used in this way for at least two decades, **typically on constrained object categories**, such as faces [28, 33] and pedestrians [29]. In order to **maintain high spatial resolution**, these CNNs typically only have two convolutional and pooling layers. We also considered adopting a sliding-window approach. However, **units high up in our network, which has five convolutional layers, have very large receptive fields (195 × 195 pixels) and strides (32×32 pixels) in the input image, which makes precise localization within the sliding-window paradigm an open technical challenge.**

 与图像分类不同，检测需要对图像中的（可能是许多）物体进行定位。有一种方法将定位视为一个回归问题。然而，与我们同时进行的Szegedy等人[31]的工作表明，这种策略在实践中可能并不理想（他们报告说，VOC 2007的mAP为30.5%，而我们的方法达到58.5%）。另一个选择是建立一个滑动窗口检测器。CNN已经以这种方式使用了至少20年，通常用于受限的物体类别，如人脸[28, 33]和行人[29]。为了保持高空间分辨率，这些CNN通常只有两个卷积层和池化层。我们也考虑过采用滑动窗口的方法。然而，在我们的网络中，有五个卷积层的高位单元在输入图像中具有非常大的感受野（195×195像素）和步幅（32×32像素），这使得滑动窗口范式中的精确定位成为一个公开的技术挑战。

 Instead, we solve the CNN localization problem by operating within the **“recognition using regions” paradigm**, as argued for by Gu et al. in [18]. At test-time, **our method generates around 2000 category-independent region proposals for the input image,** **extracts a fixed-length feature vector from each proposal using a CNN**, **and then classifies each region with category-specific linear SVMs.** **We use a simple technique (affine image warping) to compute a fixed-size CNN input from each region proposal,** **regardless of the region’s shape.** **Figure 1 presents an overview of our method and highlights some of our results. Since our system combines region proposals with CNNs, we dub the method R-CNN: Regions with CNN features.** 

![image-20220415112518936](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220415112518936.png)



 相反，我们通过在 "使用候选区域 "范式内操作来解决CNN的定位问题，正如Gu等人在[18]中所论证的。在测试时间，我们的方法为输入图像生成大约2000个与类别无关的候选区域，使用CNN从每个建议中提取一个固定长度的特征向量，然后用特定类别的线性SVM对每个区域进行分类。我们使用一种简单的技术（仿生图像扭曲）从每个区域的建议中计算出一个固定大小的CNN输入，而不管该区域的形状如何。图1展示了我们方法的概况，并强调了我们的一些结果。由于我们的系统将候选区域与CNN结合起来，我们将该方法称为R-CNN。具有CNN特征的区域。

A second challenge faced in detection is that **labeled data is scarce and the amount currently available is insufficient for training a large CNN.** The **conventional solution** to this problem is to use unsupervised pre-training, followed by supervised fine-tuning (e.g., [29]). **The second major contribution of this paper is to show that supervised pre-training on a large auxiliary dataset (ILSVRC), followed by domain-specific fine-tuning on a small dataset (PASCAL), is an effective paradigm for learning high-capacity CNNs when data is scarce.** In our experiments, fine-tuning for detection improves mAP performance by 8 percentage points. After fine-tuning, our system achieves a mAP of 54% on VOC 2010 compared to 33% for the **highly-tuned**, HOG-based **deformable** part model (DPM) [14, 17].

检测中面临的第二个挑战是，标记的数据很少，目前可用的数量不足以训练一个大型的CNN。这个问题的传统解决方案是使用无监督的预训练，然后再进行有监督的微调（例如，[29]）。本文的第二个主要贡献是表明，在大型辅助数据集（ILSVRC）上进行监督预训练，然后在小型数据集（PASCAL）上进行特定领域的微调，是在数据匮乏时学习大容量CNN的有效范式。在我们的实验中，检测的微调使mAP性能提高了8个百分点。在微调之后，我们的系统在2010年的VOC上实现了54%的mAP，而高度调谐的、基于HOG的可变形部件模型（DPM）[14, 17]则为33%。

Our system is also quite efficient. **The only class-specific computations are a reasonably small matrix-vector product and greedy non-maximum suppression. This computational property follows from features that are shared across all categories and that are also two orders of magnitude lower-dimensional than previously used region features** (cf. [32]). 

我们的系统也是相当高效的。唯一针对类别的计算是一个相当小的矩阵-向量乘积和贪婪的非最大抑制。这种计算特性来自于所有类别共享的特征，而且比以前使用的区域特征低两个数量级（参见[32]）。

One advantage of HOG-like features is their simplicity: it’s easier to understand the information they carry (although [34] shows that our intuition can fail us). Can we gain insight into the representation learned by the CNN? Perhaps the densely connected layers, with more than 54 million parameters, are the key? They are not. **We “lobotomized” the CNN and found that a surprisingly large proportion, 94%, of its parameters can be removed with only a moderate drop in detection accuracy. Instead, by probing units in the network we see that the convolutional layers learn a diverse set of rich features (Figure 3).** 

![image-20220415114034194](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220415114034194.png)

类似HOG的特征的一个优点是它们的简单性：更容易理解它们携带的信息（尽管[34]表明我们的直觉可能会让我们失望）。我们能深入了解CNN学到的表征吗？也许有超过5400万个参数的密集连接层是关键所在？它们不是。我们对CNN进行了 "脑叶切除术"，发现它的大部分参数（94%）可以被移除，而检测准确率仅有适度的下降。相反，通过探测网络中的单元，我们看到卷积层学习了一套多样化的丰富特征（图3）。

Understanding the failure modes of our approach is also critical for improving it, and so we report results from the detection analysis tool of Hoiem et al. [20]. As an immediate consequence of this analysis, **we demonstrate that a simple bounding box regression method significantly reduces mislocalizations, which are the dominant error mode.** 

了解我们方法的失败模式对于改进它也很关键，因此我们报告了Hoiem等人[20]的检测分析工具的结果。作为这一分析的直接结果，我们证明了一个简单的边界盒回归方法大大减少了错误定位，而这是最主要的错误模式。

Before developing technical details, **we note that because R-CNN operates on regions it is natural to extend it to the task of semantic segmentation.** With minor modifications, we also achieve state-of-the-art results on the PASCAL VOC segmentation task, with an average segmentation accuracy of 47.9% on the VOC 2011 test set.

在发展技术细节之前，我们注意到，由于R-CNN在区域上运作，因此很自然地将其扩展到语义分割的任务中。经过细微的修改，我们在PASCAL VOC分割任务上也取得了最先进的结果，在VOC 2011测试集上的平均分割精度为47.9%。

## 2. Object detection with R-CNN 

Our object detection system consists of three modules. The first generates category-independent region proposals. These proposals define the set of candidate detections available to our detector. The second module is a large convolutional neural network that extracts a fixed-length feature vector from each region. The third module is a set of classspecific linear SVMs. In this section, we present our design decisions for each module, describe their test-time usage, detail how their parameters are learned, and show results on PASCAL VOC 2010-12. 

我们的物体检测系统由三个模块组成。**第一个模块产生独立于类别的区域建议。这些建议定义了可供我们的检测器使用的候选检测的集合。第二个模块是一个大型卷积神经网络，从每个区域提取一个固定长度的特征向量。第三个模块是一组特定类别的线性SVMs。**在本节中，我们将介绍我们对每个模块的设计决定，描述它们在测试时的使用情况，详细说明它们的参数是如何学习的，并展示PASCAL VOC 2010-12的结果。

### 2.1. Module design 

#### Region proposals. 

A variety of recent papers offer methods for generating category-independent region proposals. **Examples include: objectness [1], selective search [32], category-independent object proposals [11], constrained parametric min-cuts (CPMC) [5], multi-scale combinatorial grouping [3], and Cires¸an et al. [6], who detect mitotic cells by applying a CNN to regularly-spaced square crops, which are a special case of region proposals. While R-CNN is agnostic to the particular region proposal method, we use *selective search* to enable a controlled comparison with prior Feature extraction.** We extract a 4096-dimensional feature vector from each region proposal using the Caffe [21] implementation of the CNN described by Krizhevsky et al. [22]. **Features are computed by forward propagating a mean-subtracted 227 × 227 RGB image through five convolutional layers and two fully connected layers.** We refer readers to [21, 22] for more network architecture details.

最近有很多论文提供了生成与类别无关的区域建议的方法。**例子包括：对象性[1]、选择性搜索[32]、与类别无关的对象建议[11]、受限参数最小化（CPMC）[5]、多尺度组合分组[3]，以及Cires¸an等人[6]，他们通过将CNN应用于规则间隔的方形作物来检测有丝分裂细胞，这是区域建议的一个特殊情况。虽然R-CNN与特定的区域提议方法无关，但我们使用*选择性搜索*，以便与之前的特征提取进行对照。**我们使用Krizhevsky等人[22]描述的CNN的Caffe[21]实现，从每个区域提议中提取一个4096维的特征向量。**特征的计算是通过前向传播一个平均减去227×227的RGB图像，通过五个卷积层和两个完全连接层来进行的。**我们请读者参考[21，22]以了解更多的网络结构细节。

 In order to compute features for a region proposal, we must first **convert the image data in that region into a form that is compatible with the CNN (its architecture requires inputs of a fixed 227 × 227 pixel size)**. Of the many possible transformations of our arbitrary-shaped regions, we opt for the simplest. Regardless of the size or aspect ratio of the candidate region, we warp all pixels in a tight bounding box around it to the required size. Prior to warping, we dilate the tight bounding box so that at the warped size there are exactly p pixels of warped image context around the original box (we use p = 16). Figure 2 shows a random sampling of warped training regions. The supplementary material discusses alternatives to warping.

![image-20220418171644220](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220418171644220.png)

 为了计算一个候选区域的特征，我们必须首先**将该区域的图像数据转换为与CNN兼容的形式（其架构要求输入固定为227×227像素大小）**。在我们的任意形状区域的许多可能的转换中，我们选择了最简单的。无论候选区域的大小或长宽比如何，我们都将其周围一个紧密的边界框内的所有像素扭曲成所需的大小。在翘曲之前，我们扩张紧缩边界框，以便在翘曲的尺寸下，原始框周围正好有p像素的翘曲图像背景（我们使用p=16）。图2显示了一个随机抽样的翘曲的训练区域。补充材料中讨论了翘曲的替代方法。

### 2.2. Test-time detection

 At test time, we run selective search on the test image to extract around 2000 region proposals (we use selective search’s “fast mode” in all experiments). We warp each proposal and forward propagate it through the CNN in order to read off features from the desired layer. Then, for each class, we score each extracted feature vector using the SVM trained for that class. Given all scored regions in an image, we apply a greedy non-maximum suppression (for each class independently) that rejects a region if it has an intersection-over-union (IoU) overlap with a higher scoring selected region larger than a learned threshold. 

 在测试时间，我们在测试图像上运行选择性搜索，以提取大约2000个区域建议（我们在所有实验中使用选择性搜索的 "快速模式"）。我们对每个提议进行扭曲，并通过CNN进行前向传播，以便从所需层读出特征。然后，对于每个类别，我们使用为该类别训练的SVM对每个提取的特征向量进行评分。考虑到图像中的所有得分区域，我们应用一个贪婪的非最大限度的抑制（对每个类别独立），如果一个区域与一个得分较高的选定区域的重叠部分大于学习阈值，则拒绝该区域。

#### Run-time analysis. 

Two properties make detection efficient. First, all CNN parameters are shared across all categories. Second, the feature vectors computed by the CNN are low-dimensional when compared to other common approaches, such as spatial pyramids with bag-of-visual-word encodings. The features used in the UVA detection system [32], for example, are two orders of magnitude larger than ours (360k vs. 4k-dimensional). The result of such sharing is that the time spent computing region proposals and features (13s/image on a GPU or 53s/image on a CPU) is amortized over all classes. The only class-specific computations are dot products between features and SVM weights and non-maximum suppression. In practice, all dot products for an image are batched into a single matrix-matrix product. The feature matrix is typically 2000×4096 and the SVM weight matrix is 4096×N, where N is the number of classes.

有两个特性使检测变得高效。首先，所有的CNN参数在所有类别中都是共享的。第二，与其他常见的方法相比，CNN计算的特征向量是低维的，例如带有视觉词包编码的空间金字塔。例如，UVA检测系统[32]中使用的特征比我们的大两个数量级（360k vs. 4k-维）。这种共享的结果是，计算区域建议和特征的时间（GPU上的13s/图像或CPU上的53s/图像）被分摊到所有类别中。唯一针对类的计算是特征和SVM权重之间的点乘和非最大抑制。在实践中，一个图像的所有点积都被打包成一个单一的矩阵-矩阵乘积。特征矩阵通常为2000×4096，SVM权重矩阵为4096×N，其中N为类的数量。

 This analysis shows that R-CNN can scale to thousands of object classes without resorting to approximate techniques, such as hashing. Even if there were 100k classes, the resulting matrix multiplication takes only 10 seconds on a modern multi-core CPU. This efficiency is not merely the result of using region proposals and shared features. The UVA system, due to its high-dimensional features, would be two orders of magnitude slower while requiring 134GB of memory just to store 100k linear predictors, compared to just 1.5GB for our lower-dimensional features. 

这一分析表明，R-CNN可以扩展到成千上万的对象类，而不需要借助近似的技术，如散列。即使有10万个类，所产生的矩阵乘法在现代多核CPU上只需要10秒。这种效率不仅仅是使用区域建议和共享特征的结果。UVA系统，由于其高维特征，会慢两个数量级，同时需要134GB的内存来存储100k的线性预测器，而我们的低维特征只需要1.5GB。

It is also interesting to contrast R-CNN with the recent work from Dean et al. on scalable detection using DPMs and hashing [8]. They report a mAP of around 16% on VOC 2007 at a run-time of 5 minutes per image when introducing 10k distractor classes. With our approach, 10k detectors can run in about a minute on a CPU, and because no approximations are made mAP would remain at 59% (Section 3.2). 

将R-CNN与Dean等人最近关于使用DPM和散列的可扩展检测工作进行对比也很有意思[8]。他们报告说，在引入10k个干扰物类别时，VOC 2007的mAP约为16%，每幅图像的运行时间为5分钟。使用我们的方法，10k个检测器可以在CPU上运行大约1分钟，而且由于没有进行近似处理，mAP将保持在59%（第3.2节）。

### 2.3. Training 

#### Supervised pre-training. 

We discriminatively pre-trained the CNN on a large auxiliary dataset (ILSVRC 2012) with image-level annotations (i.e., no bounding box labels). Pretraining was performed using the open source Caffe CNN library [21]. In brief, our CNN nearly matches the performance of Krizhevsky et al. [22], obtaining a top-1 error rate 2.2 percentage points higher on the ILSVRC 2012 validation set. This discrepancy is due to simplifications in the training process. 

我们在一个大型辅助数据集（ILSVRC 2012）上对CNN进行了辨别性的预训练，该数据集具有图像级别的注释（即没有边界框标签）。预训练是使用开源的Caffe CNN库[21]进行的。简而言之，我们的CNN几乎与Krizhevsky等人[22]的性能相匹配，在ILSVRC 2012的验证集上获得的最高1级错误率高出2.2个百分点。这一差异是由于训练过程中的简化造成的。

#### Domain-specific fine-tuning. 

To adapt our CNN to the new task (detection) and the new domain (warped VOC windows), we continue stochastic gradient descent (SGD) training of the CNN parameters using only warped region proposals from VOC. Aside from replacing the CNN’s ImageNet-specific 1000-way classification layer with a randomly initialized 21-way classification layer (for the 20 VOC classes plus background), the CNN architecture is unchanged. We treat all region proposals with ≥ 0.5 IoU overlap with a ground-truth box as positives for that box’s class and the rest as negatives. We start SGD at a learning rate of 0.001 (1/10th of the initial pre-training rate), which allows fine-tuning to make progress while not clobbering the initialization. In each SGD iteration, we uniformly sample 32 positive windows (over all classes) and 96 background windows to construct a mini-batch of size 128. We bias the sampling towards positive windows because they are extremely rare compared to background 

为了使我们的CNN适应新的任务（检测）和新的领域（扭曲的VOC窗口），我们继续对CNN参数进行随机梯度下降（SGD）训练，只使用VOC的扭曲区域建议。除了用一个随机初始化的21路分类层（针对20个VOC类别和背景）取代CNN的ImageNet特定的1000路分类层外，CNN的架构没有改变。我们将所有与地面真相框重叠≥0.5 IoU的区域建议作为该框类别的阳性，其余的作为阴性。我们以0.001的学习率（初始预训练率的1/10）开始SGD，这允许微调取得进展，同时不会使初始化崩溃。在SGD的每次迭代中，我们均匀地对32个正向窗口（所有类别）和96个背景窗口进行采样，以构建一个128大小的小型批次。我们偏向于对正向窗口进行抽样，因为与背景窗口相比，正向窗口极为罕见。

**Object category classifiers.** 

Consider training a binary classifier to detect cars. It’s clear that an image region tightly enclosing a car should be a positive example. Similarly, it’s clear that a background region, which has nothing to do with cars, should be a negative example. Less clear is how to label a region that partially overlaps a car. We resolve this issue with an IoU overlap threshold, below which regions are defined as negatives. The overlap threshold, 0.3, was selected by a grid search over {0, 0.1, . . . , 0.5} on a validation set. We found that selecting this threshold carefully is important. Setting it to 0.5, as in [32], decreased mAP by 5 points. Similarly, setting it to 0 decreased mAP by 4 points. Positive examples are defined simply to be the ground-truth bounding boxes for each class. 

考虑训练一个二元分类器来检测汽车。很明显，一个紧密包围着汽车的图像区域应该是一个正面的例子。同样，很明显，一个与汽车无关的背景区域应该是一个负面的例子。不太清楚的是如何标记一个与汽车部分重叠的区域。我们用一个IoU重叠阈值来解决这个问题，低于这个阈值的区域被定义为负面的。重叠阈值，0.3，是通过在{0, 0.1, . . . , 0.5}的验证集上进行网格搜索。我们发现，谨慎地选择这个阈值是很重要的。如同在[32]中，将其设置为0.5，mAP减少了5个点。同样地，将其设置为0会使mAP减少4分。正面的例子被简单地定义为每个类别的地面真实边界框。

Once features are extracted and training labels are applied, we optimize one linear SVM per class. Since the training data is too large to fit in memory, we adopt the standard hard negative mining method [14, 30]. Hard negative mining converges quickly and in practice mAP stops increasing after only a single pass over all images.

 In supplementary material we discuss why the positive and negative examples are defined differently in fine-tuning versus SVM training. We also discuss why it’s necessary to train detection classifiers rather than simply use outputs from the final layer (fc8) of the fine-tuned CNN. 

一旦提取了特征并应用了训练标签，我们就对每个类别的一个线性SVM进行优化。由于训练数据太大，无法装入内存，我们采用了标准的硬阴性挖掘方法[14, 30]。硬阴性挖掘法收敛很快，在实践中，mAP仅在对所有图像进行一次处理后就不再增加。

 在补充材料中，我们讨论了为什么在微调与SVM训练中，正负例子的定义是不同的。我们还讨论了为什么有必要训练检测分类器而不是简单地使用微调CNN最后一层（fc8）的输出。

### 2.4. Results on PASCAL VOC 2010-12 

Following the PASCAL VOC best practices [12], we validated all design decisions and hyperparameters on the VOC 2007 dataset (Section 3.2). For final results on the VOC 2010-12 datasets, we fine-tuned the CNN on VOC 2012 train and optimized our detection SVMs on VOC 2012 trainval. We submitted test results to the evaluation server only once for each of the two major algorithm variants (with and without bounding box regression). 

Table 1 shows complete results on VOC 2010. We compare our method against four strong baselines, including SegDPM [15], which combines DPM detectors with the output of a semantic segmentation system [4] and uses additional inter-detector context and image-classifier rescoring. The most germane comparison is to the UVA system from Uijlings et al. [32], since our systems use the same region proposal algorithm. To classify regions, their method builds a four-level spatial pyramid and populates it with densely sampled SIFT, Extended OpponentSIFT, and RGB-SIFT descriptors, each vector quantized with 4000-word codebooks. Classification is performed with a histogram intersection kernel SVM. Compared to their multi-feature, non-linear kernel SVM approach, we achieve a large improvement in mAP, from 35.1% to 53.7% mAP, while also being much faster (Section 2.2). Our method achieves similar performance (53.3% mAP) on VOC 2011/12 test. 

按照PASCAL VOC的最佳实践[12]，我们在VOC 2007数据集上验证了所有的设计决策和超参数（3.2节）。对于VOC 2010-12数据集的最终结果，我们在VOC 2012 train上对CNN进行了微调，并在VOC 2012 trainval上优化了我们的检测SVMs。对于两种主要的算法变体（有边界盒回归和无边界盒回归），我们只向评估服务器提交了一次测试结果。

表1显示了VOC 2010上的完整结果。我们将我们的方法与四个强大的基线进行了比较，包括SegDPM[15]，它将DPM检测器与语义分割系统[4]的输出相结合，并使用额外的检测器之间的背景和图像分类器的重新评分。最有意义的比较是与Uijlings等人[32]的UVA系统，因为我们的系统使用相同的区域提议算法。为了对区域进行分类，他们的方法建立了一个四级空间金字塔，并用密集采样的SIFT、Extended OpponentSIFT和RGB-SIFT描述符填充它，每个向量用4000字的编码簿量化。分类是用直方图相交核SVM进行的。与他们的多特征、非线性核SVM方法相比，我们在mAP方面取得了很大的改进，从35.1%到53.7%的mAP，同时速度也更快（第2.2节）。我们的方法在VOC 2011/12测试中取得了类似的性能（53.3% mAP）。

## 3. Visualization, ablation, and modes of error 

可视化、消融和错误模式 

### 3.1. Visualizing learned features 

First-layer filters can be visualized directly and are easy to understand [22]. They capture oriented edges and opponent colors. Understanding the subsequent layers is more challenging. Zeiler and Fergus present a visually attractive deconvolutional approach in [36]. We propose a simple (and complementary) non-parametric method that directly shows what the network learned.

第一层过滤器可以直接可视化，而且很容易理解[22]。它们能捕捉到定向的边缘和对手的颜色。理解后续层则更具挑战性。Zeiler和Fergus在[36]中提出了一种视觉上有吸引力的去卷积方法。我们提出了一个简单的（和互补的）非参数方法，直接显示网络学到了什么。

 The idea is to single out a particular unit (feature) in the network and use it as if it were an object detector in its own right. That is, we compute the unit’s activations on a large set of held-out region proposals (about 10 million), sort the proposals from highest to lowest activation, perform non-maximum suppression, and then display the top-scoring regions. Our method lets the selected unit “speak for itself” by showing exactly which inputs it fires on. We avoid averaging in order to see different visual modes and gain insight into the invariances computed by the unit.

 我们的想法是在网络中挑出一个特定的单元（特征），并将其作为一个物体检测器来使用，就像它本身一样。也就是说，我们计算出该单元在一大组被搁置的区域建议上的激活度（大约1000万），将这些建议从最高激活度到最低激活度进行排序，执行非最大抑制，然后显示出得分最高的区域。我们的方法让所选单元 "为自己说话"，准确地显示它在哪些输入上开火。我们避免平均化，以便看到不同的视觉模式，并深入了解该单元计算的不变性。

 We visualize units from layer pool5 , which is the maxpooled output of the network’s fifth and final convolutional layer. The pool5 feature map is 6 × 6 × 256 = 9216- dimensional. Ignoring boundary effects, each pool5 unit has a receptive field of 195×195 pixels in the original 227×227 pixel input. A central pool5 unit has a nearly global view, while one near the edge has a smaller, clipped support. 

 我们将池5层的单元可视化，它是网络的第五层也是最后一层卷积层的主要输出。pool5的特征图是6×6×256=9216维的。忽略边界效应，每个pool5单元在原始227×227像素的输入中具有195×195像素的感受野。一个中央的pool5单元有一个几乎全局的视野，而靠近边缘的单元有一个较小的、被剪切的支持。

Each row in Figure 3 displays the top 16 activations for a pool5 unit from a CNN that we fine-tuned on VOC 2007 trainval. Six of the 256 functionally unique units are visualized (the supplementary material includes more). These units were selected to show a representative sample of what the network learns. In the second row, we see a unit that fires on dog faces and dot arrays. The unit corresponding to the third row is a red blob detector. There are also detectors for human faces and more abstract patterns such as text and triangular structures with windows. The network appears to learn a representation that combines a small number of class-tuned features together with a distributed representation of shape, texture, color, and material properties. The subsequent fully connected layer fc6 has the ability to model a large set of compositions of these rich features. 

图3中的每一行显示了我们在VOC 2007 trainval上微调的一个CNN的pool5单元的前16个激活。在256个功能独特的单元中，有6个是可视化的（补充材料中包括更多）。选择这些单元是为了显示网络学习的代表性样本。在第二行，我们看到一个单元在狗脸和点阵上发射。第三行对应的单元是一个红色的圆球检测器。还有一些检测器用于检测人脸和更抽象的图案，如文字和带窗口的三角形结构。该网络似乎在学习一种表征，将少量的类调整特征与形状、纹理、颜色和材料属性的分布式表征结合在一起。随后的全连接层fc6有能力对这些丰富特征的大量组合进行建模。

### 3.2. Ablation studies 

#### Performance layer-by-layer, without fine-tuning. 

To understand which layers are critical for detection performance, we analyzed results on the VOC 2007 dataset for each of the CNN’s last three layers. Layer pool5 was briefly described in Section 3.1. The final two layers are summarized below. Layer fc6 is fully connected to pool5 . To compute features, it multiplies a 4096×9216 weight matrix by the pool5 feature map (reshaped as a 9216-dimensional vector) and then adds a vector of biases. This intermediate vector is component-wise half-wave rectified (x ← max(0, x)).

 Layer fc7 is the final layer of the network. It is implemented by multiplying the features computed by fc6 by a 4096 × 4096 weight matrix, and similarly adding a vector of biases and applying half-wave rectification. 

为了了解哪些层对检测性能至关重要，我们分析了CNN最后三层中每一层在VOC 2007数据集上的结果。第3.1节中简要介绍了层池5。下面总结了最后两层的情况。fc6层与pool5完全连接。为了计算特征，它将4096×9216的权重矩阵与pool5的特征图（重塑为9216维的向量）相乘，然后加入一个偏置向量。这个中间向量经过分量的半波整流（x ← max(0, x)）。

 fc7层是网络的最后一层。它是通过将fc6计算的特征乘以4096×4096的权重矩阵来实现的，同样地，加入一个偏置矢量并应用半波整流。

We start by looking at results from the CNN without fine-tuning on PASCAL, i.e. all CNN parameters were pretrained on ILSVRC 2012 only. Analyzing performance layer-by-layer (Table 2 rows 1-3) reveals that features from fc7 generalize worse than features from fc6. This means that 29%, or about 16.8 million, of the CNN’s parameters can be removed without degrading mAP. More surprising is that removing both fc7 and fc6 produces quite good results even though pool5 features are computed using only 6% of the CNN’s parameters. Much of the CNN’s representational power comes from its convolutional layers, rather than from the much larger densely connected layers. This finding suggests potential utility in computing a dense feature map, in the sense of HOG, of an arbitrary-sized image by using only the convolutional layers of the CNN. This representation would enable experimentation with sliding-window detectors, including DPM, on top of pool5 features. 

我们首先看一下没有在PASCAL上进行微调的CNN的结果，即所有CNN参数只在ILSVRC 2012上进行了预训练。逐层分析性能（表2第1-3行），发现fc7的特征比fc6的特征泛化得差。这意味着29%，即大约1680万个CNN的参数可以被移除而不降低mAP。更令人惊讶的是，尽管pool5的特征只用了CNN的6%的参数来计算，但去除fc7和fc6都会产生相当好的结果。CNN的大部分表征能力来自其卷积层，而不是来自更大的密集连接层。这一发现表明，只用CNN的卷积层就能计算出任意大小图像的密集特征图，即HOG的意义。这种表示方法将使人们能够在池5特征的基础上进行滑动窗口检测器的实验，包括DPM。

#### Performance layer-by-layer, with fine-tuning. 

We now look at results from our CNN after having fine-tuned its parameters on VOC 2007 trainval. The improvement is striking (Table 2 rows 4-6): fine-tuning increases mAP by 8.0 percentage points to 54.2%. The boost from fine-tuning is much larger for fc6 and fc7 than for pool5 , which suggests that the pool5 features learned from ImageNet are general and that most of the improvement is gained from learning domain-specific non-linear classifiers on top of them. 

我们现在看看我们的CNN在VOC 2007 trainval上微调了参数后的结果。改进是惊人的（表2第4-6行）：微调使mAP增加了8.0个百分点，达到54.2%。微调对fc6和fc7的提升远远大于pool5，这表明从ImageNet学到的pool5特征是通用的，大部分的改进是在它们之上学习特定领域的非线性分类器而获得的。

#### Comparison to recent feature learning methods.

 Relatively few feature learning methods have been tried on PASCAL VOC detection. We look at two recent approaches that build on deformable part models. For reference, we also include results for the standard HOG-based DPM [17]. The first DPM feature learning method, DPM ST [25], augments HOG features with histograms of “sketch token” probabilities. Intuitively, a sketch token is a tight distribution of contours passing through the center of an image patch. Sketch token probabilities are computed at each pixel by a random forest that was trained to classify 35×35 pixel patches into one of 150 sketch tokens or background. The second method, DPM HSC [27], replaces HOG with histograms of sparse codes (HSC). To compute an HSC, sparse code activations are solved for at each pixel using a learned dictionary of 100 7 × 7 pixel (grayscale) atoms. The resulting activations are rectified in three ways (full and both half-waves), spatially pooled, unit `2 normalized, and then power transformed (x ← sign(x)|x| α). 

相对来说，在PASCAL VOC检测上尝试的特征学习方法很少。我们看一下最近两种建立在可变形部件模型上的方法。作为参考，我们也包括基于HOG的标准DPM[17]的结果。第一种DPM特征学习方法，DPM ST[25]，用 "草图标记 "概率直方图来增强HOG特征。直观地说，草图标记是通过图像斑块中心的轮廓线的紧密分布。草图标记概率是由一个随机森林在每个像素上计算出来的，该随机森林被训练成将35×35像素的斑块分类为150个草图标记或背景之一。第二种方法，DPM HSC[27]，用稀疏代码直方图（HSC）取代了HOG。为了计算HSC，使用100个7×7像素（灰度）原子的学习字典来解决每个像素的稀疏代码激活问题。得到的激活以三种方式（全波和半波）进行整顿，空间汇集，单位`2归一化，然后进行功率变换（x ← sign(x)|x| α）。

All R-CNN variants strongly outperform the three DPM baselines (Table 2 rows 8-10), including the two that use feature learning. Compared to the latest version of DPM, which uses only HOG features, our mAP is more than 20 percentage points higher: 54.2% vs. 33.7%—a 61% relative improvement. The combination of HOG and sketch tokens yields 2.5 mAP points over HOG alone, while HSC improves over HOG by 4 mAP points (when compared internally to their private DPM baselines—both use nonpublic implementations of DPM that underperform the open source version [17]). These methods achieve mAPs of 29.1% and 34.3%, respectively. 

所有的R-CNN变体都强烈地超越了三个DPM基线（表2第8-10行），包括两个使用特征学习的变体。与只使用HOG特征的DPM的最新版本相比，我们的mAP高出20多个百分点：54.2%对33.7%--61%的相对改进。HOG和草图标记的组合比单独的HOG产生了2.5个mAP点，而HSC比HOG提高了4个mAP点（当与他们的私有DPM基线进行内部比较时--两者都使用了DPM的非公开实现，性能低于开源版本[17]）。这些方法的mAPs分别为29.1%和34.3%。

### 3.3. Detection error analysis 

We applied the excellent detection analysis tool from Hoiem et al. [20] in order to reveal our method’s error modes, understand how fine-tuning changes them, and to see how our error types compare with DPM. A full summary of the analysis tool is beyond the scope of this paper and we encourage readers to consult [20] to understand some finer details (such as “normalized AP”). Since the analysis is best absorbed in the context of the associated plots, we present the discussion within the captions of Figure 4 and Figure 5

我们应用了Hoiem等人[20]的优秀检测分析工具，以揭示我们方法的错误模式，了解微调如何改变它们，并查看我们的错误类型与DPM的比较。对分析工具的全面总结超出了本文的范围，我们鼓励读者查阅[20]以了解一些更精细的细节（如 "规范化AP"）。由于分析最好是在相关图表的背景下进行，我们在图4和图5的标题中提出讨论

###  3.4. Bounding box regression 

Based on the error analysis, we implemented a simple method to reduce localization errors. Inspired by the bounding box regression employed in DPM [14], we train a linear regression model to predict a new detection window given the pool5 features for a selective search region proposal. Full details are given in the supplementary material. Results in Table 1, Table 2, and Figure 4 show that this simple approach fixes a large number of mislocalized detections, boosting mAP by 3 to 4 points.

基于误差分析，我们实施了一个简单的方法来减少定位误差。受DPM[14]中采用的边界盒回归的启发，我们训练了一个线性回归模型，以预测一个新的检测窗口，给定池5的特征，以提出一个选择性的搜索区域。完整的细节在补充材料中给出。表1、表2和图4中的结果显示，这种简单的方法修复了大量的错误定位的检测，使mAP提高了3到4个点。

## 4. Semantic segmentation 

Region classification is a standard technique for semantic segmentation, allowing us to easily apply R-CNN to the PASCAL VOC segmentation challenge. To facilitate a direct comparison with the current leading semantic segmentation system (called O2P for “second-order pooling”) [4], we work within their open source framework. O2P uses CPMC to generate 150 region proposals per image and then predicts the quality of each region, for each class, using support vector regression (SVR). The high performance of their approach is due to the quality of the CPMC regions and the powerful second-order pooling of multiple feature types (enriched variants of SIFT and LBP). We also note that Farabet et al. [13] recently demonstrated good results on several dense scene labeling datasets (not including PASCAL) using a CNN as a multi-scale per-pixel classifier. 

We follow [2, 4] and extend the PASCAL segmentation training set to include the extra annotations made available by Hariharan et al. [19]. Design decisions and hyperparameters were cross-validated on the VOC 2011 validation set. Final test results were evaluated only once. 

区域分类是语义分割的标准技术，使我们能够轻松地将R-CNN应用于PASCAL VOC分割的挑战。为了便于与目前领先的语义分割系统（被称为O2P的 "二阶集合"）[4]进行直接比较，我们在其开源框架内工作。O2P使用CPMC为每幅图像生成150个区域建议，然后使用支持向量回归（SVR）预测每个区域的质量，适用于每个类别。他们的方法的高性能是由于CPMC区域的质量和多种特征类型（SIFT和LBP的丰富变体）的强大二阶集合。我们还注意到，Farabet等人[13]最近在几个密集场景标签数据集（不包括PASCAL）上使用CNN作为多尺度每像素分类器证明了良好的结果。

我们遵循[2, 4]并扩展了PASCAL分割训练集，以包括由Hariharan等人[19]提供的额外注释。设计决策和超参数在VOC 2011验证集上进行了交叉验证。最终的测试结果只被评估了一次。

**CNN features for segmentation.** 

We evaluate three strategies for computing features on CPMC regions, all of which begin by warping the rectangular window around the region to 227 × 227. The first strategy (full) ignores the region’s shape and computes CNN features directly on the warped window, exactly as we did for detection. However, these features ignore the non-rectangular shape of the region. Two regions might have very similar bounding boxes while having very little overlap. Therefore, the second strategy (fg) computes CNN features only on a region’s foreground mask. We replace the background with the mean input so that background regions are zero after mean subtraction. The third strategy (full+fg) simply concatenates the full and fg features; our experiments validate their complementarity. 

我们评估了三种计算CPMC区域特征的策略，所有这些策略都是从将区域周围的矩形窗口扭曲为227×227开始的。第一种策略（完全）忽略了区域的形状，直接在扭曲的窗口上计算CNN特征，与我们在检测时的做法完全一样。然而，这些特征忽略了区域的非矩形形状。两个区域可能有非常相似的边界框，同时又有非常少的重叠。因此，第二种策略（fg）只在一个区域的前景遮罩上计算CNN特征。我们用均值输入替换背景，这样背景区域在均值减去后为零。第三种策略（full+fg）简单地将full和fg的特征连接起来；我们的实验验证了它们的互补性。

**Results on VOC 2011.** 

Table 3 shows a summary of our results on the VOC 2011 validation set compared with O2P. (See supplementary material for complete per-category results.) Within each feature computation strategy, layer fc6 always outperforms fc7 and the following discussion refers to the fc6 features. The fg strategy slightly outperforms full, indicating that the masked region shape provides a stronger signal, matching our intuition. However, full+fg achieves an average accuracy of 47.9%, our best result by a margin of 4.2% (also modestly outperforming O2P), indicating that the context provided by the full features is highly informative even given the fg features. Notably, training the 20 SVRs on our full+fg features takes an hour on a single core, compared to 10+ hours for training on O2P features. 

In Table 4 we present results on the VOC 2011 test set, comparing our best-performing method, fc6 (full+fg), against two strong baselines. Our method achieves the highest segmentation accuracy for 11 out of 21 categories, and the highest overall segmentation accuracy of 47.9%, averaged across categories (but likely ties with the O2P result under any reasonable margin of error). Still better performance could likely be achieved by fine-tuning. 

表3显示了我们在VOC 2011验证集上与O2P比较的结果摘要。(在每个特征计算策略中，fc6层总是优于fc7层，下面的讨论指的是fc6特征。fg策略略微优于full，表明被掩盖的区域形状提供了更强的信号，与我们的直觉相吻合。然而，full+fg达到了47.9%的平均准确率，以4.2%的幅度达到了我们的最佳结果（也略微超过了O2P），表明即使考虑到fg特征，full特征所提供的背景也是非常有价值的。值得注意的是，在我们的完整特征+fg特征上训练20个SVR只需要一个小时，而在O2P特征上训练则需要10多个小时。

在表4中，我们展示了VOC 2011测试集的结果，将我们表现最好的方法fc6（full+fg）与两个强大的基线进行比较。我们的方法在21个类别中的11个取得了最高的分割准确率，总体分割准确率最高，为47.9%，跨类别的平均数（但在任何合理的误差范围内都可能与O2P的结果持平）。通过微调，可能还能取得更好的性能。

## 5. Conclusion 

In recent years, object detection performance had stagnated. The best performing systems were complex ensembles combining multiple low-level image features with high-level context from object detectors and scene classifiers. This paper presents a simple and scalable object detection algorithm that gives a 30% relative improvement over the best previous results on PASCAL VOC 2012. 

近年来，物体检测性能停滞不前。表现最好的系统是将多个低层次图像特征与来自物体检测器和场景分类器的高层次背景相结合的复杂组合。本文提出了一种简单的、可扩展的物体检测算法，比以前在PASCAL VOC 2012上的最佳结果有30%的相对改进。

We achieved this performance through two insights. The first is to apply high-capacity convolutional neural networks to bottom-up region proposals in order to localize and segment objects. The second is a paradigm for train- ing large CNNs when labeled training data is scarce. We show that it is highly effective to pre-train the network— with supervision—for a auxiliary task with abundant data (image classification) and then to fine-tune the network for the target task where data is scarce (detection). We conjecture that the “supervised pre-training/domain-specific finetuning” paradigm will be highly effective for a variety of data-scarce vision problems. 

我们通过两种见解实现了这种性能。首先是将高容量的卷积神经网络应用于自下而上的区域建议，以便对物体进行定位和分割。第二是在标记的训练数据稀少时训练大型CNN的范式。我们表明，对网络进行预训练是非常有效的，在监督下对具有丰富数据的辅助任务（图像分类）进行训练，然后对数据稀缺的目标任务（检测）进行微调。我们猜想，"有监督的预训练/特定领域的微调 "范式将对各种数据稀缺的视觉问题非常有效。

We conclude by noting that it is significant that we achieved these results by using a combination of classical tools from computer vision and deep learning (bottom-up region proposals and convolutional neural networks). Rather than opposing lines of scientific inquiry, the two are natural and inevitable partners. Acknowledgments. This research was supported in part by DARPA Mind’s Eye and MSEE programs, by NSF awards IIS-0905647, IIS-1134072, and IIS-1212798, MURI N000014-10-1-0933, and by support from Toyota. The GPUs used in this research were generously donated by the NVIDIA Corporation.

我们最后指出，重要的是，我们通过使用计算机视觉和深度学习的经典工具（自下而上的区域建议和卷积神经网络）的组合来实现这些结果。与其说是科学探究的对立路线，不如说这两者是自然的、不可避免的伙伴。鸣谢。这项研究得到了DARPA心智之眼和MSEE项目、美国国家科学基金会IIS-0905647、IIS-1134072和IIS-1212798奖项、MURI N000014-10-1-0933的部分支持，并得到了丰田公司的支持。本研究中使用的GPU是由英伟达公司慷慨捐赠的。

