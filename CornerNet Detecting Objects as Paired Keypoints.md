# CornerNet: Detecting Objects as Paired Keypoints

## Abstract. 

We propose CornerNet, a new approach to object detection where we detect an object bounding box as a pair of keypoints, the top-left corner and the bottom-right corner, using a single convolution neural network. By detecting objects as paired keypoints, we eliminate（消除） the need for designing a set of anchor boxes commonly used in prior single-stage detectors. In addition to our novel formulation, we introduce corner pooling, a new type of pooling layer that helps the network better localize corners. Experiments show that CornerNet achieves a 42.1% AP on MS COCO, outperforming all existing one-stage detectors.

我们提出了CornerNet，这是一种新的物体检测方法，我们使用一个单一的卷积神经网络将物体的边界盒检测为一对关键点，即左上角和右下角。通过将物体作为成对的关键点进行检测，我们不需要设计一组锚定框，这在以前的单阶段检测中是常用的。除了我们新颖的表述外，我们还引入了corner pooling，一种新型的池层，帮助网络更好地定位corners。实验表明，CornerNet在MS COCO上取得了42.1%的AP，超过了所有现有的单阶段检测器。

## 1 Introduction

Object detectors based on convolutional neural networks (ConvNets) [20, 36, 15] have achieved state-of-the-art results on various challenging benchmarks [24, 8, 9]. A common component of state-of-the-art approaches is anchor boxes [32, 25], which are boxes of various sizes and aspect ratios（长宽比） that serve as detection candidates. Anchor boxes are extensively used in one-stage detectors [25, 10, 31, 23], which **can achieve results highly competitive with** two-stage detectors [32, 12, 11, 13] while being more efficient. One-stage detectors place anchor boxes densely（密集地） over an image and generate final box predictions by scoring anchor boxes and refining their coordinates through regression. But the use of anchor boxes has two drawbacks. First, we typically need a very large set of anchor boxes, e.g. more than 40k in DSSD [10] and more than 100k in RetinaNet [23]. This is because the detector is trained to classify whether each anchor box sufficiently overlaps（充分重叠） with a ground truth box, and a large number of anchor boxes is needed to ensure sufficient overlap with most ground truth boxes. As a result, only **a tiny fraction of** anchor boxes will overlap with ground truth; this creates a huge imbalance between positive and negative anchor boxes and slows down training [23]. Second, the use of anchor boxes introduces many hyperparameters and design choices. These include how many boxes, what sizes, and what aspect ratios. Such choices have largely been made via ad-hoc heuristics, and can become even more complicated when combined with multiscale architectures where a single network makes separate predictions at multiple resolutions, with each scale using different features and its own set of anchor boxes [25, 10, 23]. 

基于卷积神经网络（ConvNets）的物体检测器[20, 36, 15]已经在各种挑战性的基准上取得了最先进的结果[24, 8, 9]。最先进的方法的一个共同组成部分是锚定盒[32, 25]，即各种尺寸和长宽比的盒子，作为检测候选对象。锚箱被广泛用于单阶段检测器[25, 10, 31, 23]，它可以取得与双阶段检测器[32, 12, 11, 13]高度竞争的结果，同时更有效率。一阶段检测器在图像上密集地放置锚定盒，并通过对锚定盒的评分和通过回归完善其坐标来生成最终的盒子预测。但使用锚箱有两个缺点。首先，我们通常需要一个非常大的锚箱集，例如在DSSD[10]中超过40k，在RetinaNet[23]中超过100k。这是因为检测器的训练是为了对每个锚定框是否与地面真相框充分重叠进行分类，需要大量的锚定框来确保与大多数地面真相框充分重叠。因此，只有极小部分的锚箱会与地面实况重叠；这就造成了正负锚箱之间的巨大不平衡，减缓了训练速度[23]。其次，锚箱的使用引入了许多超参数和设计选择。这包括多少个盒子，什么尺寸，以及什么长宽比。这些选择主要是通过临时的启发式方法做出的，当与多尺度结构相结合时，会变得更加复杂，在这种结构中，一个网络在多个分辨率下进行单独的预测，每个尺度使用不同的特征和它自己的锚定框[25, 10, 23]。

In this paper we introduce CornerNet, a new one-stage approach to object detection that does away with anchor boxes. We detect an object as a pair of keypoints—the top-left corner and bottom-right corner of the bounding box. We use a single convolutional network to predict a heatmap for the top-left corners of all instances of the same object category, a heatmap for all bottom-right corners, and an embedding vector for each detected corner. The embeddings serve to group a pair of corners that belong to the same object—the network is trained to predict similar embeddings for them. Our approach greatly simplifies the output of the network and eliminates the need for designing anchor boxes. Our approach is inspired by the associative embedding method proposed by Newell et al. [27], who detect and group keypoints in the context of multiperson human-pose estimation. Fig. 1 illustrates the overall pipeline of our approach.

![image-20220429164115694](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429164115694.png)

在本文中，我们介绍了CornerNet，这是一种新的单阶段的物体检测方法，不需要锚定盒。我们将一个物体检测为一对关键点--包围盒的左上角和右下角。我们使用一个单一的卷积网络来预测同一物体类别的所有实例的左上角的热图，所有右下角的热图，以及每个检测到的角落的嵌入向量。嵌入的作用是将属于同一物体的一对角分组--网络被训练成预测它们的类似嵌入。我们的方法大大简化了网络的输出，并消除了设计锚定盒的需要。我们的方法受到Newell等人[27]提出的关联嵌入方法的启发，他们在多人姿势估计的背景下检测和分组关键点。图1说明了我们方法的整体流程。

Another novel component of CornerNet is corner pooling, a new type of pooling layer that helps a convolutional network better localize corners of bounding boxes. A corner of a bounding box is often outside the object—consider the case of a circle as well as the examples in Fig. 2. In such cases a corner cannot be localized based on local evidence. Instead, to determine whether there is a top-left corner at a pixel（像素点） location, we need to look horizontally（水平地） towards the right for the topmost boundary （最上边界）of the object, and look vertically（垂直地） towards the bottom for the leftmost boundary（最左边界）. *This motivates our corner pooling layer*: it takes in（吸收） two feature maps; at each pixel location（在每一个像素点） it **max-pools all feature vectors to the right** from the first feature map, **max-pools all feature vectors directly below** from the second feature map, and then **adds the two pooled results together**. An example is shown in Fig. 3.

![image-20220429164803559](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429164803559.png)

![image-20220429164838219](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429164838219.png)

CornerNet的另一个新组件是corner pooling,，这是一种新型的集合层，帮助卷积网络更好地定位边界盒的角。界限盒的一个角往往在物体之外--考虑到圆的情况以及图2中的例子。在这种情况下，不能根据局部证据对角进行定位。相反，为了确定在一个像素点上是否有一个左上角，我们需要在水平方向上向右寻找物体的最上边界，在垂直方向上向下寻找最左边界。这就促使我们的角集合层：它吸收了两个特征图；在每个像素位置，它从第一个特征图中最大限度地集合右边的所有特征向量，从第二个特征图中最大限度地集合正下方的所有特征向量，然后将两个集合的结果加在一起。图3中显示了一个例子。

We hypothesize(假设) two reasons why detecting corners would work better than bounding box centers or proposals. First, **the center of a box can be harder to localize because it depends on all 4 sides of the object**, **whereas locating a corner depends on 2 sides and is thus easier**, and even more so with corner pooling, which encodes some explicit（明确） prior knowledge（先验知识） about the definition of corners. Second, corners provide a more efficient way of densely discretizing（离散） the space of boxes: we just need O(wh) corners to represent O(w_2 h_2 ) possible anchor boxes. We demonstrate the effectiveness of CornerNet on MS COCO [24]. CornerNet achieves a 42.1% AP, outperforming all existing one-stage detectors. In addition, through ablation studies we show that corner pooling is critical to the superior performance of CornerNet. Code is available at https://github.com/umichvl/CornerNet

我们假设有两个原因可以说明为什么检测角落会比边界盒中心或建议效果更好。首先，一个盒子的中心可能更难定位，因为它取决于物体的所有4个面，而定位一个角则取决于2个面，因此更容易，甚至在角集合的情况下更容易，因为它编码了一些关于角的定义的明确的先验知识。其次，角提供了一种更有效的密集离散盒子空间的方式：我们只需要O(wh)个角来代表O(w_2 h_2)个可能的锚点盒子。我们在MS COCO[24]上证明了CornerNet的有效性。CornerNet实现了42.1%的AP，超过了所有现有的单阶段检测器。此外，通过消融研究，我们表明角池对CornerNet的卓越性能至关重要。代码可在https://github.com/umichvl/CornerNet。

## 2 Related Works

### Two-stage object detectors 

Two-stage approach was first introduced and popularized by R-CNN [12]. Two-stage detectors generate a sparse set of regions of interest (RoIs) and classify each of them by a network. R-CNN generates RoIs using a low level vision algorithm [41, 47]. Each region is then extracted from the image and processed by a ConvNet independently, which creates lots of redundant computations. Later, SPP [14] and Fast-RCNN [11] improve R-CNN by designing a special pooling layer that pools each region from feature maps instead. However, both still rely on separate proposal algorithms and cannot be trained end-to-end. Faster-RCNN [32] does away low level proposal algorithms by introducing a region proposal network (RPN), which generates proposals from a set of pre-determined candidate boxes, usually known as anchor boxes. This not only makes the detectors more efficient but also allows the detectors to be trained end-to-end. R-FCN [6] further improves the efficiency of Faster-RCNN by replacing the fully connected sub-detection network with a fully convolutional sub-detection network. Other works focus on incorporating sub-category information [42], generating object proposals at multiple scales with more contextual information [1, 3, 35, 22], selecting better features [44], improving speed [21], cascade procedure [4] and better training procedure [37].

两阶段方法是由R-CNN[12]首次提出并推广的。两阶段检测器产生一个稀疏的兴趣区域（RoIs）集，并通过网络对每个区域进行分类。R-CNN使用低级视觉算法生成RoIs[41, 47]。然后每个区域从图像中提取出来，由ConvNet独立处理，这就产生了很多冗余的计算。后来，SPP[14]和Fast-RCNN[11]通过设计一个特殊的池化层来改进R-CNN，代替从特征图中池化每个区域。然而，两者仍然依赖于单独的提议算法，不能进行端到端的训练。Faster-RCNN[32]通过引入一个区域提议网络（RPN）来摒弃低级别的提议算法，该网络从一组预先确定的候选框（通常称为锚框）中生成提议。这不仅使检测器更有效率，而且还允许检测器进行端到端的训练。R-FCN[6]通过用全卷积子检测网络取代全连接子检测网络，进一步提高了Faster-RCN的效率。其他工作的重点是纳入子类别信息[42]，用更多的上下文信息在多个尺度上生成物体建议[1, 3, 35, 22]，选择更好的特征[44]，提高速度[21]，级联程序[4]和更好的训练程序[37]。

### One-stage object detectors 

On the other hand, YOLO [30] and SSD [25] have popularized the one-stage approach, which removes the RoI pooling step and detects objects in a single network. One-stage detectors are usually more computationally efficient than two-stage detectors while maintaining competitive performance on different challenging benchmarks. SSD places anchor boxes densely over feature maps from multiple scales, directly classifies and refines each anchor box. YOLO predicts bounding box coordinates directly from an image, and is later improved in YOLO9000 [31] by switching to anchor boxes. DSSD [10] and RON [19] adopt networks similar to the hourglass network [28], enabling them to combine low-level and high-level features via skip connections to predict bounding boxes more accurately. However, these one-stage detectors are still outperformed by the two-stage detectors until the introduction of RetinaNet [23]. In [23], the authors suggest that the dense anchor boxes create a huge imbalance between positive and negative anchor boxes during training. This imbalance causes the training to be inefficient and hence the performance to be suboptimal. They propose a new loss, Focal Loss, to dynamically adjust the weights of each anchor box and show that their one-stage detector can outperform the two-stage detectors. RefineDet [45] proposes to filter the anchor boxes to reduce the number of negative boxes, and to coarsely adjust the anchor boxes. DeNet [39] is a two-stage detector which generates RoIs without using anchor boxes. It first determines how likely each location belongs to either the top-left, top-right, bottom-left or bottom-right corner of a bounding box. It the generates RoIs by enumerating all possible corner combinations, and follows the standard two-stage approach to classify each RoI. Our approach is very different from DeNet. First, DeNet does not identify if two corners are from the same objects and relies on a sub-detection network to reject poor RoIs. In contrast, our approach is a one-stage approach which detects and groups the corners using a single ConvNet. Second, DeNet selects features at manually determined locations relative to a region for classification, while our approach does not require any feature selection step. Third, we introduce corner pooling, a novel type of layer to enhance corner detection. Our approach is inspired by Newell et al. work [27] on Associative Embedding in the context of multi-person pose estimation. Newell et al. propose an approach that detects and groups human joints in a single network. In their approach each detected human joint has an embedding vector. The joints are grouped based on the distances between their embeddings. To the best of our knowledge, we are the first to formulate the task of object detection as a task of detecting and grouping corners simultaneously. Another novelty of ours is the corner pooling layers that help better localize the corners. We also significantly modify the hourglass architecture and add our novel variant of focal loss [23] to help better train the network.

另一方面，YOLO[30]和SSD[25]已经普及了单阶段方法，它取消了RoI池的步骤，在单个网络中检测物体。单阶段检测器通常比两阶段检测器的计算效率更高，同时在不同的挑战性基准上保持有竞争力的性能。SSD在多个尺度的特征图上密集放置锚盒，直接对每个锚盒进行分类和细化。YOLO直接从图像中预测边界盒坐标，后来在YOLO9000[31]中通过改用锚定盒进行改进。DSSD[10]和RON[19]采用了与沙漏网络[28]类似的网络，使它们能够通过跳过连接将低级和高级特征结合起来，更准确地预测边界盒。然而，这些单阶段检测器在引入RetinaNet[23]之前，仍然比双阶段检测器的性能要好。在[23]中，作者提出密集的锚箱在训练过程中造成了正负锚箱之间的巨大不平衡。这种不平衡导致训练效率低下，从而使性能处于次优状态。他们提出了一种新的损失，即Focal Loss，以动态调整每个锚箱的权重，并表明他们的单阶段检测器可以胜过两阶段检测器。RefineDet[45]提出对锚箱进行过滤以减少负箱的数量，并对锚箱进行粗略的调整。DeNet[39]是一个两阶段的检测器，它不使用锚定框来生成RoI。它首先确定每个位置属于一个边界盒的左上角、右上角、左下角或右下角的可能性。它通过列举所有可能的角的组合来生成RoI，并遵循标准的两阶段方法对每个RoI进行分类。我们的方法与DeNet非常不同。首先，DeNet不能识别两个角是否来自相同的物体，而是依靠一个子检测网络来拒绝不良的RoI。相比之下，我们的方法是一个单阶段的方法，使用单一的ConvNet检测和分组角落。第二，DeNet在手工确定的相对于区域的位置选择特征进行分类，而我们的方法不需要任何特征选择步骤。第三，我们引入了角落池，这是一种新型的层来加强角落检测。我们的方法受到Newell等人在多人姿势估计方面的关联嵌入工作[27]的启发。Newell等人提出了一种在单一网络中检测和分组人体关节的方法。在他们的方法中，每个检测到的人体关节都有一个嵌入向量。关节根据其嵌入之间的距离进行分组。据我们所知，我们是第一个将物体检测的任务表述为同时检测和分组角落的任务。我们的另一个创新之处在于角集合层，它有助于更好地定位角。我们还对沙漏结构进行了大幅修改，并增加了新的焦点损失[23]的变体，以帮助更好地训练网络。

## 3 CornerNet

### 3.1 Overview

In CornerNet, we detect an object as a pair of keypoints—the top-left corner and bottom-right corner of the bounding box. A convolutional network predicts two sets of heatmaps to represent the locations of corners of different object categories, one set for the top-left corners and the other for the bottom-right corners. **The network also predicts an embedding vector for each detected corner [27] such that the distance between the embeddings of two corners from the same object is small.** 

To produce tighter bounding boxes, the network also predicts offsets to slightly adjust the locations of the corners. With the predicted heatmaps, embeddings and offsets, we apply a simple post-processing algorithm to obtain the final bounding boxes. Fig. 4 provides an overview of CornerNet. We use the **hourglass network** [28] as the backbone network of CornerNet. The hourglass network is **followed by two prediction modules**. One module is for the top-left corners, while the other one is for the bottom-right corners. **Each module has its own corner pooling module to pool features from the hourglass network before predicting the heatmaps, embeddings and offsets.** Unlike many other object detectors, we do not use features from different scales to detect objects of different sizes. We only apply both modules to the output of the hourglass network.

在CornerNet中，我们将一个物体检测为一对关键点--包围盒的左上角和右下角。卷积网络预测了两组热图来表示不同物体类别的角的位置，一组是左上角，另一组是右下角。该网络还为每个检测到的角预测了一个嵌入向量[27]，这样来自同一物体的两个角的嵌入之间的距离就很小。

为了产生更紧密的边界盒，该网络还预测了偏移量，以略微调整角的位置。有了预测的热图、嵌入和偏移，我们应用一个简单的后处理算法来获得最终的边界盒。图4提供了CornerNet的概况。我们使用**沙漏网络**[28]作为CornerNet的主干网络。沙漏网络的后面有两个预测模块**。一个模块用于预测左上角，另一个模块用于预测右下角。**每个模块都有自己的角落汇集模块，在预测热图、嵌入和偏移之前，汇集来自沙漏网络的特征。与许多其他物体检测器不同，我们不使用不同尺度的特征来检测不同尺寸的物体。我们只将这两个模块应用于沙漏网络的输出。

![image-20220429174422291](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429174422291.png)

### 3.2 Detecting Corners

We predict two sets of heatmaps, one for top-left corners and one for bottom-right corners. Each set of heatmaps has C channels, **where C is the number of categories**, and is of size H × W. There is no background channel. **Each channel is a binary mask indicating the locations of the corners for a class.** 

**For each corner, there is one ground-truth positive location, and all other locations are negative. During training, instead of equally penalizing(惩罚) negative locations, we reduce the penalty given to negative locations within a radius of the positive location. This is because a pair of false corner detections, if they are close to their respective ground truth locations, can still produce a box that sufficiently overlaps the ground-truth box (Fig. 5). We determine the radius by the size of an object by ensuring that a pair of points within the radius would generate a bounding box with at least *t*  IoU with the ground-truth annotation (we set *t* to 0.7 in all experiments). Given the radius, the amount of penalty reduction is given by an unnormalized 2D Gaussian,** 
$$
e − (x^2+y^2)/2σ^2
$$
 , whose center is at the positive location and whose *σ* is 1/3 of the radius.



![image-20220429175113712](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429175113712.png)

我们预测了两套热图，一套用于左上角，另一套用于右下角。每组热图都有C个通道，**其中C是类别的数量**，大小为H×W，不存在背景通道。每个通道是一个二进制掩码，表示一个类别的角的位置。

对于每个角，有一个真实的正面位置，所有其他位置都是负面的。在训练过程中，我们不是对负面位置进行平均惩罚，而是减少对正面位置半径内的负面位置的惩罚。这是因为一对错误的角落检测，如果它们接近各自的地面真相位置，仍然可以产生一个与地面真相方框充分重叠的方框（图5）。我们通过物体的大小来确定半径，确保半径内的一对点会产生一个与地面实况注释至少*t* IoU的边界盒（我们在所有实验中把*t*设置为0.7）。给定半径后，惩罚的减少量由一个非正常化的二维高斯给出

Let $ p_cij $ be the score at location (*i, j*) for class *c* in the predicted heatmaps, and let $y_cij$ be the “ground-truth” heatmap augmented with the unnormalized Gaussians. We design a variant of focal loss [23]:

![image-20220429172924830](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429172924830.png)

where N is the number of objects in an image, and *α* and *β* are the hyperparameters which control the contribution of each point (we set *α* to 2 and *β* to 4 in all experiments). With the Gaussian bumps encoded in $y_cij$ , the (1 − $y_cij$ ) term reduces the penalty around the ground truth locations. Many networks [15, 28] involve downsampling layers to gather global information and to reduce memory usage. When they are applied to an image fully convolutionally, the size of the output is usually smaller than the image. Hence, a location *(x, y)* in the image is mapped to the location 

![image-20220429172946016](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429172946016.png)

where *o_k* is the offset, *x_k* and *y_k* are the x and y coordinate for corner *k*. In particular, we predict one set of offsets shared by the top-left corners of all categories, and another set shared by the bottom-right corners. For training, we apply the smooth L1 Loss [11] at ground-truth corner locations:

![image-20220429173000435](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429173000435.png)

### 3.3 Grouping Corners

Multiple objects may appear in an image, and thus multiple top-left and bottom-right corners may be detected. We need to determine if a pair of the top-left corner and bottom-right corner is from the same bounding box. Our approach is inspired by the Associative Embedding method proposed by Newell et al. [27] for the task of multi-person pose estimation. Newell et al. detect all human joints and generate an embedding for each detected joint. They group the joints based on the distances between the embeddings.

 The idea of associative embedding is also applicable to our task. The network predicts an embedding vector for each detected corner such that if a top-left corner and a bottom-right corner belong to the same bounding box, the distance between their embeddings should be small. We can then group the corners based on the distances between the embeddings of the top-left and bottom-right corners. The actual values of the embeddings are unimportant. Only the distances between the embeddings are used to group the corners. 

一个图像中可能出现多个物体，因此可能检测到多个左上角和右下角。我们需要确定一对左上角和右下角是否来自同一个包围盒。我们的方法受到Newell等人[27]为多人姿势估计任务提出的关联嵌入方法的启发。Newell等人检测所有的人体关节并为每个检测到的关节生成一个嵌入。他们根据嵌入之间的距离对这些关节进行分组。

 关联嵌入的想法也适用于我们的任务。该网络为每个检测到的角预测一个嵌入向量，这样，如果左上角和右下角属于同一个边界盒，它们的嵌入之间的距离应该很小。然后，我们可以根据左上角和右下角的嵌入之间的距离对角进行分组。嵌入的实际值并不重要。只有嵌入之间的距离被用来分组角。

We follow Newell et al. [27] and use embeddings of 1 dimension. Let etk be the embedding for the top-left corner of object k and ebk for the bottom-right corner. As in [26], **we use the “pull” loss to train the network to group the corners and the “push” loss to separate the corners:**

![image-20220429173035773](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429173035773.png)

where ek is the average of etk and ebk and we set ∆ to be 1 in all our experiments. Similar to the offset loss, we only apply the losses at the ground-truth corner location.

### 3.4 Corner Pooling

As shown in Fig. 2, there is often no local visual evidence for the presence of corners. To determine if a pixel is a top-left corner, we need to look horizontally towards the right for the topmost boundary of an object and vertically towards the bottom for the leftmost boundary. We thus propose corner pooling to better localize the corners by encoding explicit prior knowledge.

![image-20220429173111465](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429173111465.png)

Suppose we want to determine if a pixel at location (i, j) is a top-left corner. Let ft and fl be the feature maps that are the inputs to the top-left corner pooling layer, and let ftij and flij be the vectors at location (i, j) in ft and fl respectively. With H ×W feature maps, the corner pooling layer first max-pools all feature vectors between (i, j) and (i, H) in ft to a feature vector tij , and max-pools all feature vectors between (i, j) and (W, j) in fl to a feature vector lij . Finally, it adds tij and lij together. This computation can be expressed by the following equations:

![image-20220429173137277](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429173137277.png)

where we apply an elementwise max operation. Both tij and lij can be computed efficiently by dynamic programming as shown Fig. 6. We define bottom-right corner pooling layer in a similar way. It max-pools all feature vectors between (0, j) and (i, j), and all feature vectors between (i, 0) and (i, j) before adding the pooled results. The corner pooling layers are used in the prediction modules to predict heatmaps, embeddings and offsets.

![image-20220429173155981](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220429173155981.png)

The architecture of the prediction module is shown in Fig. 7. The first part of the module is a modified version of the residual block [15]. In this modified residual block, we replace the first 3×3 convolution module with a corner pooling module, which first processes the features from the backbone network by two 3×3 convolution modules 1 with 128 channels and then applies a corner pooling layer. Following the design of a residual block, we then feed the pooled features into a 3 × 3 Conv-BN layer with 256 channels and add back the projection shortcut. The modified residual block is followed by a 3 × 3 convolution module with 256 channels, and 3 Conv-ReLU-Conv layers to produce the heatmaps, embeddings and offsets.

### 3.5 Hourglass Network

CornerNet uses the hourglass network [28] as its backbone network. The hourglass network was first introduced for the human pose estimation task. It is a fully convolutional neural network that consists of one or more hourglass modules. An hourglass module first downsamples the input features by a series of convolution and max pooling layers. It then upsamples the features back to the original resolution by a series of upsampling and convolution layers. Since details are lost in the max pooling layers, skip layers are added to bring back the details to the upsampled features. The hourglass module captures both global and local features in a single unified structure. When multiple hourglass modules are stacked in the network, the hourglass modules can reprocess the features to capture higher-level of information. These properties make the hourglass network an ideal choice for object detection as well. In fact, many current detectors [35, 10, 22, 19] already adopted networks similar to the hourglass network. 

Our hourglass network consists of two hourglasses, and we make some modifications to the architecture of the hourglass module. Instead of using max pooling, we simply use stride 2 to reduce feature resolution. We reduce feature resolutions 5 times and increase the number of feature channels along the way (256, 384, 384, 384, 512). When we upsample the features, we apply 2 residual modules followed by a nearest neighbor upsampling. Every skip connection also consists of 2 residual modules. There are 4 residual modules with 512 channels in the middle of an hourglass module. Before the hourglass modules, we reduce the image resolution by 4 times using a 7 × 7 convolution module with stride 2 and 128 channels followed by a residual block [15] with stride 2 and 256 channels. 

Following [28], we also add intermediate supervision in training. However, we do not add back the intermediate predictions to the network as we find that this hurts the performance of the network. We apply a 3 × 3 Conv-BN module to both the input and output of the first hourglass module. We then merge them by element-wise addition followed by a ReLU and a residual block with 256 channels, which is then used as the input to the second hourglass module. The depth of the hourglass network is 104. Unlike many other state-of-the-art detectors, we only use the features from the last layer of the whole network to make predictions.