## abstract

  每年有大量的路面会遭受到不同的破坏，比如路面上的裂缝等，这将对交通安全造成巨大的安全问题，所以及时找出这些路面破损位置并修复他们是及其重要的。当前道路养护部门都是配备专门的检车车辆，例如激光扫描车对路面破损位置进行检测，这些措施虽然可以得到很好的效果，但他的经济消耗非常高。针对这个问题，近几年也有很多的学者试图研究一些更加节约成本的方法来解决这个问题。比如L. Zhang等人和H. Maeda等人尝试使用普通的安卓手机进行路损检测。但是其速度和准确率都不佳。随后在2019年，Siyu Chen, Yin Zhang 等人开发了一个通过深度神经网络检测路面裂缝的嵌入式系统，其在速度和准确率上都比L. Zhang等人和H. Maeda等人的成果要好。但是其速度上还没有达到实时检测的效果，一旦车速过快，其嵌入式系统的工作效果不好，而且召回率还有待提高，仍然有不少数的裂缝检测不出来。

​    在本文，我们对当前目标检测发展历程进行研究，最终选择当前在目标检测处于领先地位的yolov5,并对其进行改进，使之更加适合我们所需要的效果，并用之前Siyu Chen, Yin Zhang 等人使用的数据集，对每一个模型进行训练，然后将他们部署在嵌入式板子上，得到了比他们更好的准确率和检测速度。分别是。。。。。。得出哪个模型准确率和推理速度都是最高的。然后我们尝试用新的数据集rdd2019,rdd2020和之前的数据集进行合并，训练出一个更加普遍适用的模型。其准确率为。。。

## 1. introduction

  随着经济的不断增长，运输业也在高速发展。世界上的公路里程正在增加。在道路总长度不断增加的同时，老旧的道路也随着时间的推移慢慢损坏。随着时间的推移，老旧的道路逐渐被损坏，各地区的道路维修部门面临着 越来越多的维修需求。在道路维修部门进行路面维修工作之前，发现路面的损坏是一个必要的步骤。传统上，道路损坏检测主要是通过人工搜索来实现的。的方式，这种方式非常耗费时间和人力。迫切需要开发一种能够轻松发现道路 亟需开发一种能够轻松发现道路损坏的方案。事实上，已经有一些学者[1]研究了这个问题。

  目前，道路损坏的自动检测主要通过以下三种方法进行 激光、雷达和视觉。2006年，Hinton等人提出了深度学习的概念[2]。从那时起，基于深度学习的技术 深度学习的技术开始迅速发展。基于深度神经网络的计算机视觉的研究和产品开发 基于深度神经网络的计算机视觉研究和产品开发正在迅速崛起。利用图像处理技术和 学者[3,4]利用图像处理技术和深度神经网络，对路损检测方案进行了新一轮的探索和实践。在路损检测的方案上进行了新一轮的探索和实践。在2019年，本文基于刘伟等人Siyu Chen, Yin Zhang 等人提出的SSD物体检测方法，设计了一个嵌入式系统。实现了比以往更好的效果。本文运行 它运行一个物体检测模型，可以通过普通摄像头检测道路损坏。当道路损坏系统可以通过GPS模块自动获取地理位置信息，并将相关数据保存在内存中。模块，并将相关数据保存在内存中，包括图像和位置信息。该系统已经 经测试，该系统以较低的经济成本达到了有用的检测召回率。

  在接下来的第二部分，本文将介绍一些相关工作，包括先前对于路面裂缝不同的解决方法，和当前目标检测的发展历程，以及目前最先进的目标检测模型。在第三部分，我们将介绍用什么样的模型来来训练数据集，以及整个路面裂缝检测系统，第四部分我们将对方法进行验证，包括在训练模型阶段，选出最佳的模型，以及在真实场景中进行实验验证，并于先前工作进行对比和进行分析，第五部分将对我们的工作做出总结与展望。

## 2. related Work

  目标检测任务是找出图像或视频中人们感兴趣的物体，并同时检测出它们的位置和大小。不同于图像分类任务，目标检测不仅要解决分类问题，还要解决定位问题，是属于Multi-Task的问题。目标检测的发展脉络可以划分为两个周期：传统目标检测算法时期(1998年-2014年)和基于深度学习的目标检测算法时期(2014年-至今)。而基于深度学习的目标检测算法又发展成了两条技术路线：**Anchor based**方法(一阶段，二阶段)和**Anchor free**方法。

### 2.1 two-stage

二阶段检测算法主要分为以下两个阶段**Stage1**：从图像中生成region proposals**Stage2**：从region proposals生成最终的物体边框。

#### **RCNN**

**【简介】** **RCNN[4]**由Ross Girshick于2014年提出，RCNN首先通过选择性搜索算法Selective Search从一组对象候选框中选择可能出现的对象框，然后将这些选择出来的对象框中的图像resize到某一固定尺寸的图像，并喂入到CNN模型(经过在ImageNet数据集上训练过的CNN模型，如AlexNet)提取特征，最后将提取出的特征送入到![[公式]](https://www.zhihu.com/equation?tex=SVM)分类器来预测该对象框中的图像是否存在待检测目标，并进一步预测该检测目标具体属于哪一类。

**【性能】** RCNN算法在VOC-07数据集上取得了非常显著的效果，平均精度由33.7%(DPM-V5, 传统检测的SOTA算法)提升到58.5%。相比于传统检测算法，基于深度学习的检测算法在精度上取得了质的飞跃。

**【不足】** 虽然RCNN算法取得了很大进展，但缺点也很明显：重叠框(一张图片大2000多个候选框)特征的冗余计算使得整个网络的检测速度变得很慢(使用GPU的情况下检测一张图片大约需要14S)。

为了减少大量重叠框带来的冗余计算，K. He等人提出了SPPNet

### **SPPNet**

**【简介】** **SPPNet[5]**提出了一种空间金字塔池化层(Spatial Pyramid Pooling Layer, SPP)。它的主要思路是对于一副图像分成若干尺度的图像块(比如一副图像分成1份，4份，8份等)，然后对每一块提取的特征融合在一起，从而兼顾多个尺度的特征。SPP使得网络在全连接层之前能生成固定尺度的特征表示，而不管输入图片尺寸如何。当使用SPPNet网络用于目标检测时，整个图像只需计算一次即可生成相应特征图，不管候选框尺寸如何，经过SPP之后，都能生成固定尺寸的特征表示图，这避免了卷积特征图的重复计算。

**【性能】** 相比于RCNN算法，SPPNet在Pascal-07数据集上不牺牲检测精度(VOC-07, mAP=59.2%)的情况下，推理速度提高了20多倍。

**【不足】** 和RCNN一样，SPP也需要训练CNN提取特征，然后训练SVM分类这些特征，这需要巨大的存储空间，并且多阶段训练的流程也很繁杂。除此之外，SPPNet只对全连接层进行微调，而忽略了网络其它层的参数。

### **Fast RCNN**

**【简介】** **Fast RCNN[6]**网络是RCNN和SPPNet的改进版，该网路使得我们可以在相同的网络配置下同时训练一个检测器和边框回归器。该网络首先输入图像，图像被传递到CNN中提取特征，并返回感兴趣的区域ROI，之后再ROI上运用ROI池化层以保证每个区域的尺寸相同，最后这些区域的特征被传递到全连接层的网络中进行分类，并用Softmax和线性回归层同时返回边界框。

**【性能】** Fast RCNN在VOC-07数据集上将检测精度mAP从58.5%提高到70.0%，检测速度比RCNN提高了200倍。

**【不足】** Fast RCNN仍然选用选择性搜索算法来寻找感兴趣的区域，这一过程通常较慢，与RCNN不同的是，Fast RCNN处理一张图片大约需要2秒，但是在大型真实数据集上，这种速度仍然不够理想。

### **Faster RCNN**

**【简介】** **Faster RCNN[7]**是第一个端到端，最接近于实时性能的深度学习检测算法，该网络的主要创新点就是提出了区域选择网络用于申城候选框，能几大提升检测框的生成速度。该网络首先输入图像到卷积网络中，生成该图像的特征映射。在特征映射上应用Region Proposal Network，返回object proposals和相应分数。应用Rol池化层，将所有proposals修正到同样尺寸。最后，将proposals传递到完全连接层，生成目标物体的边界框。

**【性能】** 该网络在当时VOC-07，VOC-12和COCO数据集上实现了SOTA精度，其中COCO mAP@.5=42.7%, COCO mAP@[.5,.95]=21.9%, VOC07 mAP=73.2%, VOC12 mAP=70.4%, 17fps with ZFNet

**【不足】** 虽然Faster RCNN的精度更高，速度更快，也非常接近于实时性能，但它在后续的检测阶段中仍存在一些计算冗余；除此之外，如果IOU阈值设置的低，会引起噪声检测的问题，如果IOU设置的高，则会引起过拟合。

### **FPN**

**【简介】** 2017年，T.-Y.Lin等人在Faster RCNN的基础上进一步提出了特征金字塔网络**FPN[8]**(Feature Pyramid Networks)技术。在FPN技术出现之前，大多数检测算法的检测头都位于网络的最顶层(最深层)，虽说最深层的特征具备更丰富的语义信息，更有利于物体分类，但更深层的特征图由于空间信息的缺乏不利于物体定位，这大大影响了目标检测的定位精度。为了解决这一矛盾，FPN提出了一种具有横向连接的自上而下的网络架构，用于在所有具有不同尺度的高底层都构筑出高级语义信息。FPN的提出极大促进了检测网络精度的提高(尤其是对于一些待检测物体尺度变化大的数据集有非常明显的效果)。

**【性能】** 将FPN技术应用于Faster RCNN网络之后，网络的检测精度得到了巨大提高(COCO mAP@.5=59.1%, COCO mAP@[.5,.95]=36.2%)，再次成为当前的SOTA检测算法。此后FPN成为了各大网络(分类，检测与分割)提高精度最重要的技术之一。

### **Cascade RCNN**

**【简介】** Faster RCNN完成了对目标候选框的两次预测，其中RPN一次，后面的检测器一次，而**Cascade RCNN[9]**则更进一步将后面检测器部分堆叠了几个级联模块，并采用不同的IOU阈值训练，这种级联版的Faster RCNN就是Cascade RCNN。通过提升IoU阈值训练级联检测器，可以使得检测器的定位精度更高，在更为严格的IoU阈值评估下，Cascade R-CNN带来的性能提升更为明显。Cascade RCNN将二阶段目标检测算法的精度提升到了新的高度。

**【性能】** Cascade RCNN在COCO检测数据集上，不添加任何Trick即可超过现有的SOTA单阶段检测器，此外使用任何基于RCNN的二阶段检测器来构建Cascade RCNN，mAP平均可以提高2-4个百分点。

### 2.2 one-stage

一阶段目标检测算法不需要region proposal阶段，直接产生物体的类别概率和位置坐标值，经过一个阶段即可直接得到最终的检测结果，因此有着更快的检测速度。

#### anchor-base

###  **YOLO v1**

**【简介】** **YOLO v1[10]**是第一个一阶段的深度学习检测算法，其检测速度非常快，该算法的思想就是将图像划分成多个网格，然后为每一个网格同时预测边界框并给出相应概率。例如某个待检测目标的中心落在图像中所划分的一个单元格内，那么该单元格负责预测该目标位置和类别。

**【性能】** YOLO v1检测速度非常快，在VOC-07数据集上的mAP可达52.7%，实现了155 fps的实时性能，其增强版性能也很好(VOC-07 mAP=63.4%, 45 fps, VOC-12 mAP=57.9%)，性能要优于DPM和RCNN。

**【不足】** 相比于二阶段的目标检测算法，尽管YOLO v1算法的检测速度有了很大提高，但精度相对教低(尤其是对于一些小目标检测问题)。

### **SSD**

**【简介】** **SSD[11]**算法的主要创新点是提出了Multi-reference和Multi-resolution的检测技术。SSD算法和先前的一些检测算法的区别在于：先前的一些检测算法只是在网络最深层的分支进行检测，而SSD有多个不同的检测分支，不同的检测分支可以检测多个尺度的目标，所以SSD在多尺度目标检测的精度上有了很大的提高，对小目标检测效果要好很多。

**【性能】** 相比于YOLO v1算法，SSD进一步提高了检测精度和速度(VOC-07 mAP=76.8%, VOC-12 mAP=74.9%, COCO mAP@.5=46.5%, mAP@[.5,.95]=26.8%, SSD的精简版速度达到59 fps)。

### **YOLO v2**

**【简介】** 相比于YOLO v1，**YOLO v2[12]**在精度、速度和分类数量上都有了很大的改进。在速度上(Faster)，YOLO v2使用DarkNet19作为特征提取网络，该网络比YOLO v2所使用的VGG-16要更快。在分类上(Stronger)，YOLO v2使用目标分类和检测的联合训练技巧，结合Word Tree等方法，使得YOLO v2的检测种类扩充到了上千种。下图2-2展示了YOLO v2相比于YOLO v1在提高检测精度(Better)上的改进策略。

**【性能】** YOLO v2算法在VOC 2007数据集上的表现为67 FPS时，mAP为76.8，在40FPS时，mAP为78.6。

**【不足】** YOLO v2算法只有一条检测分支，且该网络缺乏对多尺度上下文信息的捕获，所以对于不同尺寸的目标检测效果依然较差，尤其是对于小目标检测问题。

###  **RetinaNet**

**【简介】** 尽管一阶段检测算推理速度快，但精度上与二阶段检测算法相比还是不足。**RetinaNet[13]**论文分析了一阶段网络训练存在的类别不平衡问题，提出能根据Loss大小自动调节权重的Focal loss，代替了标准的交叉熵损失函数，使得模型的训练更专注于困难样本。同时，基于FPN设计了RetinaNet，在精度和速度上都有不俗的表现。

**【性能】** RetinaNet在保持高速推理的同时，拥有与二阶段检测算法相媲美的精度(COCO mAP@.5=59.1%, mAP@[.5, .95]=39.1%)。

### **YOLO v3**

**【简介】** 相比于YOLO v2，**YOLO v3[14]**将特征提取网络换成了DarkNet53，对象分类用Logistic取代了Softmax，并借鉴了FPN思想采用三条分支（三个不同尺度/不同感受野的特征图）去检测具有不同尺寸的对象。

**【性能】** YOLO v3在VOC数据集，Titan X上处理608![[公式]](https://www.zhihu.com/equation?tex=%5Ctimes)608图像速度达到20FPS，在COCO的测试数据集上mAP@0.5达到57.9%。其精度比SSD高一些，比Faster RCNN相比略有逊色(几乎持平)，比RetinaNet差，但速度是SSD、RetinaNet和Faster RCNN至少2倍以上，而简化后的Yolov3 tiny可以更快。

**【不足】** YOLO v3采用MSE作为边框回归损失函数，这使得YOLO v3对目标的定位并不精准，之后出现的IOU，GIOU，DIOU和CIOU等一系列边框回归损失大大改善了YOLO v3对目标的定位精度。

### **YOLO v4**

**【简介】** 相比于YOLO v4，**YOLO v4[15]**在输入端，引入了Mosaic数据增强、cmBN、SAT自对抗训练；在特征提取网络上，YOLO v4将各种新的方式结合起来，包括CSPDarknet53，Mish激活函数，Dropblock；在检测头中，引入了SPP模块，借鉴了FPN+PAN结构；在预测阶段，采用了CIOU作为网络的边界框损失函数，同时将NMS换成了DIOU_NMS等等。总体来说，YOLO v4具有极大的工程意义，将近年来深度学习领域最新研究的tricks都引入到了YOLO v4做验证测试，在YOLO v3的基础上更进一大步。

**【性能】** YOLO v4在COCO数据集上达到了43.5%AP(65.7% AP50)，在Tesla V100显卡上实现了65 fps的实时性能，下图2-3展示了在COCO检测数据集上YOLO v4和其它SOTA检测算法的性能对比。

#### anchor-free

基于Anchor的物体检测问题通常被建模成对一些候选区域进行分类和回归的问题，在一阶段检测器中，这些候选区域就是通过滑窗方式产生Anchor box，而在二阶段检测器中，候选区域是RPN生成的Proposal，但是RPN本身仍然是对滑窗方式产生的Anchor进行分类和回归。基于Anchor的检测算法由于Anchor太多导致计算复杂，及其所带来的大量超参数都会影响模型性能。近年的Anchor free技术则摒弃Anchor，通过确定关键点的方式来完成检测，大大减少了网络超参数的数量。

###  **CornerNet**

**【简介】** **CornerNet[16]**是Anchor free技术路线的开创之作，该网络提出了一种新的对象检测方法，将网络对目标边界框的检测转化为一对关键点的检测(即左上角和右下角)，通过将对象检测为成对的关键点，而无需设计Anchor box作为先验框。

**【性能】** 实验表明，CornerNet在COCO数据集上实现了42.1%AP，该精度优于所有现有的单阶段检测网络。下图2-3展示了在COCO检测数据集上CornerNet和其它SOTA检测算法的性能对比。

**【不足】** CornerNet只关注边缘和角点，缺乏目标内部信息，容易产生FP；该网络还是需要不少的后处理，比如如何选取分数最高的点，同时用offset来微调目标定位，也还需要做NMS。

### **CenterNet**

**【简介】** 与CornerNet检测算法不同，**CenterNet[17]**的结构十分简单，它摒弃了左上角和右下角两关键点的思路，而是直接检测目标的中心点，其它特征如大小，3D位置，方向，甚至姿态可以使用中心点位置的图像特征进行回归，是真正意义上的Anchor free。该算法在精度和召回率上都有很大提高，同时该网络还提出了两个模块：级联角池化模块和中心池化模块，进一步丰富了左上角和右下角收集的信息，并提供了

**【性能】** 相比于一阶段和二阶段检测算法，CenterNet的速度和精度都有不少的提高，在COCO数据集上，CenterNet实现了47.0%的AP，比现有的一阶段检测器至少高出4.9%。下图2-4展示了在COCO检测数据集上CenterNet和其它SOTA检测算法的性能对比。

**【不足】** 在训练过程中，同一类别中的如果某些物体靠的比较近，那么其Ground Truth中心点在下采样时容易挤到一块，导致两个物体GT中心点重叠，使得网络将这两个物体当成一个物体来训练(因为只有一个中心点了)；而在模型预测阶段，如果两个同类物体在下采样后的中心点也重叠了，那么网络也只能检测出一个中心点。

### **FSAF**

**【简介】** **FSAF[18]**网络提出了一种FSAF模块用于训练特征金字塔中的Anchor free分支，让每一个对象都自动选择最合适的特征。在该模块中，Anchor box的大小不再决定选择哪些特征进行预测，使得Anchor的尺寸成为了一种无关变量，实现了模型自动化学习选择特征。

**【性能】** 下图2-5展示了在COCO检测数据集上FSAF算法和其它SOTA检测算法的性能对比。

### **FCOS**

**【简介】** **FCOS[19]**网络是一种基于FCN的逐像素目标检测算法，实现了无锚点(Anchor free)，无提议(Proposal free)的解决方案，并且提出了中心度Center ness的思想。该算法通过去除Anchor，完全避免了Anchor的复杂运算，节省了训练过程中大量的内存占用，将总训练内存占用空间减少了2倍左右。

**【性能】** FCOS的性能优于现有的一阶段检测器，同时FCOS还可用作二阶段检测器Faster RCNN中的RPN，并且很大程度上都要优于RPN。下图2-6展示了在COCO检测数据集上FCOS算法和其它SOTA检测算法的性能对比。

### **SAPD**

**【简介】** **SAPD[20]**论文作者认为Anchor point的方法性能不高主要还是在于训练的不充分，主要是注意力偏差和特征选择。因而作者提出了两种策略：1)Soft-weighted anchor points对不同位置的样本进行权重分配 2)Soft-selected pyramid levels，将样本分配到多个分辨率，并进行权重加权。而在训练阶段，作者前6个epoch采用FSAF的方式，而后6个epoch会将特征选择的预测网络加入进行联合训练。

**【性能】** 下图2-6展示了在COCO检测数据集上SAPD算法和其它SOTA检测算法的性能对比。

### 2.3 TensorRT

TensorRT基本特性和用法

![image-20220509111009590](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509111009590.png)

![image-20220509111046102](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509111046102.png)

![image-20220509111117270](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509111117270.png)

![image-20220509111212591](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509111212591.png)

![image-20220509111324721](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509111324721.png)



## 3. methods

### 3.1 datasets

#### RDD2019,RDD2020

#### VOC2YOLO

#### Data Analyze

#### Data Process

##### train

##### val

### 3.2 model

#### overview

![image-20220509112810823](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509112810823.png)

![image-20220509112923569](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509112923569.png)

#### Backbone

![image-20220509112936327](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509112936327.png)

#### component

##### CBL

![image-20220509112953561](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509112953561.png)

![image-20220509112958259](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509112958259.png)



##### Res_unit

![image-20220509113012338](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113012338.png)

![image-20220509113016438](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113016438.png)



##### C3

![image-20220509113029158](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113029158.png)

![image-20220509113033428](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113033428.png)

##### SPP

![image-20220509113047318](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113047318.png)

![image-20220509113051716](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113051716.png)

##### NECK

![image-20220509113101080](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113101080.png)

![image-20220509113112493](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113112493.png)



##### Detect

![image-20220509113124784](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113124784.png)

![image-20220509113134926](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113134926.png)

![image-20220509113139554](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113139554.png)

### 3.3 optimizer

#### SGD

#### ADAM

### 3.4 loss

#### build_targets

#### Focal_loss

#### IOU_loss

目标检测任务的损失函数一般由Classification Loss（分类损失函数）和Bounding Box Regression Loss（回归损失函数）两部分组成。Bounding Box Regression Loss的发展历程：Smooth L1 Loss -> IOU Loss（2016）-> GIOU Loss（2019）-> DIOU Loss（2020）-> CIOU Loss（2020）

IOU计算方式

![image-20220509113236782](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113236782.png)

![image-20220509113306131](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113306131.png)

![image-20220509113326060](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113326060.png)

![image-20220509113346150](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113346150.png)

这样CIOU_Loss就将目标框回归函数应该考虑三个重要几何因素：重叠面积、中心点距离、长宽比全都考虑进去了。

再来综合看下各个Loss函数的不同点：

IOU_Loss：主要考虑检测框和目标框重叠面积。

GIOU_Loss：在IOU的基础上，解决边界框不重合时的问题。

DIOU_Loss：在IOU和GIOU的基础上，考虑边界框中心点距离的信息。

CIOU_Loss：在DIOU的基础上，考虑边界框宽高比的尺度信息。

#### obj_loss

#### class_loss

### 3.5 embed

#### Docker

Docker 架构: 使用客户端-服务器 (C/S) 架构模式，使用远程API来管理和创建Docker容器。
镜像（Image）:相当于是一个 root 文件系统。比如官方镜像 ubuntu:16.04  ------类
容器（Container）:镜像是静态的定义，容器是镜像运行时的实体。 ------对象   Docker 容器通过 Docker 镜像来创建。
仓库（Repository）:用来保存镜像。

#### pt -> onnx -> engine

![image-20220509111957923](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509111957923.png)

![image-20220509112327803](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509112327803.png)

![image-20220509112408813](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509112408813.png)

![image-20220509112423298](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509112423298.png)



### 3.6 System

#### overview

![image-20220509113642692](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113642692.png)

#### Pyqt5

![image-20220509113717398](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113717398.png)

#### Server

接口1（php）

接收路面损失信息存入数据库

![image-20220509113500757](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113500757.png)

接口2（php）

图片接口

![image-20220509113522401](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113522401.png)

接口3（php）

小程序调用api

![image-20220509113539376](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113539376.png)



#### Database

信息子表

![image-20220509113554410](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113554410.png)

信息主表

![image-20220509113603410](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113603410.png)



#### 小程序

![image-20220509113749589](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220509113749589.png)

## 4. experiments

#### hpy_parameter   Ablation Study

## 5. conclusion