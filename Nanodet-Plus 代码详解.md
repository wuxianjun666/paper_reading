#                           Nanodet-Plus 代码详解

## backbone

1+（1+2+3+4+3+3+1）+1=19层
将（7，14，19）传入下一阶段
s =（1，2，2，2，1，2，1）
（16，24，32，64，96，160，320，1280）
以上backbone：mobilenetv2
输出三个tensor
（24，32，52，52）
（24，96，26，26）
（24，1280，13，13）

## FPN+PAN

然后ghostpan

### 先是reduce_layers：

3个CBL，分别作用于三个尺度的feature，然后都将三个feature的维度变成96

### 紧跟着top-down：

两轮循环，先将深度feature上采样13至26，然后输入到ghostbottleneck，一共两个.每个ghostbottleneck由两个ghostmodule组成，第二个ghostmodule无激活函数，ghost module由两部分组合，一个基础卷积，加上一个深度可分离卷积.经过ghost模块后，输出的feature和backbone出来的中间层feature进行cat，然后再对cat后的feature进行上采样26至52，重复以上操作，输出的feature和backbone输出的低层feature进行cat，这样就形成了新的三个feature.遗漏了一点，在上面第二个ghost module会进行shortcut

### 紧跟着bottle-up：

先对top-down输出的大尺度feature进行下采样52至26，然后输入到ghost模块，输出的feature和top-down输出的中间尺度的feature进行cat，然后将cat后的feature进行下采样26至13，然后输入到ghost模块，然后和top-down输出的小尺度feature进行cat，最终bottle-up也生成了三个尺度的feature.

### 然后有一个extra_layer：

由两部分组成，但各自都是由一个CBL组成，会将尺度降低一半，维度不变，in的部分输入是backbone出来的小尺度feature，out部分输入是bottle-up输出的小尺度部分，尺度都是13*13，然后将两者相加后append到list中，最终ghostpan出来将会有四个尺度的feature
（24 96 52 52）
（24 96 26 26）
（24 96 13 13）
（24 96 7 7）
后面会有一个辅助的aux_fpn，这一完全复制ghostpan，输出的feature同上，此处省略，然后将bottle-up出来的feature和aux-fpn出来的feature进行cat，得到dual_fpn_feat

至此fpn结束

### 接下来就是head

head有两个，一个是nanodetplushead，一个是simpleconvhead，两个head的input分别是fpn_feats和dual_fpn_feats

#### 对于nanodetplushead，

对于不同尺度的feature进行两个CBL，不改变维度和尺度，检测头的大小为5X5，作者说这样有利于扩大感受野，然后各个尺度再进行gfl_cls卷积，只有一个conv，将维度从96变为（num_class+8X4），然后将各个尺度的feature进行cat，得出的输出为（24，3598，34）

#### 对于simpleconvhead，

传入dual_fpn_feat，不同的feature维度为（24，192，n，n），然后各个尺度的feature进行复制两份，一份进行cls_conv卷积（四个CBL，其中归一化用的是组归一化），不改变维度，都是192，然后在进行gfl_cls卷积，改变维度为num_class，输出cls_score，一份进行reg_conv，操作和另一份一样，但是输出维度为8*4=32.输出为bbox_pred，然后将cls_score和bbox_pred进行cat，最后将各个尺度的feature进行cat，最终输出是（24，3598，34）

### 接下来进入loss：

第一步get grid cells of one images
第二步将nanodetplushead的输出split为cls_preds和reg_preds，维度分别为（24，3598，num_class）和（24，3598，32）

然后将reg_preds（24，3598，32）送入distribution_project函数，该函数：先将输入reshape成（24，3598，4，8），然后对最后一维进行softmax，（最后一维八个数相加为1），得到x：（24，3598，4，8），然后将x与[0 1 2 3 4 5 6 7]进行linear(),输出为dis_preds（24，3598，4）.
然后将center_priors的第三维的前两列（作为point）（24，3598，2）和上面通过softmax和linear的dis_preds（24，3598，4）（作为distance）传入distance2bbox函数计算，得到decoded_bboxes（24，3598，4）
point为中心点，四个distance代表该点到四个边界的距离

![image-20220516103645918](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220516103645918.png)

同理aux_preds也split成aux_cls_pred和aux_reg_pred，维度分别为（24，3598，num_class）和（24，3598，32）
同上计算出aux_dis_preds（24，3598，4）
同上计算出aux_decoded_bbox（24，3598，4）：xxyy

将辅助网络得出的输出输入到**target_assign_single_image函数**，输入分别为aux_cls_preds（num_priors，num_classes），center_priors（num_priors，4），decoded_bboxes（num_priors，4），gt_bboxes（num_gts，4），gt_labels（num_gts）.

生成bbox_targets（num_priors，4）：全是0，

同理也生成dist_targets，（num_priors，4）：全是0，

还有一个labels（num_priors，）：值全是num_class，

和label_scores（num_priors，），值全是0.

之后将传入target_assign_single_imag的参数传入**DSLA类下的函数assign**，

首先将assigned_gt_inds赋值为0，维度为（num_priors，），

计算每个网格点到真实框的左上角的点的xy距离：**lt_（num_priors，numgts，2）**，

同理计算每个网格点到真实框的右下角的xy距离：**rb_（num_priors，num_gts，2）**.

然后将lt_和rb在最后一个维度进行cat，得到deltas（num_priors，num_gts，4），**该变量记录了真实框对角两个点到各个网格点的距离**.

然后is_in_gts（num_priors，num_gts）=torch.min（deltas，dim=-1）.values>0:**该操作是为了找出位于真实框内的网格点.valid_mask（num_priors，）=is_in_gts.sum（dim=1）>0,**.

之后根据valid_mask作为来取出preds_score和decoded_bbox对应网格valid_mask为true的点，就是处于真实框中的网格点的预测的目标类别**valid_preds_score**（此次维度为（num_valid，num_class））和目标框信息**valid_decoded_bbox**（此次维度维（num_valid，4））.

用筛选出来的框和真实框做交并比计算，over_bboxoverlaps得出pairwise_ious（num_valid，num_gts），然后计算iou_loss（num_valid，num_gts）=-torch.log（pairwise_ious）.

然后将真实标签用F.onehot（）变成one_hot，并且repeat（num_valid，1，1）得到gt_one_hot（num_valid，num_gts，num_classes），

同时将valid_preds_score也进行repeat（1，num_gt，1）成（num_valid，num_gts，num_classes），

然后用get_one_hot（num_valid，num_gts，num_classes）和pairwise_iou（num_valid，num_gts，o1）进行相乘，得到soft_label（num_valid，num_gts，num_classes），

然后用soft_label减去valid_preds_score得到scale_fctor（num_valid，num_gts，num_classes）。

然后用F.binary_cross_entropy（valid_preds_score，soft_label）** scale_fctor.abs（）.pow（2）进行计算cls_cost（num_valid，num_gts，num_classes），然后对cls_cost.sum（dim=-1）（num_valid，num_gts），然后就得到了cost_matrix （num_valid，num_gts）= cls_loss+iou_loss*iou_factor.

然后将cost_matrix，pairwise_ious，num_gts，valid_mask传入**函数dynamic_k_matching**，

初始化一个和cost_matrix一样维度的matching_matrix（num_valid，num_gts），值全为0。

然后用torch.topk根据pairwise_ious筛选出对应于每个gt前candidate_topk个网格点topk_ious（candidate_topk，num_gts）

然后对这些网格点的ious进行sum（0），得到**dynamic_ks（num_gts，1）**，

然后又用torch.topk取出每个目标对应的cost_matrix的前dynamic_ks个所对应的索引pos_inds，

根据这个pos_inds将**matching_matrix**中对应的位置赋值为1。然后筛选出认为存在多个正样本的网格点，**prior_match_gt_mask（num_valid，1）**（存在多个正样本的位置值为True）=matching_matrix.sum（1）>1，如果存在多个正样本的网格点，根据prior_match_gt_mask，去筛选cost_matrix，此时筛选出的维度为（dynamic_ks，2），然后对其进行cost_min（值），cost_argmin（索引）=torch.min（dim=1）选出损失值最小的网格点，，后面这两步还不懂.

然后跳出判断，找出存在正样本的网格点**fg_mask_inboxes（num_valid，）**，赋值为True，然后将存了所有网格点的valid_mask中存在正样本的点赋值为True.然后用fg_mask_inboxes筛选出matching_matrix中存在正样本的点维度（sum（dynamic_ks），2）.argmax（1），最终得到**matched_gt_inds（sum（dynamic_ks），1）**，然后将存在num_valid网格点*pairwise_ious.然后取出存在正样本的网格点的ious，**matched_pred_ious（sum（dynamic_ks），1）**。

**解释：**dynamic_k_matching这个函数其实就是先传入在真实框内部的网格点的cost_matrix：与真实框的的类别和交并比损失之和，以及pairwise_ious交并比，然后根据交并比选取topk个网格点，然后将这些网格点的交并比求和，得到一个数值dynamic_ks，用这个数值去cost_matrix中取出dynamic_ks个网格点.认为这些网格点中存在正样本，最终返回这些网格点的索引和iou

将存在正样本的网格点记录下来，首先是assigned_gt_inds，原本所有的网格点（3598）都为零，现在将存在正样本的网格点赋成其他值，与框的数量有关.然后是assigned_labels，原本所有的网格点（3598）都为-1，然后现在将存在正样本的网格点赋值成他所属单位类别的id，最后说max_overlaps，原本所有的网格点都为-INF，然后现在将存在正样本的网格点赋值成他所对应的iou值.然后把这些值传入**AssignResult类.**

### 以上就是动态软标签匹配

然后将assign_result传入**sample函数：**

然后将assign_result（num_gt，assigned_gt_inds，max_overlap，assigned_labels）传入sample函数：然后取出存在正样本的网格点pos_inds，再取出负样本的网格点neg_inds，

后面这个不是很懂：pos_assigned_gt_inds=assign_result.get_inds[pos_inds]-1

然后将assign_result（num_gt，assigned_gt_inds，max_overlap，assigned_labels）和gt_bbox（num_gt，4）传入sample函数：然后取出存在正样本的网格点pos_inds，再取出负样本的网格点neg_inds，

后面这个不是很懂：pos_assigned_gt_inds=assign_result.get_inds[pos_inds]-1，

最后再来一个pos_gt_bboxes=gt_bboxes[pos_assigned_gt_inds,:]网格点对应于哪一个框（当有多个框时），

最终该函数返回pos_inds正样本网格点，neg_inds，负样本网格点，pos_gt_bboxes，各个网格点对应的框，pos_assigned_gt_inds各个网格点对应于哪个框。pos_ious：正样本网格点对应的iou

### 最终在函数target_assign_single_image函数中：

将return这么几个变量：bbox_targets，除了筛选出来认为存在正样本的网格点赋值为真实框的信息，其余全为零.维度（3598，4），distance_target，除了筛选出来认为存在正样本的网格点赋值为真实框的各条边到网格点的距离外，其余全为0.维度为（3598，4） 接下来是labels：除了筛选出来认为存在正样本的网格点赋值为真实类别的id外，其余全为num_class，维度为（3598，），最后一个是label_scores：除了筛选出来认为存在正样本的网格点赋值为pos_ious外，其余全为零，该函数最终返回一个元组（labels，label_scores，bbox_targets，dist_targets，num_pos_per_img） batch_assign_res

### 接下来就是计算损失的函数：_get_loss_from_assign（cls_preds，reg_preds，decoded_bboxes，assign）：

首先将assign拆解开来，然后计算num_total_samples，总的正样本数，然后将下面这些变量reshape，labels:(24X3598,),label_scores:(24X3598,),bbox_targets:(24X3598,4),cls_preds:(24X3598,num_classes),reg_preds(24X3598,32),decoded_bboxes:(24X3598,4) 

### 接下来就是quality_focalloss：传入：preds=cls_preds，targets=（labels，label_scores），avg_factor=num_total_samples）

![image-20220516110516559](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220516110516559.png)

首先label，score=targets，然后对cls_preds进行sigmoid（），得到pred_sigmoid.然后计算损失时，分成两类，当y=0，即为负样本，表达式中为y=0.当正样本时，筛选出正样本的索引pos，然后计算scale_factor=score[pos]-pred_sigmoid[pos,pos_label],然后对应于网格点的loss值如下图计算，最后将loss=loss.sum（dim=1）维度为（24*3598，0）

![image-20220516110407676](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220516110407676.png)

然后是返回总loss函数，pos_inds是存在正样本的网格点.然后用pos_inds筛选出cls_preds的网格点，因为是五维的，所以选出五个中最大的，然后对这筛选出的值进行求和，得出bbox_avg_factor，一个浮点数.然后将decoded_bboxes[pos_inds],bbox_targets[pos_inds]进行giou计算，avg_factor为上面的bbox_avg_factor

### 然后计算distribution_focalloss，

![image-20220516110539073](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220516110539073.png)

传入的参数是:

reg_preds[pos_inds].reshape(-1,8),

dist_target[pos_inds].reshape(-1),

weight=weight_targets,

avg_factor=bbox_avg_factor*4

![image-20220516110628476](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220516110628476.png)

