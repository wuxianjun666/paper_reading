**Abstract:** Road pavement could be damaged due to various reasons, causing damages such as cracks and pits. These damages cause potential dangers in traffic safety. It is necessary for road maintenance departments to find damages in time before maintenance. At present, maintenance departments of some high-level roads are equipped with specialized detection vehicles such as laser scanning vehicles to detect road damages. These kinds of devices can get good detection performance, but the economic cost is very high. In this paper, we use a road damage image dataset to train an deep convolutional neural network and deploy it on a low-cost object detection model based on embedded platform to form an embedded system. The system uses a common camera mounted on windshield of a common vehicle as sensor to detect road damages. The embedded system consumes about 352 ms to process one frame of image and can achieve a recall rate of about 76% which is higher than some previous related works. The recall rate of this scheme using common cameras is less than that of high-level specialized detectors, but the economic cost is much lower than them. After subsequent development, the road maintenance department with limited funds can consider about schemes like this. 

**Keywords****:** road damage detection; road survey; embedded system; object detection; convolutional neural network; deep learning

 

With continuous growth of economy, transportation industry is also developing at high speed and road mileage in the world is increasing. While total length of road is increasing, old roads are slowly becoming damaged over time and road maintenance departments in various areas are facing increasing maintenance demands. Finding damages of road surface is a necessary step before road maintenance department perform repair work on road surface. Traditionally, road damage detection is mainly through manual search, which is extremely time-consuming and labor-intensive. A scheme that can easily find road damage is urgently needed to be developed. In fact, there have some scholars [1] researched this problem previously. At present, automatic detection of road damage mainly through three methods of laser, radar and vision. In 2006, Hinton et al. proposed the concept of deep learning [2]. Since then, technology based on deep learning have begun to develop rapidly. Research and product development of computer vision based on deep neural network are rapidly emerging. Using image processing technology and deep neural network, a new round of exploration and practice has been carried out by scholars [3,4] on scheme of road damage detection. [5], MobileNet object classification method proposed by A. G. Howard et al. [6], road damage image dataset proposed by H. Maeda et al. [4]. Main processor of the embedded system is Rockchip RK3399Pro SoC which integrates a separate neural processing unit (abbr. NPU). It runs an object detection model that can detect road damages through common camera. When road damage is detected, the system can automatically acquire geographic location information through GPS module and save related data in memory including image and location information. The system has been tested to achieve a useful detection recall rate with low economic cost. In the following section 2, this paper will introduce some related work, including various schemes of road damage detection proposed by previous works and some current object detection methods based on deep convolutional neural network. In section 3, we will from top to bottom introduce overall architecture, detection process and core object detection method of our embedded system. In section 4, this paper will present performance test results of the system we designed and compare it with a previous work.



 **Related** **works**

*2.1.* *Road damage detection*

Maintenance departments of highways and other high-level roads usually have sufficient funds to use expensive specialty devices to detect road damages. These expensive specialty devices usually can achieve very good accuracy [1,7]. Scholars began trying to use various schemes to detect road damages many years ago. After a long period of development, some products for road damage detection have come out, such as JG-1 laser 3-D road condition intelligent detection system [7], ZOYON-RTM road detection vehicle system [1], road condition rapid detection system based on line sweep technology CiCS [1] and so on. These devices typically can achieve an accuracy of over 95% and are widely used in practice.

However, such only maintenance departments of high-level roads can afford to choose them. On common roads, most road maintenance departments still use method of manual searching which greatly slows down the efficiency of road maintenance work. In recent years, object detection technology based on deep convolutional neural network has developed rapidly and related technologies have gradually landed in various application scenarios. In 2016, L. Zhang et al. used a low-cost smartphone to take 500 photos of road cracks on campus of Temple University and trained a simple 10-layer convolutional neural network model to determine whether there were road cracks in a photo. This model can achieve about 92.51% recall rate for recognition of the images in test set. However, because of the small size of their train set, generalization ability of their model is poor. Recognition performance of images from test set and other images is quite different and it cannot be applied in real world. However, it is undeniable that this is the first proposed method to detect road cracks by deep neural network [3]. In 2018, H. Maeda et al. took 9,053 photos containing 15,435 road damages in seven cities of Japan [4] and trained two models based on SSD-MobileNet or SSD-Inception. They packaged one model into an Android app runs on a common Android smartphone. Images are captured by smartphone camera for road damage detection, their device can achieve a recall rate of about 50% and a precision rate of about 75% on weighted average. Our system retrained a new model using  dataset shared by them on GitHub and further optimized it. 

*2.2.* *Object detection*

![image-20220420234349090](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220420234349090.png)

Road damage detection through vision is a problem of object detection. Object detection is a basic problem in the field of computer vision. As an example shown in Figure 1, input of object detection is a frame of image; output is all existing targets in the image and each target is given class information and position information. In many real-world scenarios including road damage detection, object detection is often the first step in visual perception, which has been researched by many institutions apply on our system. and scholars [5,8–11]. We have several following algorithms to select to

2.2.1.   Faster R-CNN: Faster regions with CNN feature

R-CNN is an object detection algorithm proposed by R. Girshick et al. [8] in 2013. This algorithm can be said to be the first pioneering work of object detection using deep learning. After development of R-CNN and Fast R-CNN, S. Ren et al. proposed Faster R-CNN [9] in NIPS2015. Faster R-CNN unifies four basic steps of object detection (proposal generation, feature extraction, region classification, region refinement) into one same deep neural network framework, to reduce repetitive computation and greatly improve running speed compared with previous generations. However, compared with other common networks, Faster R-CNN is still running more slowly and not suitable for running on mobile poor-performance embedded platforms.

2.2.2.   YOLO v3: You only look once v3

YOLO is an object detection algorithm proposed by J. Redmon et al. [10] in CVPR2016. Core idea of YOLO is to transform object detection into a regression problem to solve and based on a single end-to-end network to complete the object location and category output from the original image input. YOLO v3 [11] is the third version of YOLO, which has a fast running speed but is less sensitive to small object detection such as road cracks.

2.2.3.   SSD: Single shot multiBox detector

SSD is an object detection algorithm proposed by W. Liu et al. [5] in ECCV2016. It has obvious speed advantage compared with Faster R-CNN and obvious accuracy advantage compared with YOLO.

Comprehensively evaluated strengths and weaknesses of each model, combined with the effectiveness of training and detection of road damage dataset, we select to use SSD as our object detection network.

**Proposed** **method**

*3.1.* *System* *architecture*

L. Zhang et al. and H. Maeda et al. have tried to use a common Android phone for road damage detection [3,4]. After our tests, computing and imaging performance of smartphones are not so satisfying with road damage detection. Considering that more suitable computing chips and cameras can be selected, we decided to design a dedicated embedded system. The embedded system architecture we proposed is as follows.

Main processor of this embedded system is Rockchip RK3399Pro SoC, which is connected to a High-Dynamic-Range (abbr. HDR) camera for capturing road images and a GPS module for acquiring geographical location. It is also connected to peripherals such as touch screen to facilitate user operation. We select RK3399Pro SoC as main processor cause the chip integrates a separate NPU which can accelerate computation of deep neural network. The NPU has computing performance of 2.4 TOPs.

We select an HDR camera to capture road image. Compared with ordinary smartphone camera, it performs better in backlight condition and can clearly capture road surface image when it is directly exposed to sunlight. Using HDR camera allows the system can be used in more illumination condition. Overall architecture of our embedded system is shown in Figure 2 and a usage example is shown in Figure 3.

![image-20220420234627371](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220420234627371.png)

![image-20220420234637982](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220420234637982.png)

We convert the trained object detection model into a “.rknn” file that can be accelerated using Rockchip NPU, then use OpenCV, NumPy, pySerial, PyQt5 and other Python packages to write programs for reading images, post-processing, reading etc.. That constitutes the entire software system. GPS data, Graphic User Interface

*3.2.* *Overall detection process*

Mount the camera on front windshield of car, align the road ahead with a suitable angle and run the software to start the detection. System will continuously capture images of road ahead and input images into object detection model. After computing of the model and related post-processing, it can be determined whether a frame of image contains road damages. If an image is considered to contain road damages, system will automatically acquire geographic location information through GPS module and save related data in memory including image and location information for manual query and processing in later stage.

![image-20220420234758088](C:\Users\72758\AppData\Roaming\Typora\typora-user-images\image-20220420234758088.png)

*3.3.* *Core object detection model*

Single Shot MultiBox Detector (SSD) [5] is a one-stage detector for multiple categories, which is slightly less accurate than two-stage network (such as Faster R-CNN [9], etc.). However, two-stage network needs to process image to generate a series of proposals and then classify proposals by convolutional neural network, while one-stage network directly convert problem ofobject border locating into regression problem. Therefore, one-stage network such as SSD usually runs faster than two-stage network.

In the demo code given in original paper of SSD, VGG [12] is used as the backbone



7988

 

network for



feature extraction. In real-world scenario applications, when SSD is used for object detection on



mobile platforms such as smartphones or embedded systems, MobileNet [6] is usually used to



replace VGG network to achieve faster computing speed. MobileNet is a convolutional neural



network proposed by A. G. Howard et al. particularly designed for mobile platforms. It uses



depth-wise separable convolution to replace traditional convolution, in order to reduce network



weight parameters, which makes model more lightweight and can achieve between efficiency and accuracy.



a reasonable balance



Considered about computing performance of the embedded platform and actual application



scenario, we select SSD whose backbone is replaced to MobileNet, namely SSD-MobileNet, for road



damage detection. We used RoadDamageDataset [4] to train an SSD-MobileNet-based object



detection model using TensorFlow deep learning framework on platform of NVIDIA GeForce GTX



1060 GPU. Input of the model is a matrix of 300 × 300 × 3 (i.e., an RGB image of 300 × 300 pixels)



and output the image.



is a matrix



contains information



of position



and confidence of road damages



existing in



The algorithm flow from camera capturing image

in Figures 5 and 6.

 





 

 