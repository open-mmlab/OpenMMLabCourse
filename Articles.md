- [1. 目标检测 & MMDetection](#1-目标检测--mmdetection)
- [2. 3D目标检测 & MMDetection3D](#2-3d目标检测--mmdetection3d)
- [3. 图像分类 & MMClassification](#3-图像分类--mmclassification)
- [4. 语义分割 & MMSegmentation](#4-语义分割--mmsegmentation)
- [5. 模型部署 & MMdeploy](#5-模型部署--mmdeploy)
- [6. 姿态估计 & MMPose](#6-姿态估计--mmpose)
- [7. 视频感知 & MMTracking](#7-视频感知--mmtracking)
- [8. 光流估计 & MMFlow](#8-光流估计--mmflow)
- [9. 计算机视觉基础库 & MMCV](#9-计算机视觉基础库--mmcv)
- [10. 自监督学习 & MMSelfSup](#10-自监督学习--mmselfsup)
- [11. 视频和图像编辑 & MMEditing](#11-视频和图像编辑--mmediting)
- [12. 旋转框检测 & MMRotate](#12-旋转框检测--mmrotate)
- [13. 模型压缩 & MMRazor](#13-模型压缩--mmrazor)
- [14. 人体参数化模型 & MMHuman3D](#14-人体参数化模型--mmhuman3d)
- [15. 少样本学习 & MMFewShot](#15-少样本学习--mmfewshot)
- [16. 行为理解 & MMAction2](#16-行为理解--mmaction2)
- [17. 文本检测识别理解 & MMOCR](#17-文本检测识别理解--mmocr)
- [18. 生成模型 & MMGeneration](#18-生成模型--mmgeneration)
- [19. Pytorch & General](#19-pytorch--general)







## 1. 目标检测 & MMDetection

<!--- [<img src="https://github.com/open-mmlab/mmdetection/blob/master/resources/mmdet-logo.png" height="36">](https://github.com/open-mmlab/mmdetection) --->


- \[2021/08/11\] [# YOLOX 在 MMDetection 中复现全流程解析](https://zhuanlan.zhihu.com/p/398545304)
- \[2021/08/23\] [# 喂喂喂！你可以减重了！小模型 MMDetection 新增SSDLite 、 MobileNetV2YOLOV3 两大经典算法](https://zhuanlan.zhihu.com/p/402781143)
- \[2021/09/01\] [# OpenMMLab 社区专访之 YOLOX 复现篇 ](https://zhuanlan.zhihu.com/p/405913343)
- \[2021/11/18\] [# K-Net: Kernel is All YOU Need for Image Segmentation?（迈向统一的图像分割）](https://zhuanlan.zhihu.com/p/436639174)
- \[2021/12/09\] [# 小白都能看懂！手把手教你使用混淆矩阵分析目标检测](https://zhuanlan.zhihu.com/p/443499860)
- \[2022/01/24\] [# 轻松掌握 MMDetection 整体构建流程(一)](https://zhuanlan.zhihu.com/p/337375549)
- \[2022/01/25\] [# 轻松掌握 MMDetection 整体构建流程(二)](https://zhuanlan.zhihu.com/p/341954021)
- \[2022/02/28\] [# 是时候该学会 MMDetection 进阶之非典型操作技能（一）](https://zhuanlan.zhihu.com/p/473707171)
- \[2022/04/07\] [# ResNet 高精度预训练模型在 MMDetection 中的最佳实践](https://zhuanlan.zhihu.com/p/494609932)
- \[2022/04/29\] [# OpenMMLab 上海交大精品课带你 4 小时入门深度学习](https://zhuanlan.zhihu.com/p/507386830)
- \[2022/05/18\] [# 超 10 个点的提升！ Open Images 在 MMDetection 的实现](https://zhuanlan.zhihu.com/p/516419148)
- \[2022/05/31\] [# CVPR2022 Group R-CNN : 化框为点，简化物体检测数据标注](https://zhuanlan.zhihu.com/p/522683049)
- \[2022/06/22\] [# MaskFormer 在 MMDtection 中复现全流程解析](https://zhuanlan.zhihu.com/p/532168933)
- \[2021/11/23\] [# MMDet居然能用MMCls的Backbone？论配置文件的打开方式](https://zhuanlan.zhihu.com/p/436865195)
- \[2022/05/25\] [# 目标检测的首选深度框架？](https://www.zhihu.com/answer/2500571323)
- \[2022/03/29\] [# 计算机视觉中，目前有哪些经典的目标跟踪算法？](https://www.zhihu.com/answer/2412612945)
- \[2021/10/29\] [# Mmdetection中SOTA论文源码中将训练过程中BN层的eval打开?](https://www.zhihu.com/answer/2195540892)
- \[2021/09/22\] [# 你是如何自学 Python 的？](https://www.zhihu.com/answer/2134322010)
- \[2021/05/31\] [# COCO数据集上1x模式下为什么不采用多尺度训练?](https://www.zhihu.com/answer/1915119662)
- \[2021/04/26\] [# 想知道目标检测领域中还有哪些方向能做？](https://www.zhihu.com/answer/1855223790)
- \[2021/04/23\] [# 深度学习小白，毕业设计想做深度学习跟踪目标方面的，有什么建议？](https://www.zhihu.com/answer/1850035178)
- \[2021/04/22\] [# 如何具体上手实现目标检测呢？](https://www.zhihu.com/answer/1848561187)
- \[2021/04/17\] [# 基于PyTorch的MMDetection中训练的随机性来自何处？](https://www.zhihu.com/answer/1839683634)
- \[2021/04/13\] [# MMDetection如何学习源码？](https://www.zhihu.com/answer/1832498963)
- \[2021/03/12\] [# 想要涉足目标检测领域，有推荐的系统学习路线吗？](https://www.zhihu.com/answer/1776343553)
- \[2021/02/02\] [# mmdetection如何解决安装mmcv遇到的问题？](https://www.zhihu.com/answer/1710754148)
- \[2020/12/22\] [# 如何看待商汤的Deformable DETR？能否取代Faster-RCNN范式？](https://www.zhihu.com/answer/1640597255)
- \[2020/12/15\] [# 目标检测领域还有什么可以做的？](https://www.zhihu.com/answer/1627885518)
- \[2020/12/10\] [# 单阶段、双阶段、anchor-based、anchor-free这四者之间有什么联系吗？](https://www.zhihu.com/answer/1619925296)
- \[2020/12/05\] [# 目标检测的深度学习方法，有推荐的书籍或资料吗？](https://www.zhihu.com/answer/1612593817)
- \[2020/12/05\] [# 大佬们，刚入学研究生，想入门目标检测，有什么学习路线可以入门的？](https://www.zhihu.com/answer/1612580715)
- \[2022/05/25\] [# 目标检测的首选深度框架？](https://www.zhihu.com/answer/2500571323)







## 2. 3D目标检测 & MMDetection3D

<!--- [<img src="https://github.com/open-mmlab/mmdetection3d/blob/master/resources/mmdet3d-logo.png" height="36">](https://github.com/open-mmlab/mmdetection3d) --->


- \[2021/08/19\] [# "3Dfy" A General 2D Detector: 纯视觉 3D 检测再思考](https://zhuanlan.zhihu.com/p/400191167)
- \[2021/09/22\] [# 点云语义分割，现已加入 MMDet3D 全家桶！](https://zhuanlan.zhihu.com/p/402839129)
- \[2021/11/17\] [# 单目 3D 检测最新进展调研与思考](https://zhuanlan.zhihu.com/p/435139846)
- \[2021/12/12\] [# 概率和几何深度：在三维空间中检测物体](https://zhuanlan.zhihu.com/p/442753563)
- \[2022/03/09\] [# 带你玩转 3D 检测和分割（一）：MMDetection3D 整体框架介绍](https://zhuanlan.zhihu.com/p/478307528)
- \[2022/04/01\] [# 带你玩转 3D 检测和分割 （二）：核心组件分析之坐标系和 Box](https://zhuanlan.zhihu.com/p/491614921)
- \[2022/04/25\] [# 带你玩转 3D 检测和分割 （三）：有趣的可视化](https://zhuanlan.zhihu.com/p/504862433)
- \[2022/05/23\] [# 【预告】社区开放麦第 6 期：基于视觉的车速估计技术](https://zhuanlan.zhihu.com/p/518772393)
- \[2022/06/24\] [# 厉害了！有了它，发顶会顶刊拿赛事大奖轻松多了！](https://zhuanlan.zhihu.com/p/533267898)
- \[2022/07/01\] [# 【3D 目标检测模型部署】全链条打通！PointPillars 从模型到部署](https://zhuanlan.zhihu.com/p/536323578)
- \[2021/08/25\] [# 做 Transformer, OpenMMLab 了解一下？](https://zhuanlan.zhihu.com/p/403661977)
- \[2022/03/08\] [# 如何入门激光雷达点云的3D目标检测？](https://www.zhihu.com/answer/2379324138)
- \[2021/11/04\] [# OpenPCDet和mmdetection3d有什么区别?](https://www.zhihu.com/answer/2206147084)







## 3. 图像分类 & MMClassification

<!--- [<img src="https://github.com/open-mmlab/mmclassification/blob/master/resources/mmcls-logo.png" height="36">](https://github.com/open-mmlab/mmclassification) --->


- \[2021/10/20\] [# MMClassificiation 实现数据增强的 N 种方法](https://zhuanlan.zhihu.com/p/424133612)
- \[2021/11/19\] [# MMClassification 数据增强介绍（二）](https://zhuanlan.zhihu.com/p/436238223)
- \[2022/01/04\] [# 类别激活热力图可视化工具介绍](https://zhuanlan.zhihu.com/p/453182477)
- \[2022/01/21\] [# Vision Transformer 必读系列之图像分类综述(一)：概述](https://zhuanlan.zhihu.com/p/459828118)
- \[2022/01/26\] [# Vision Transformer 必读系列之图像分类综述(二): Attention-based](https://zhuanlan.zhihu.com/p/461700507)
- \[2022/01/27\] [# Vision Transformer 必读系列之图像分类综述(三): MLP、ConvMixer 和架构分析](https://zhuanlan.zhihu.com/p/462463183)
- \[2022/03/18\] [# 以动制动 Transformer 如何处理动态输入尺寸](https://zhuanlan.zhihu.com/p/483309470)
- \[2022/04/12\] [# 用 OpenMMLab 轻松搭建主干网络，多种视觉任务一网打尽](https://zhuanlan.zhihu.com/p/497363694)
- \[2022/07/15\] [# OpenMMLab 进阶指南，模型训练测试全流程解析](https://zhuanlan.zhihu.com/p/541934131)
- \[2022/03/21\] [# 什么是图像分类的Top-5错误率？](https://www.zhihu.com/answer/2400009714)
- \[2022/03/10\] [# 图像分类中的max pooling和average pooling是对特征的什么来操作的，结果是什么？](https://www.zhihu.com/answer/2382478754)







## 4. 语义分割 & MMSegmentation

<!--- [<img src="https://github.com/open-mmlab/mmsegmentation/blob/master/resources/mmseg-logo.png" height="36">](https://github.com/open-mmlab/mmsegmentation) --->


- \[2022/05/26\] [# 超详细！带你轻松掌握 MMSegmentation 整体构建流程](https://zhuanlan.zhihu.com/p/520397255)
- \[2022/06/07\] [# 超详细！手把手带你轻松用 MMSegmentation 跑语义分割数据集](https://zhuanlan.zhihu.com/p/525422379)
- \[2021/09/17\] [# 解读 OpenMMLab 的 Hook 机制](https://zhuanlan.zhihu.com/p/387483425)
- \[2022/06/17\] [# 语义分割该如何走下去？](https://www.zhihu.com/answer/2532479123)
- \[2021/10/25\] [# 研究生图像分割怎么学习？](https://www.zhihu.com/answer/2188347132)
- \[2021/09/14\] [# 大佬们，我刚开始接触图像分割，对于图像分割深度学习这块一头雾水，可以给出一个从零开始学习的路线吗？](https://www.zhihu.com/answer/2120515790)
- \[2021/06/01\] [# 医学图像分割请问要分割出目标，想先确定目标区域，然后在区域中提取目标应该怎么预处理图片呢？](https://www.zhihu.com/answer/1916793153)
- \[2021/04/29\] [# 刚上研一，方向是医学影像处理，课题是关于分割的，但是毫无头绪，求问怎么学习图像分割？](https://www.zhihu.com/answer/1860816906)
- \[2021/04/18\] [# 研一学生，准备做有关医学图像分割的内容，想请教一下大家，创新点都有从哪些方面研究，谢谢大家了！?](https://www.zhihu.com/answer/1841253967)
- \[2020/12/29\] [# 为啥U-Net训练数据得到的是全黑的图？](https://www.zhihu.com/answer/1650919034)
- \[2020/12/14\] [# 图像语义分割如何下手？算法如何实现？](https://www.zhihu.com/answer/1625973937)
- \[2020/12/13\] [# 图像处理方向打算做分割，导师让着手实验，从最简单的单层网络开始入手，有没有师兄师姐建议怎么开始呀？](https://www.zhihu.com/answer/1624600717)
- \[2020/12/11\] [# 有关语义分割的奇技淫巧有哪些？](https://www.zhihu.com/answer/1621508656)
- \[2022/06/17\] [# 语义分割该如何走下去？](https://www.zhihu.com/answer/2532479123)







## 5. 模型部署 & MMdeploy

<!--- [<img src="https://github.com/open-mmlab/mmdeploy/blob/master/resources/mmdeploy-logo.png" height="36">](https://github.com/open-mmlab/mmdeploy) --->


- \[2021/12/27\] [# 千行百业智能化落地，MMDeploy 助你一“部”到位](https://zhuanlan.zhihu.com/p/450342651)
- \[2022/03/08\] [# 模型部署入门教程（一）：模型部署简介](https://zhuanlan.zhihu.com/p/477743341)
- \[2022/03/11\] [# 模型部署入门教程（二）：解决模型部署中的难题](https://zhuanlan.zhihu.com/p/479290520)
- \[2022/03/21\] [# 手把手教你在 ubuntu 上使用 MMDeploy](https://zhuanlan.zhihu.com/p/484842986)
- \[2022/04/02\] [# 想要模型部署玩得好，这些我们要知道：MMDeploy 进展一览](https://zhuanlan.zhihu.com/p/492090146)
- \[2022/05/12\] [# 模型部署入门教程（四）：在 PyTorch 中支持更多 ONNX 算子](https://zhuanlan.zhihu.com/p/513387413)
- \[2022/05/19\] [# 模型部署入门教程（五）：ONNX 模型的修改与调试](https://zhuanlan.zhihu.com/p/516920606)
- \[2022/06/17\] [# TorchScript 解读（四）：Torch jit 中的别名分析](https://zhuanlan.zhihu.com/p/530242380)
- \[2022/07/13\] [# 神奇的 StyleGAN，用 18 支画笔作画的它究竟有多强](https://zhuanlan.zhihu.com/p/541196270)
- \[2022/07/20\] [# 模型部署入门教程（六）：实现 PyTorch-ONNX 精度对齐工具](https://zhuanlan.zhihu.com/p/543973749)
- \[2022/03/24\] [# TorchScript 解读（一）：初识 TorchScript](https://zhuanlan.zhihu.com/p/486914187)
- \[2022/03/28\] [# TorchScript 解读（二）：Torch jit tracer 实现解析](https://zhuanlan.zhihu.com/p/489090393)
- \[2022/04/06\] [# TorchScript 解读（三）：jit 中的 subgraph rewriter](https://zhuanlan.zhihu.com/p/493955209)
- \[2022/04/14\] [# 模型部署入门教程（三）：PyTorch 转 ONNX 详解](https://zhuanlan.zhihu.com/p/498425043)
- \[2022/03/30\] [# 视觉算法的工业部署及落地方面的技术知识，怎么学？](https://www.zhihu.com/answer/2414724618)
- \[2022/03/22\] [# 如何评价框架共用的模型文件格式ONNX？](https://www.zhihu.com/answer/2401790053)
- \[2022/03/18\] [# 如何选择深度学习推理框架？](https://www.zhihu.com/answer/2395418101)
- \[2022/03/17\] [# 如何用 C++ 部署深度学习模型？](https://www.zhihu.com/answer/2393173576)







## 6. 姿态估计 & MMPose

<!--- [<img src="https://github.com/open-mmlab/mmpose/blob/master/resources/mmpose-logo.png" height="36">](https://github.com/open-mmlab/mmpose) --->


- \[2021/09/06\] [# 自顶向下的 2D 人体姿态估计](https://zhuanlan.zhihu.com/p/394060630)
- \[2021/09/14\] [# 来咯来咯！AI 黑玉断续膏：自底向上的二维人体姿态估计](https://zhuanlan.zhihu.com/p/410284435)
- \[2022/01/18\] [# 3D 人体姿态估计简述](https://zhuanlan.zhihu.com/p/400922771)
- \[2022/02/11\] [# 一户一墩？墩墩生成器安排了！](https://zhuanlan.zhihu.com/p/466281786)
- \[2022/02/14\] [# 抓住情人节的尾巴，和 Ta 炫一手独家高级操作！](https://zhuanlan.zhihu.com/p/467408110)
- \[2022/03/22\] [# MMPose 初体验：推理、导出 ONNX、转 MNN](https://zhuanlan.zhihu.com/p/485549154)
- \[2022/05/30\] [# 【预告】社区开放麦第 7 期：MMPose 姿态估计创意大赛技术指南](https://zhuanlan.zhihu.com/p/522183234)
- \[2022/06/15\] [# 特效大片背后的多视角 3D 人体姿态估计技术](https://zhuanlan.zhihu.com/p/529219789)
- \[2022/04/24\] [# 【回放】 社区开放麦第 2 期：学习 CVPR 前沿姿态估计论文](https://www.zhihu.com/zvideo/1504457861418061824)
- \[2022/03/24\] [# 人体姿态估计中回归出了heatmap如何去计算关键点的坐标位置？](https://www.zhihu.com/answer/2404996258)







## 7. 视频感知 & MMTracking

<!--- [<img src="https://github.com/open-mmlab/mmtracking/blob/master/resources/mmtrack-logo.png" height="36">](https://github.com/open-mmlab/mmtracking) --->


- \[2021/09/15\] [# 号外号外～ MMTracking 要开始持续更新啦](https://zhuanlan.zhihu.com/p/411005827)
- \[2021/09/27\] [# 快速上手！MMTracking 食用指南 之 VID 篇（附 AAAI2021 论文解读 ！）](https://zhuanlan.zhihu.com/p/412817354)
- \[2021/10/09\] [# MMTracking 多目标跟踪(MOT)任务的食用指南](https://zhuanlan.zhihu.com/p/414625166)
- \[2021/10/15\] [# 上新！MMTracking 单目标跟踪任务食用指南](https://zhuanlan.zhihu.com/p/421031509)
- \[2021/11/11\] [# 最新上线！MMTracking 视频实例分割食用指南](https://zhuanlan.zhihu.com/p/439562841)
- \[2021/11/29\] [# 最新上线！MMTracking 视频实例分割食用指南](https://zhuanlan.zhihu.com/p/439562841)
- \[2021/10/13\] [# 使用深度学习算法实现图像目标跟踪，该怎么做？机器学习刚入门，完全没头绪。?](https://www.zhihu.com/answer/2168982029)
- \[2021/05/03\] [# 如果我想要深入的学习计算机目标跟踪方向的内容，应该从哪个方面开始入手，比如说看什么书?](https://www.zhihu.com/answer/1866682282)







## 8. 光流估计 & MMFlow

<!--- [<img src="https://github.com/open-mmlab/mmflow/blob/master/resources/mmflow-logo.png" height="36">](https://github.com/open-mmlab/mmflow) --->


- \[2021/11/16\] [# 重磅开源！OpenMMLab 光流算法框架：MMFlow](https://zhuanlan.zhihu.com/p/434037886)
- \[2021/12/20\] [# 光流模型概述：从 PWC-Net 到 RAFT](https://zhuanlan.zhihu.com/p/446739441)







## 9. 计算机视觉基础库 & MMCV

<!--- [<img src="https://github.com/open-mmlab/mmcv/blob/master/docs/en/mmcv-logo.png" height="36">](https://github.com/open-mmlab/mmcv) --->


- \[2021/10/13\] [# OpenMMLab 的 cfg 模式和 Registry 机制](https://zhuanlan.zhihu.com/p/387484734)
- \[2021/10/28\] [# 基于 MMCV 走上开源大佬之路？](https://zhuanlan.zhihu.com/p/391144979)
- \[2021/11/15\] [# 拿什么拯救我的 4G 显卡](https://zhuanlan.zhihu.com/p/430123077)
- \[2021/12/24\] [# MMCV Hook 食用指南](https://zhuanlan.zhihu.com/p/448600739)
- \[2022/01/14\] [# 训练可视化工具哪款是你的菜？MMCV一行代码随你挑](https://zhuanlan.zhihu.com/p/387078211)
- \[2022/01/20\] [# 解读 OpenMMLab 的 Hook 机制](https://zhuanlan.zhihu.com/p/387483425)
- \[2022/02/09\] [# 手把手教你如何高效地在 MMCV 中贡献算子](https://zhuanlan.zhihu.com/p/464492627)
- \[2022/03/15\] [# logging 详解第一期：是谁偷偷动了我的 logger](https://zhuanlan.zhihu.com/p/481383590)
- \[2022/03/25\] [# logging 详解第二期：三句话，让 logger 言听计从](https://zhuanlan.zhihu.com/p/487524917)
- \[2022/04/21\] [# logging 详解第三期：Logging 不为人知的二三事](https://zhuanlan.zhihu.com/p/502610682)
- \[2022/05/20\] [# OpenMMLab 支持 IPU 训练芯片](https://zhuanlan.zhihu.com/p/517527926)
- \[2022/06/13\] [# 【社区开放麦】第 9 期 揭秘 OpenMMLab 模块化设计背后的功臣](https://www.zhihu.com/zvideo/1521928802674864128)
- \[2021/12/30\] [# PyTorch & MMCV Dispatcher 机制解析](https://zhuanlan.zhihu.com/p/451671838)
- \[2022/07/21\] [# 深度学习方面的科研工作中的实验代码有什么规范和写作技巧？如何妥善管理实验数据？](https://www.zhihu.com/answer/2586000037)
- \[2022/05/11\] [# 深度学习科研，如何高效进行代码和实验管理？](https://www.zhihu.com/answer/2480772257)
- \[2021/12/07\] [# Pytorch有什么节省显存的小技巧？](https://www.zhihu.com/answer/2260661999)
- \[2022/07/21\] [# 深度学习方面的科研工作中的实验代码有什么规范和写作技巧？如何妥善管理实验数据？](https://www.zhihu.com/answer/2586000037)







## 10. 自监督学习 & MMSelfSup

<!--- [<img src="https://github.com/open-mmlab/mmselfsup/blob/master/resources/mmselfsup_logo.png" height="36">](https://github.com/open-mmlab/mmselfsup) --->


| 标题 | 类型 | 知乎 | 公众号 | 发表日期 |
|--|--|--|--|--|
| 向我们迎面走来的，是有较强自我管理意识的MMSelfSup！ | 文章 | [# 向我们迎面走来的是：有较强自我管理意识的MMSelfSup！](https://zhuanlan.zhihu.com/p/445771658) | [# 向我们迎面走来的，是有较强自我管理意识的MMSelfSup！](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247485681&idx=1&sn=6d4032437fc81ff8e7edffbc1db746cf&chksm=ebb50013dcc289056829f8212bb7a5b7c586150e2ffb871f3aaccfdaa2e9015438bae0c7d536&token=1125533908&lang=zh_CN#rd) | 2021/12/16 |
| MMSelfSup - MAE 尝鲜版来啦！ | 文章 | [# MMSelfSup - MAE 尝鲜版来啦！](https://zhuanlan.zhihu.com/p/454358280) | [# MMSelfSup - MAE 尝鲜版来啦！](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247485976&idx=1&sn=e4cd66e8b879a62bc9e62d5e09ef3bb7&chksm=ebb502fadcc28becd3006a56b03d8e5e5dd71af94a818b61d8813a623dc993701631c3c83df3&token=1125533908&lang=zh_CN#rd) | 2022/01/07 |
| 自监督这么热，快抓紧时间上车：基于 Pretext Task 的自监督学习 | 文章 | [# 自监督学习系列（一）：基于 Pretext Task](https://zhuanlan.zhihu.com/p/470914640) | [# 自监督这么热，快抓紧时间上车：基于 Pretext Task 的自监督学习](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247487906&idx=1&sn=1f2f08006b58b17c590a536efa2e2857&chksm=ebb51940dcc29056347a1be7dd7c6b72b2fc2c9980e933d84b76bc3a30285cbc191313fde5be#rd) | 2022/02/23 |
| 拉开你和别人的距离，只差 Contrastive Learning 这一步 | 文章 | [# 自监督学习系列（二）：基于 Contrastive Learning](https://zhuanlan.zhihu.com/p/474847821) | [# 拉开你和别人的距离，只差 Contrastive Learning 这一步](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247488233&idx=1&sn=4f4a5e6b2f00c80c90b49e71e8be9321&chksm=ebb51a0bdcc2931da43171ba39d602469201e7fc857d02dfba12ccba8acab8410352c0e59ee3#rd) | 2022/03/02 |
| 见微知著，掩码自监督学习让你一叶知秋 | 文章 | [# 自监督学习系列（三）：基于 Masked Image Modeling](https://zhuanlan.zhihu.com/p/475952825) | [# 见微知著，掩码自监督学习让你一叶知秋](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247488305&idx=1&sn=d7285f62ce28630a9774917585ddac32&chksm=ebb51bd3dcc292c59ccf63b7888f9f8c97e016fd181ae1dbdf2eab61e8964728c5e7ade41e71#rd) | 2022/03/04 |
| 简单的结构，优异的性能，SimMIM 来了！ | 文章 | [# 简单的结构，优异的性能，SimMIM 来了！](https://zhuanlan.zhihu.com/p/491004196) | [# 简单的结构，优异的性能，SimMIM 来了！](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247489971&idx=1&sn=a57cb9b9f0a7e94fadf2c75588524ebc&chksm=ebb51151dcc29847ff22fc5edab44f9a81d4b69fb3f7bc4e42365d0d62eecca9d7d7f5427325#rd) | 2022/03/31 |
| 更好的性能！新型自监督学习方法 CAE 了解一下 | 文章 | [# 更好的性能！新型自监督学习方法 CAE 了解一下](https://zhuanlan.zhihu.com/p/510279419) | [# 更好的性能！新型自监督学习方法 CAE 了解一下](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247490521&idx=1&sn=f3ff8dffd3fb1ca212def6c8bc4af175&chksm=ebb5133bdcc29a2d1628c59fb49c2b5b7cb2fc99f4d662d247bf57838f94d5831f86e2fd778b#rd) | 2022/05/06 |
| 手把手带你高效复现最新自监督算法 | 文章 | [# 【预告】社区开放麦第 4 期：手把手带你高效复现最新自监算法](https://zhuanlan.zhihu.com/p/511711378) | [# 手把手带你高效复现最新自监督算法](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247490544&idx=1&sn=e979779ecf3fa9387c7770bcf0728cde&chksm=ebb51312dcc29a047bb78d20baccf9bf4cca14f08dccb9e719a1e7872c16983b4d997614b2f9#rd) | 2022/05/09 |
| CVPR22 Oral TransRank: 利用排序损失提供高质量自监督信号 | 文章 | [# CVPR22 Oral TransRank: 利用排序损失提供高质量自监督信号](https://zhuanlan.zhihu.com/p/526591316) | [# CVPR22 Oral TransRank: 利用排序损失提供高质量自监督信号](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247492600&idx=1&sn=55565af41ea561e7134117daf3e449f8&chksm=ebb6eb1adcc1620cab9973d26aad3369041dd6f13b318c577ec11198f8742b9f28df97ecb139&token=1125533908&lang=zh_CN#rd) | 2022/06/09 |
| 你见过哪些新颖的或有效的「自监督学习样本构建技巧」？ | 问答 | [# 你见过哪些新颖的或有效的「自监督学习样本构建技巧」？](https://www.zhihu.com/answer/2418397841) | | 2022/04/01 |
| 如何评价FAIR提出的MaskFeat：一种适用图像和视频分类的自监督学习方法？ | 问答 | [# 如何评价FAIR提出的MaskFeat：一种适用图像和视频分类的自监督学习方法？](https://www.zhihu.com/answer/2388634728) | | 2022/03/14 |
| 有监督和无监督学习都各有哪些有名的算法和深度学习？ | 问答 | [# 有监督和无监督学习都各有哪些有名的算法和深度学习？](https://www.zhihu.com/answer/2377782709) | | 2022/03/07 |
| 自监督学习（Self-supervised Learning）有什么比较新的思路？ | 问答 | [# 自监督学习（Self-supervised Learning）有什么比较新的思路？](https://www.zhihu.com/answer/2368764990) | | 2022/03/01 |







## 11. 视频和图像编辑 & MMEditing

<!--- [<img src="https://github.com/open-mmlab/mmediting/blob/master/docs/en/_static/image/mmediting-logo.png" height="36">](https://github.com/open-mmlab/mmediting) --->


| 标题 | 类型 | 知乎 | 公众号 | 发表日期 |
|--|--|--|--|--|
| MMEditing 新视频超分算法冠军BasicVSR++来了 | 文章 | [# BasicVSR++: MMEditing 让你离 NTIRE 冠军只有一步之遥](https://zhuanlan.zhihu.com/p/397941254) | [# MMEditing 新视频超分算法冠军BasicVSR++来了](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247483847&idx=1&sn=4c7b166efab7cc15af1f3b12ccd0a7b6&chksm=ebb50925dcc28033f56d3111533ab9fdfd551d6f351dd2a89f130208ed99134ccc3cb65612c4&token=1125533908&lang=zh_CN#rd) | 2021/08/13 |
| 零基础 Pytorch 入门超分辨率 | 文章 | [# 零基础 PyTorch 入门超分辨率](https://zhuanlan.zhihu.com/p/393371989) | [# 零基础 Pytorch 入门超分辨率](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247484944&idx=1&sn=a5beb51cc709484519e3c60c2f9c5557&chksm=ebb50ef2dcc287e42c9da13c784e60a3ca03affdb5313e69c4d7fc6009bc6e45ec34f1275b15&token=1125533908&lang=zh_CN#rd) | 2021/10/26 |
| 让 GLEAN 还原你女神的美妙容颜 | 文章 | [# GLEAN：一键让你跟低清人脸说再见](https://zhuanlan.zhihu.com/p/448072439) | [# 让 GLEAN 还原你女神的美妙容颜](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247485438&idx=1&sn=4db5342eee135b66757638743d2b4c5f&chksm=ebb50f1cdcc2860a2e3bce2c7bf4619172959475d4fcf2c39e4662286e8d750b76e93d236f24&token=1125533908&lang=zh_CN#rd) | 2021/11/30 |
| 一键慢镜头：视频插帧，让老电影“纵享丝滑” | 文章 | [# 一键慢镜头：视频插帧，让老电影“纵享丝滑”](https://zhuanlan.zhihu.com/p/471878119) | [# 一键慢镜头：视频插帧，让老电影“纵享丝滑”](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247487939&idx=1&sn=3075c73123e514ca8c3ae4dd903d6d7f&chksm=ebb51921dcc290375224adc5f28a53038112cc5f528475b953014825d32d4f7989277c3260df#rd) | 2022/02/24 |
| 不容错过！作者亲自解读 RealBasicVSR | 文章 | [# 不容错过！作者亲自解读 CVPR 2022 RealBasicVSR](https://zhuanlan.zhihu.com/p/482656858) | [# 不容错过！作者亲自解读 RealBasicVSR](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247489508&idx=1&sn=07d27e99713d052c8a088553707f1adf&chksm=ebb51f06dcc29610761f2623fe3e86f0403bbf1c939eb86cce2b55faf3b315ac0d39d39254ac#rd) | 2022/03/17 |
| 视觉底层任务优秀开源工作：MMEditing 库使用方法 | 文章 | [# 视觉底层任务优秀开源工作：MMEditing 库使用方法](https://zhuanlan.zhihu.com/p/466999485) | [# 视觉底层任务优秀开源工作：MMEditing 库使用方法](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247489786&idx=1&sn=a103a893b38f66759b969590a98f1475&chksm=ebb51018dcc2990e66ec61b34cc925b87858d3dec41075d71772f0602406e912cd64d07e40b7#rd) | 2022/03/29 |
| 手把手带你训练 CVPR2022 视频超分模型 | 文章 | [# 手把手带你训练 CVPR2022 视频超分模型](https://zhuanlan.zhihu.com/p/500687519) | [# 手把手带你训练 CVPR2022 视频超分模型](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247490241&idx=1&sn=519312b9df321181385e84d996c91832&chksm=ebb51223dcc29b35e079408e76d0a58a9486b012c21fa81b8813296c89cf7945ad7589a4821f#rd) | 2022/04/18 |
| 还在眨眼补帧？手把手教你插帧算法，让视频顺滑如丝 | 文章 | [# 基于光流的视频插帧算法 TOFlow 解读教程](https://zhuanlan.zhihu.com/p/535492591) | [# 还在眨眼补帧？手把手教你插帧算法，让视频顺滑如丝](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247493512&idx=1&sn=92f81a0ab8b45a686d1100e3aae78144&chksm=ebb6ef6adcc1667c491ec104db9e39257656aae5ffb23fc425228ae5368ab0ebad1d31d0ed22&token=1125533908&lang=zh_CN#rd) | 2022/06/29 |
| 传统的图像修复和利用深度学习的图像修复的优缺点比较？ | 问答 | [# 传统的图像修复和利用深度学习的图像修复的优缺点比较？](https://www.zhihu.com/answer/1618198292) | | 2020/12/09 |







## 12. 旋转框检测 & MMRotate

<!--- [<img src="https://github.com/open-mmlab/mmrotate/blob/main/resources/mmrotate-logo.png" height="36">](https://github.com/open-mmlab/mmrotate) --->


| 标题 | 类型 | 知乎 | 公众号 | 发表日期 |
|--|--|--|--|--|
| 旋转~跳跃~检测王者 MMDetection 的好兄弟来啦！ | 文章 | [# OpenMMLab 正式开源 MMRotate, 专注于旋转目标检测](https://zhuanlan.zhihu.com/p/469065580) | [# 旋转~跳跃~检测王者 MMDetection 的好兄弟来啦！]() | 2022/02/18 |
| 目标检测中旋转问题有哪些常用的解决方案？ | 问答 | [# 目标检测中旋转问题有哪些常用的解决方案？](https://www.zhihu.com/answer/2359366595) | | 2022/02/22 |
| 如何把一个水平框的目标检测框架改成旋转框的目标检测框架？ | 问答 | [# 如何把一个水平框的目标检测框架改成旋转框的目标检测框架？](https://www.zhihu.com/answer/2359334765) | | 2022/02/22 |







## 13. 模型压缩 & MMRazor

<!--- [<img src="https://github.com/open-mmlab/mmrazor/blob/master/resources/mmrazor-logo.png" height="36">](https://github.com/open-mmlab/mmrazor) --->


| 标题 | 类型 | 知乎 | 公众号 | 发表日期 |
|--|--|--|--|--|
| 蒸馏、剪枝、网络结构搜索全方向覆盖！模型轻量化，没有比 MMRazor 更锋利的 | 文章 | [# 蒸馏、剪枝、网络结构搜索全方向覆盖！模型轻量化，没有比MMRazor更锋利的](https://zhuanlan.zhihu.com/p/448896019) | [# 蒸馏、剪枝、网络结构搜索全方向覆盖！模型轻量化，没有比 MMRazor 更锋利的](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247485734&idx=1&sn=f0701f5a6745c18a33e6c5a289bf345a&chksm=ebb501c4dcc288d2968391865fc1010be5049bf045d5175f108f248db623e22f67f86711744e&token=1125533908&lang=zh_CN#rd) | 2021/12/23 |
| 经典网络结构搜索算法 SPOS，助你快速完成模型压缩 | 文章 | [# 经典网络结构搜索算法 SPOS，快速完成模型压缩](https://zhuanlan.zhihu.com/p/538779766) | [# 经典网络结构搜索算法 SPOS，助你快速完成模型压缩](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247493919&idx=1&sn=b91c4e5a514fcfc59d980fc21e284d6a&chksm=ebb6e1fddcc168ebe4632924cdbfa90c0b5c9e0ff17f087797df9022b1975b0c0a6e39af8ce1&token=1125533908&lang=zh_CN#rd) | 2022/07/07 |







## 14. 人体参数化模型 & MMHuman3D

<!--- [<img src="https://github.com/open-mmlab/mmhuman3d/blob/main/resources/mmhuman3d-logo.png" height="36">](https://github.com/open-mmlab/mmhuman3d) --->


| 标题 | 类型 | 知乎 | 公众号 | 发表日期 |
|--|--|--|--|--|
| 画形亦画骨，知面也知心，与 MMHuman3D 一道探索人体参数化模型 | 文章 | [# 画形亦画骨，知面也知心，与 MMHuman3D 一道探索人体参数化模型](https://zhuanlan.zhihu.com/p/440090661) | [# 画形亦画骨，知面也知心，与 MMHuman3D 一道探索人体参数化模型](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247485482&idx=1&sn=808e3754e37199a9ecd8d1b2adb13172&chksm=ebb500c8dcc289de531dfc948dcbeb669d0a56bda1b50fb4323905bf1e49bcf1b889f3ed15ca&token=1125533908&lang=zh_CN#rd) | 2021/12/03 |







## 15. 少样本学习 & MMFewShot

<!--- [<img src="https://github.com/open-mmlab/mmfewshot/blob/main/resources/mmfewshot-logo.png" height="36">](https://github.com/open-mmlab/mmfewshot) --->


| 标题 | 类型 | 知乎 | 公众号 | 发表日期 |
|--|--|--|--|--|
| 举一隅而以三隅反，MMFewshot 带你走近少样本分类 | 文章 | [# 举一隅而以三隅反，MMFewShot 带你走近少样本学习【MMFewshot重磅开源！】](https://zhuanlan.zhihu.com/p/437038040) | [# 举一隅而以三隅反，MMFewshot 带你走近少样本分类](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247485382&idx=1&sn=1a31969b3ba47a4cb701f30a65e8472b&chksm=ebb50f24dcc286326cbcb903cd434fd8932a3c5f7b9618284cc0802ab68bf768166cfcafa6a1&token=1125533908&lang=zh_CN#rd) | 2021/11/24 |







## 16. 行为理解 & MMAction2

<!--- [<img src="https://github.com/open-mmlab/mmaction2/blob/master/resources/mmaction2_logo.png" height="36">](https://github.com/open-mmlab/mmaction2) --->


| 标题 | 类型 | 知乎 | 公众号 | 发表日期 |
|--|--|--|--|--|
| MMAction2 基于人体姿态的动作识别新范式 PoseC3D | 文章 | [# PoseC3D: 基于人体姿态的动作识别新范式](https://zhuanlan.zhihu.com/p/395588459) | [# MMAction2 基于人体姿态的动作识别新范式 PoseC3D](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247484134&idx=1&sn=db71f358cbb49dc41bed2813560ad6f3&chksm=ebb50a04dcc28312a6f2a753bc5fa2e2d9d44e71e49132ecf362f0279845f75a7354e9dedd99&token=1125533908&lang=zh_CN#rd) | 2021/08/27 |
| 超轻量更泛化！基于人体骨骼点的动作识别 | 文章 | [# 超轻量更泛化！基于人体骨骼点的动作识别](https://zhuanlan.zhihu.com/p/426695879) | [# 超轻量更泛化！基于人体骨骼点的动作识别](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247485011&idx=1&sn=fb45c0eb5c08f6d0d2e77d0ce6647c63&chksm=ebb50eb1dcc287a746ece8a6814a7b63202e96b83f2414b6517b7288da702714fefa83c3d2a4&token=1125533908&lang=zh_CN#rd) | 2021/11/02 |
| 视频训练效率太低？试试这个加速算法 | 文章 | [# 视频训练效率太低？Multigrid 加速算法了解一下](https://zhuanlan.zhihu.com/p/481993402) | [# 视频训练效率太低？试试这个加速算法](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247489386&idx=1&sn=bcbc6aa1cc9737ad6de5ca66ce4c2206&chksm=ebb51f88dcc2969e75ed7a975f619dead48249daf6ba1b33ca147ac7b8886691768879c96683#rd) | 2022/03/16 |
| 如何学习视频识别技术？ | 问答 | [# 如何学习视频识别技术？](https://www.zhihu.com/answer/1864039491) | | 2021/05/01 |
| 行为识别(action recognition)有哪些论文适合入门？ | 问答 | [# 行为识别(action recognition)有哪些论文适合入门？](https://www.zhihu.com/answer/1616881232) | | 2020/12/08 |







## 17. 文本检测识别理解 & MMOCR

<!--- [<img src="https://github.com/open-mmlab/mmocr/blob/main/resources/mmocr-logo.png" height="36">](https://github.com/open-mmlab/mmocr) --->


| 标题 | 类型 | 知乎 | 公众号 | 发表日期 |
|--|--|--|--|--|
| MMOCR 全方位食用指南拿来吧 | 文章 | [# 拿来吧你！MMOCR 全方位食用指南](https://zhuanlan.zhihu.com/p/400578588) | [# MMOCR 全方位食用指南拿来吧你！](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247483994&idx=1&sn=3e6f426183f79c0066e3eee687c6a24c&chksm=ebb50ab8dcc283aebb758eaf1acf6b56098da40d1a14c2d9f5985d8ad748866a53c704085760&token=1125533908&lang=zh_CN#rd) | 2021/08/20 |
| 如何看待OpenMMlab最新开源项目MMOCR？ | 问答 | [# 如何看待OpenMMlab最新开源项目MMOCR？](https://www.zhihu.com/answer/1830774267) | | 2021/04/12 |







## 18. 生成模型 & MMGeneration

<!--- [<img src="./resources/mmgeneration_logo.png" height="36">](https://github.com/open-mmlab/mmgeneration) --->


| 标题 | 类型 | 知乎 | 公众号 | 发表日期 |
|--|--|--|--|--|
| MMGeneration PyTorch 零基础入门 GAN 模型 | 文章 | [# PyTorch 零基础入门 GAN 模型之基础篇](https://zhuanlan.zhihu.com/p/396010666) | | 2021/08/16 |
| MMGEN-FaceStylor 因为是你，所以每一种样子我都喜欢 | 文章 | [# MMGEN-FaceStylor 因为是你，所以每一种样子我都喜欢](https://zhuanlan.zhihu.com/p/443632127) | [# MMGEN-FaceStylor 因为是你，所以每一种样子我都喜欢](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247485615&idx=1&sn=4c2d8c9958ac47f6e34780d7f8e18179&chksm=ebb5004ddcc2895b66174fcee764e1cc926f7a8b8118430b8fd1039323370b8382e9fc212640&token=1125533908&lang=zh_CN#rd) | 2021/12/10 |
| PyTorch 零基础入门 GAN 模型之 cGAN | 文章 | [# PyTorch 零基础入门 GAN 模型之 cGAN](https://zhuanlan.zhihu.com/p/490317358) | [# PyTorch 零基础入门 GAN 模型之 cGAN](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247489957&idx=1&sn=2e2018388f01af3c18f79c21479a7e57&chksm=ebb51147dcc298514697a1e495232b2da9bd817111234565f6fd8449cd131a9b7d36e3af83cc#rd) | 2022/03/30 |
| 生成式对抗网络GAN有哪些最新的发展，可以实际应用到哪些场景中？ | 问答 | [# 生成式对抗网络GAN有哪些最新的发展，可以实际应用到哪些场景中？](https://www.zhihu.com/answer/2471545183) | | 2022/05/05 |
| GAN网络训练过拟合如何解决? | 问答 | [# GAN网络训练过拟合如何解决?](https://www.zhihu.com/answer/2416395711) | | 2022/03/31 |
| GAN今年凉了吗？ | 问答 | [# GAN今年凉了吗？](https://www.zhihu.com/answer/1618193771) | | 2020/12/09 |







## 19. Pytorch & General


- \[2022/02/22\][# 困扰我 48 小时的深拷贝，今天终于...](https://zhuanlan.zhihu.com/p/470892209)

| 标题 | 类型 | 知乎 | 公众号 | 发表日期 |
|--|--|--|--|--|
| PyTorch 零基础入门 GAN 模型之评价指标 | 文章 | [# PyTorch 零基础入门 GAN 模型之评价指标](https://zhuanlan.zhihu.com/p/428527281) | [# PyTorch 零基础入门 GAN 模型之评价指标](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247485032&idx=1&sn=3b20a89c141a69e4da1a1a9b82f59c10&chksm=ebb50e8adcc2879c1f6729bf420d6ee94b65d55b0ee3e4944a81bbe3a47b714b30c65fc47d26&token=1125533908&lang=zh_CN#rd) | 2021/11/03 |
| 一睹为快！PyTorch1.11 亮点一览 | 文章 | [# PyTorch1.11 亮点一览：TorchData、functorch、DDP 静态图](https://zhuanlan.zhihu.com/p/486222256) | [# 一睹为快！PyTorch1.11 亮点一览](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247489699&idx=1&sn=d9021a31d46eda85c0bb26c6f52c7239&chksm=ebb51041dcc2995742c5e6c691fa5117d79837670cda740fe166b10a5c2772fa19406d1f002a#rd) | 2022/03/23 |
| PyTorch 小课堂开课啦！带你解析数据处理全流程（一） | 文章 | [# PyTorch 源码解读之 torch.utils.data：解析数据处理全流程](https://zhuanlan.zhihu.com/p/337850513) | [# PyTorch 小课堂开课啦！带你解析数据处理全流程（一）](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247490074&idx=1&sn=019fcbf4574cdf32c3f2465f42bb4bf3&chksm=ebb512f8dcc29bee7a7e67c520cc121d48e7dc32792bcbb944fb8d25062d275f199984f9f772#rd) | 2022/04/08 |
| PyTorch 小课堂！带你解析数据处理全流程（二） | 文章 | [# PyTorch 源码解读之 torch.utils.data：解析数据处理全流程](https://zhuanlan.zhihu.com/p/337850513) | [# PyTorch 小课堂！带你解析数据处理全流程（二）](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247490145&idx=1&sn=bf079b1db74f59fbe8f2b1bfa382b7c2&chksm=ebb51283dcc29b9553671f31da1616431130318806cea72dc238fb87359db7f2f9b339454fa0#rd) | 2022/04/13 |
| PyTorch 小课堂！一篇看懂核心网络模块接口 | 文章 | [# PyTorch 源码解读之 nn.Module：核心网络模块接口详解](https://zhuanlan.zhihu.com/p/340453841) | [# PyTorch 小课堂！一篇看懂核心网络模块接口](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247490275&idx=1&sn=58dfde00497a20068251530bbb5ba4dd&chksm=ebb51201dcc29b1700fcdcc9054982b8e72ddfc66d0b001750b94ce28c9750215a1da533f132#rd) | 2022/04/20 |
| 一小时带你了解 Torch DDP 背后的系统设计 | 文章 | [# 【预告】社区开放麦第3期：带你了解 Torch DDP 背后的系统设计](https://zhuanlan.zhihu.com/p/508685383) | [# 一小时带你了解 Torch DDP 背后的系统设计](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247490483&idx=1&sn=6971d7a05232f9dcadd5861ed02224cd&chksm=ebb51351dcc29a47fffb1d8f813c11ac1c6a1fe240ac86006bfeb26da9bd67b65b2a38b8e71e#rd) | 2022/05/03 |
| PyTorch 小课堂！梯度计算详解 | 文章 | [# PyTorch 源码解读之 torch.autograd：梯度计算详解](https://zhuanlan.zhihu.com/p/321449610) | [# PyTorch 小课堂！梯度计算详解](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247491513&idx=1&sn=c169809b9586119147e9b1da267d6a4e&chksm=ebb5175bdcc29e4d73d55f842c2b0e7d5333df08ce9fcab6443a314f180c730fdf0c1957b36b&token=1125533908&lang=zh_CN#rd) | 2022/06/02 |
| PyTorch1.12 亮点一览！新的数据加载与处理范式 | 文章 | [# PyTorch1.12 亮点一览 DataPipe + TorchArrow 新的数据加载与处理范式](https://zhuanlan.zhihu.com/p/537868554) | [# PyTorch1.12 亮点一览！新的数据加载与处理范式](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247493740&idx=1&sn=d839a418d64b9fa585017c4dd4e835e3&chksm=ebb6e08edcc16998cb00c54e0bd7b3a5a30ccd60b7ca5f8190326762b12f6719dc6045e371f9&token=1125533908&lang=zh_CN#rd) | 2022/07/05 |
| 有没有那种直接上手就能提 PR 的？文末福利！ | 文章 | [# OpenMMLab 【成为我们的贡献者】有奖活动正式开启！](https://zhuanlan.zhihu.com/p/472372230) | [# 有没有那种直接上手就能提 PR 的？文末福利！](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247487965&idx=1&sn=8dd4691b2f6e785876c6d7c912b97e2b&chksm=ebb5193fdcc2902974b697e086e804756fc4bd488d64e8b54bc8dc1a12929b3761cb34a7338a#rd) | 2022/02/25 |
| 新栏目【社区开放麦】开播啦！首期我们聊聊这个 | 文章 | [# 【预告】社区开放麦第 1 期：基于关键点的动作识别](https://zhuanlan.zhihu.com/p/499228474) | [# 新栏目【社区开放麦】开播啦！首期我们聊聊这个](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247490227&idx=1&sn=0d32eb93192cb0976b65893aea00e6a1&chksm=ebb51251dcc29b47fa281135de1525ab9a100bccb4781aaafc1cc37282b1f76bace3d6902d2b#rd) | 2022/04/15 |
| 访问 GitHub 太慢？OpenMMLab 入驻 Gitee！ | 文章 | [# 访问 GitHub 太慢？OpenMMLab 入驻 Gitee！](https://zhuanlan.zhihu.com/p/506187883) | [# 访问 GitHub 太慢？OpenMMLab 入驻 Gitee！](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247490425&idx=1&sn=8d26eb866eaa19f0da2ad7f7d8194b11&chksm=ebb5139bdcc29a8d29f1ce386d62ad2382aae5ad96546470b3a6877e69e38987c447aafb3518#rd) | 2022/04/27 |
| 计算机研究生刚上岸，深度学习方向，想要就业的话，应该如何规划研究生三年？ | 问答 | [# 计算机研究生刚上岸，深度学习方向，想要就业的话，应该如何规划研究生三年？](https://www.zhihu.com/answer/2070065874) | | 2021/08/19 |
| 应届硕士毕业生如何拿到知名互联网公司算法岗（机器学习、数据挖掘、深度学习） offer？ | 问答 | [# 应届硕士毕业生如何拿到知名互联网公司算法岗（机器学习、数据挖掘、深度学习） offer？](https://www.zhihu.com/answer/1954623664) | | 2021/06/22 |
| 非计算机专业的学生如何入门深度学习？ | 问答 | [# 非计算机专业的学生如何入门深度学习？](https://www.zhihu.com/answer/1890872728) | | 2021/05/17 |
| 新入学的计算机研究生怎么安排三年学习深度学习？ | 问答 | [# 新入学的计算机研究生怎么安排三年学习深度学习？](https://www.zhihu.com/answer/1869741579) | | 2021/05/05 |
| 国内 top2 高校研一在读，为什么感觉深度学习越学越懵? | 问答 | [# 国内 top2 高校研一在读，为什么感觉深度学习越学越懵?](https://www.zhihu.com/answer/1868579489) | | 2021/05/04 |
| 如何在GitHub上做一个优秀的贡献者？ | 问答 | [# 如何在GitHub上做一个优秀的贡献者？](https://www.zhihu.com/answer/1859051033) | | 2021/04/28 |
| 如何成为开源项目的Committer/Collaborator/Member？ | 问答 | [# 如何成为开源项目的Committer/Collaborator/Member？](https://www.zhihu.com/answer/1853596341) | | 2021/04/25 |
| 为什么要开源？ | 问答 | [# 为什么要开源？](https://www.zhihu.com/answer/1851683514) | | 2021/04/24 |
| 对自己深度学习方向的论文有idea，可是工程实践能力跟不上，实验搞不定怎么办？ | 问答 | [# 对自己深度学习方向的论文有idea，可是工程实践能力跟不上，实验搞不定怎么办？](https://www.zhihu.com/answer/1849303303) | | 2021/04/22 |
| 如何看待国内开源项目的不可持续性？ | 问答 | [# 如何看待国内开源项目的不可持续性？](https://www.zhihu.com/answer/1828235157) | | 2021/04/10 |
| GitHub 上有哪些适合新手跟进的优质项目？ | 问答 | [# GitHub 上有哪些适合新手跟进的优质项目？](https://www.zhihu.com/answer/1809829847) | | 2021/03/31 |
| 如何通过 GitHub 加入开源项目？ | 问答 | [# 如何通过 GitHub 加入开源项目？](https://www.zhihu.com/answer/1807472905) | | 2021/03/30 |
| 我想成为一个开源代码贡献者，我该怎么做？ | 问答 | [# 我想成为一个开源代码贡献者，我该怎么做？](https://www.zhihu.com/answer/1759679319) | | 2021/03/03 |
| 在校生如何在开源社区中成长？ | 问答 | [# 在校生如何在开源社区中成长？](https://www.zhihu.com/answer/1740864956) | | 2021/02/21 |
| 如何运营一个开源项目并取得较大影响力？ | 问答 | [# 如何运营一个开源项目并取得较大影响力？](https://www.zhihu.com/answer/1722994874) | | 2021/02/09 |
| 新手如何入门pytorch？ | 问答 | [# 新手如何入门pytorch？](https://www.zhihu.com/answer/1719997534) | | 2021/02/07 |
| 如果学习从零开始学习Pytorch,有优秀的开源项目可以推荐吗？ | 问答 | [# 如果学习从零开始学习Pytorch,有优秀的开源项目可以推荐吗？](https://www.zhihu.com/answer/1705390205) | | 2021/01/30 |
| 如何最简单、通俗地理解Pytorch？ | 问答 | [# 如何最简单、通俗地理解Pytorch？](https://www.zhihu.com/answer/1691272176) | | 2021/01/22 |
| 如何看待Transformer在CV上的应用前景，未来有可能替代CNN吗？ | 问答 | [# 如何看待Transformer在CV上的应用前景，未来有可能替代CNN吗？](https://www.zhihu.com/answer/1686380553) | | 2021/01/19 |
| PyTorch把tensor的require_grad设置为True对最终的结果有什么影响？ | 问答 | [# PyTorch把tensor的require_grad设置为True对最终的结果有什么影响？](https://www.zhihu.com/answer/1652900621) | | 2020/12/30 |

