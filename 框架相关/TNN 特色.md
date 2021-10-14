前向推理框架调研



## TNN 特色

https://github.com/Tencent/TNN

- 计算优化

- - 针对不同架构在硬件指令发射、吞吐、延迟、缓存带宽、缓存延迟、寄存器数量等特点，深度优化底层算子，极致利用硬件算力
  - 主流硬件平台(CPU: ARMv7， ARMv8， GPU: Mali， Adreno， Apple) 深度调优
  - CNN 核心卷积运算通过 Winograd，Tile-GEMM， Direct Conv 等多种算法实现，保证不同参数、计算尺度下高效计算
  - Op 融合：离线分析网络计算图，多个小 Op（计算量小、功能较简单）融合运算，减少反复内存读取、kernel 启动等开销

- 低精度优化

- - 支持 INT8， FP16 低精度计算，减少模型大小、内存消耗，同时利用硬件低精度计算指令加速计算
  - 支持 INT8 Winograd 算法，(输入6bit)， 在精度满足要求的情况下，进一步降低模型计算复杂度
  - 支持单模型多种精度混合计算，加速计算同时保证模型精度

- 内存优化

- - 高效”内存池”实现：通过 DAG 网络计算图分析，实现无计算依赖的节点间复用内存，降低 90% 内存资源消耗
  - 跨模型内存复用：支持外部实时指定用于网络内存，实现“多个模型，单份内存”

![preview](https://pic2.zhimg.com/v2-f3b7326bff396526515a33184e778d3a_r.jpg)

- TNN采用统一的ONNX模型作为中转，兼容各大框架，这也反映出业界对于ONNX模型的中转方式的认可和推动；
- TNN支持FP16和int8的量化；
- TNN支持计算图的优化，至于具体采用了何种优化模式进行优化，还需要进一步深入代码了解；
- TNN通过抽象化的kernel接口使得算子可以跑在不同的硬件平台之上，支持ARM，GPU，NPU等计算。

另外，根据TNN官方的描述，它还具有优点：

- 通过 ONNX 支持 TensorFlow， PyTorch， MXNet， Caffe 等多种训练框架，充分利用和融入不断完善的 ONNX 开源生态。当前支持 ONNX 算子55个，近期会完善到约80个，覆盖主流CNN网络
- 支持主流安卓、iOS、Embedded Linux 操作系统，支持 ARM CPU， GPU 硬件平台（近期还会加入达芬奇 NPU 支持）
- 模块化设计，将模型解析、计算图构建、优化、底层硬件适配、高性能 kernel 实现各部分抽象隔离，通过 Factory Mode 注册、构建设备，方便接入更多的底层硬件、加速方案。
- Runtime 无任何第三方库依赖，CPU 动态库尺寸仅约 400KB，并提供基础图像变换操作，调用简单便捷。跨平台模型统一、调用接口统一，通过单个配置参数快速切换。





## MNN 特色

https://github.com/alibaba/MNN/blob/master/README_CN.md

## 架构设计

[![architecture](https://github.com/alibaba/MNN/raw/master/doc/architecture.png)](https://github.com/alibaba/MNN/blob/master/doc/architecture.png)

MNN可以分为Converter和Interpreter两部分。

Converter由Frontends和Graph Optimize构成。前者负责支持不同的训练框架，MNN当前支持Tensorflow(Lite)、Caffe和ONNX(PyTorch/MXNet的模型可先转为ONNX模型再转到MNN)；后者通过算子融合、算子替代、布局调整等方式优化图。

Interpreter由Engine和Backends构成。前者负责模型的加载、计算图的调度；后者包含各计算设备下的内存分配、Op实现。在Engine和Backends中，MNN应用了多种优化方案，包括在卷积和反卷积中应用Winograd算法、在矩阵乘法中应用Strassen算法、低精度计算、Neon优化、手写汇编、多线程优化、内存复用、异构计算等。

### 高性能

- 不依赖任何第三方计算库，依靠大量手写汇编实现核心运算，充分发挥ARM CPU的算力。
- iOS设备上可以开启GPU加速（Metal），常用模型上快于苹果原生的CoreML。
- Android上提供了`OpenCL`、`Vulkan`、`OpenGL`三套方案，尽可能多地满足设备需求，针对主流GPU（`Adreno`和`Mali`）做了深度调优。
- 卷积、转置卷积算法高效稳定，对于任意形状的卷积均能高效运行，广泛运用了 Winograd 卷积算法，对3x3 -> 7x7之类的对称卷积有高效的实现。
- 针对ARM v8.2的新架构额外作了优化，新设备可利用FP16半精度计算的特性获得两倍提速。

### 轻量性

- 针对端侧设备特点深度定制和裁剪，无任何依赖，可以方便地部署到移动设备和各种嵌入式设备中。
- iOS平台：armv7+arm64静态库大小5MB左右，链接生成可执行文件增加大小620KB左右，metallib文件600KB左右。
- Android平台：so大小400KB左右，OpenCL库400KB左右，Vulkan库400KB左右。

### 通用性

- 支持`Tensorflow`、`Caffe`、`ONNX`等主流模型文件格式，支持`CNN`、`RNN`、`GAN`等常用网络。
- 转换器支持149个`Tensorflow`OP、58个`TFLite` OP、47个`Caffe` OP、74个`ONNX` OP；各计算设备后端支持的MNN OP数：CPU 111个，ARM V8.2 6个，Metal 55个，OpenCL 43个，Vulkan 32个。
- 支持iOS 8.0+、Android 4.3+和具有POSIX接口的嵌入式设备。
- 支持异构设备混合计算，目前支持CPU和GPU。

### 易用性

- 有高效的图像处理模块，覆盖常见的形变、转换等需求，一般情况下，无需额外引入libyuv或opencv库处理图像。
- 支持回调机制，可以在网络运行中插入回调，提取数据或者控制运行走向。
- 支持只运行网络中的一部分，或者指定CPU和GPU间并行运行。
- （BETA）MNN Python API，让算法工程师可以轻松地使用MNN构图、训练、量化训练，无需编写C++。







# Paddle Lite

https://github.com/PaddlePaddle/Paddle-Lite

Paddle Lite是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位支持包括移动端、嵌入式以及服务器端在内的多硬件平台。

当前Paddle Lite不仅在百度内部业务中得到全面应用，也成功支持了众多外部用户和企业的生产任务。

## 架构设计

Paddle Lite 的架构设计着重考虑了对多硬件和平台的支持，并且强化了多个硬件在一个模型中混合执行的能力，多个层面的性能优化处理，以及对端侧应用的轻量化设计。

[![img](https://github.com/PaddlePaddle/Paddle-Lite/raw/develop/docs/images/architecture.png)](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/images/architecture.png)

其中，Analysis Phase 包括了 MIR(Machine IR) 相关模块，能够对原有的模型的计算图针对具体的硬件列表进行算子融合、计算裁剪 在内的多种优化。Execution Phase 只涉及到Kernel 的执行，且可以单独部署，以支持极致的轻量级部署。

## 主要特性

- 多硬件支持：
  - Paddle Lite架构已经验证和完整支持从 Mobile 到 Server [多种硬件平台](https://paddle-lite.readthedocs.io/zh/latest/introduction/support_hardware.html)，包括 ARM CPU、Mali GPU、Adreno GPU、华为 NPU，以及 FPGA 等，且正在不断增加更多新硬件支持。
  - 各个硬件平台的 Kernel 在代码层和执行层互不干扰，用户不仅可以自由插拔任何硬件，还支持任意系统可见硬件之间的[混合调度](https://paddle-lite.readthedocs.io/zh/latest/introduction/tech_highlights.html#id7)。
- 轻量级部署
  - Paddle Lite在设计上对图优化模块和执行引擎实现了良好的解耦拆分，移动端可以直接部署执行阶段，无任何第三方依赖。
  - 包含完整的80个 op+85个 Kernel 的动态库，对于ARMV7只有800K，ARMV8下为1.3M，并可以通过[裁剪预测](https://paddle-lite.readthedocs.io/zh/latest/user_guides/library_tailoring.html)库进一步减小预测库文件大小。
- 高性能：
  - 极致的 ARM CPU 性能优化：针对不同微架构特点实现kernel的定制，最大发挥计算性能，在主流模型上展现出领先的速度优势。
  - 支持 [PaddleSlim模型压缩工具](https://github.com/PaddlePaddle/PaddleSlim)：支持量化训练、离线量化等多种量化方式，最优可在不损失精度的前提下进一步提升模型推理性能。性能数据请参考 [benchmark](https://paddlepaddle.github.io/Paddle-Lite/develop/benchmark/)。
- 多模型多算子
  - Paddle Lite和PaddlePaddle训练框架的OP对齐，提供广泛的模型支持能力。
  - 目前已严格验证24个模型200个OP的精度和性能，对视觉类模型做到了较为充分的支持，覆盖分类、检测和定位，包含了特色的OCR模型的支持，并在不断丰富中。具体请参考[支持OP](https://paddle-lite.readthedocs.io/zh/latest/introduction/support_operation_list.html)。
- 强大的图分析和优化能力
  - 不同于常规的移动端预测引擎基于 Python 脚本工具转化模型， Lite 架构上有完整基于 C++ 开发的 IR 及相应 Pass 集合，以支持操作熔合，计算剪枝，存储优化，量化计算等多类计算图优化。更多的优化策略可以简单通过 [新增 Pass](https://paddle-lite.readthedocs.io/zh/latest/develop_guides/add_new_pass.html) 的方式模块化支持。

# TVM

https://blog.csdn.net/tlzhatao/article/details/93630910

### Optimizing Computational Graphs（计算图级优化）

1. Operator Fusion

2. Data Layout Transformation

### Generating Tensor Operations（算子级优化）

1. Tensor Expression and Schedule Space

2. Nested Parallelism with Cooperation
3. Tensorization
4. Explicit Memory Latency Hiding

### Automating Optimization

1. Schedule Space Specification
2. ML-Based Cost Model
3. Schedule Exploration



# OpenVINO

**主要特点：**

- 在Intel平台上提升计算机视觉相关深度学习性能达19倍以上
- 解除CNN-based的网络在边缘设备的性能瓶颈
- 对OpenCV，OpenXV*视觉库的传统API实现加速与优化
- 基于通用API接口在CPU、GPU、FPGA等设备上运行加上

![image-20200923144148930](/home/yaoym/.config/Typora/typora-user-images/image-20200923144148930.png)

降低精度

​    fp16/in8

# TensorRT

TensorRT

![image-20200923094638714](/home/yaoym/.config/Typora/typora-user-images/image-20200923094638714.png)

