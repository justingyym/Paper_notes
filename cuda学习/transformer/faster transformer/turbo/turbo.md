# 

## 相关资料：

详细介绍 https://baijiahao.baidu.com/s?id=1665109327983236824&wfr=spider&for=pc



**层融合**



**动态显存复用**

为了避免计算过程中的显存申请释放并节省显存占用，LightSeq 首先对模型中所有动态的 shape 都定义了最大值（例如最大序列长度），将所有动态shape转换为静态。接着在服务启动的时候，为计算过程中的每个中间计算结果按最大值分配显存，并对没有依赖的中间结果共用显存。这样对每个请求，模型推理时不再申请显存，做到了：不同请求的相同 Tensor 复用显存；同请求的不同 Tensor 按 shape 及依赖关系复用显存。

通过该显存复用策略，在一张 T4 显卡上，LightSeq 可以同时部署多达 8 个 Transformer big 模型（batch_size=8，最大序列长度=8，beam_size=4，vocab_size=3万）。从而在低频或错峰等场景下，大大提升显卡利用率。

**层级式解码计算**

在自回归序列生成场景中，最复杂且耗时的部分就是解码。LightSeq 目前已经支持了 beam search、diversity beam search、top-k/top-p sampling 等多种解码方法，并且可以配合 Transformer、GPT使用，达到数倍加速。这里我们以应用最多的 beam search 为例，介绍一下 LightSeq 对解码过程的优化。

```c++
//multi_headed_attention.cpp

MultiHeadedAttention

​		FuseGemm012AddBIasTranspose	
    	kernels::BatchMatMul
    	kernels::ApplyMaskAndSoftmax(GPUSoftmaxMask)
    									cub_softmax_kernel_k
    										BlockReduce
    											warpReduce							
        Reshape
   		kernels::BatchMatMul (cublasSgemmStridedBatched)
        kernels::TransposeForScore
     	kernels::MatMul
   		kernels::AddBiasLayerNorm


```





