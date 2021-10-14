# **Faster transformer3.1**

**Faster transformer3.1**

支持pytorch

First, FasterTransformer v3.1 adds the supporting of INT8 quantization of PyTorch encoder model on Turing and Ampere GPUs. Second, v3.1 improves the performances of encoder on FP16 and INT8. Compared to v3.0, v3.1 provides at most 1.2x speedup on T4 FP16, and 1.7x speedup on T4 INT8. Third, v3.1 supports the inference of GPT-2 model.



1. fastertransformer1.0：主要源码
2. 1. cuda：优化后的kernel以及对multi-head attention整体的封装（没过线性层）
   2. tf_op：tensorflow operation和OpKernel的注册（op理解为声明、Opkenerl是定义）
   3. trt_plugin：tensorRT的实现（可以支持multi streaming太赞了）
   4. bertencodertransformer.h：transformer整体的封装
3. sample：cpp、tensorflow、tensrflow_bert、tensorRT的调用FTF的示例
4. tools：根据参数选择最优的矩阵乘法（GEMM=General Matrix Multiplication）

## 简要总结

encode 公式 https://blog.csdn.net/qq_30614345/article/details/103578545



(https://github.com/bytedance/effective_transformer

## 1、除矩阵乘法外的层融合

功效：减少kernel调用，进而减少global memory 的调用（每一个kernel都会调用global memory）

方法：



![1_1](./pic/1_1.png)

将除矩阵乘法以外的所有kernel都进行了尽可能的融合，单层Transformer的计算流程如下图所示：

![640](./pic/640.webp)

如图所示，Faster Transformer只用了14个kernel就完成了原来将近60个kernel的计算逻辑。这其中，8个kernel是通过调用cuBLAS接口计算矩阵乘法（绿色框），其余6个是自定义kernel （蓝色框）。







```c++
//encoder_sample.cc
	encoder_transformer_->forward();
		//bert_encoder_transformer.h
		attention_->forward();
			//open_attention.h
			forward();
				//1-3
				cublasGemmEx //attr_kernel_Q 
                cublasGemmEx //attr_kernel_K
                cublasGemmEx //attr_kernel_V
                //sqrtf(dk)
                DataType_ scaler = 1 / sqrtf(size_per_head_ * 1.0f);
				//multi-Head attention
      			multiHeadAttr_nofuse_kernelLauncher()
                     //4 open_attention.cu  
                     // add bias
                     add_QKV_bias<DataType_><<<grid, block, 0, stream>>>
                     //5 Q*K^T
                     cublasGemmStridedBatchedEx
                     //6 softmax 
                     //https://zhuanlan.zhihu.com/p/341059988
                     //https://zhuanlan.zhihu.com/p/271740706
                    //https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
                     if(batch_size * head_num <= 120)
						softmax_kernel_v2<DataType_>;
							blockReduceMax
                                warpReduceMax
                            blockReduceSum
                                warpReduceSum
   					 else
                        softmax_kernel<DataType_>;
				     // softmax(xxxx)*V
                     cublasGemmStridedBatchedEx
                     //8 transpose
                     transpose
                //9
                cublasGemmEx
                //10 add bias + layernorm
                add_bias_input_layernorm_kernelLauncher
                     add_bias_input_layernorm
                         // add bias (cuda_kernels.cu 134)
                         
                         // layernorm  
                         //https://blog.csdn.net/liuxiao214/article/details/81037416
                         //https://liumin.blog.csdn.net/article/details/85075706
							blockReduceMax
                         		//并行规约                      	    				                          //https://blog.csdn.net/wujianing_110117/article/details/113363255
                       //https://blog.csdn.net/qq_16097611/article/details/51585379
                                warpReduceMax
                //11
                cublasGemmEx    
                //12 add bias + activation
                add_bias_act_kernelLauncher
                         add_bias_act
                         	gelu
                //13
                cublasGemmEx
                //14 add bias + layernorm
                add_bias_input_layernorm_kernelLauncher


```

![effective_transformer](/home/yaoym/code/3rd_tool/DeepLearningExamples-master/FasterTransformer/v3.1/docs/images/effective_transformer.png)

cublasGemmEx

![image-20210209092501708](/home/yaoym/.config/Typora/typora-user-images/image-20210209092501708.png)

![image-20210209092540149](/home/yaoym/.config/Typora/typora-user-images/image-20210209092540149.png)

CUBLAS_GEMM_DEFAULT_TENSOR_OP 默认可以搜索到最优的参数，特殊情况使用gemm_test进行测试，找到0-15中最优的