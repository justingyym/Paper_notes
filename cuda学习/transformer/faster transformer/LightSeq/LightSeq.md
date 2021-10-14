c# 

## 相关资料：

详细介绍 https://zhuanlan.zhihu.com/p/269478459





相同 softmax 相同（ReduceSum  ReduceMax)



encoder 代码流程

```c++
read_batch_tokenids_from_file

encoder_->run_one_infer(batch_size, batch_seq_len);

					ker_enc_embedding_launcher	/   ker_multilg_enc_emb_launcher ;

					self_attention();

  				    ffn_add_norm();

					ker_norm_layer_launcher // last layer norm
```

```c++
//transformerKernels.cc
ker_enc_embedding_launcher ->ker_enc_embedding ;

//encoder.cc.cu
self_attention();  
	  /* ---step 0. layer_norm, add output_bias to "query"--- */
	ker_norm_layer_resual_launcher
    /* ---step 1. qkv = ori_q * qkv_wei + bias, and reshape qkv for multi-head
   * gemm--- */
    cublasGemmEx
    // get q, k, v by split and reshape qkv
    ker_arrange_encself_qkv_launcher
     /* ---step 2. correlation = q * k, perform softmax on correlation--- */ 
    cublasGemmStridedBatchedEx  
        
    ker_correlation_softmax_encself_launcher
    /* ---step 3. new_q = correlation * v--- */ 
    cublasGemmStridedBatchedEx
        
    // use v to save reshaped q, since they are in same size and v
    // will not be use again before the next multi-head-attention
    ker_arrange_atten_output_launcher
        
    /* ---step 4. new_q = ori_q + new_q * output_wei--- */ //z*w^0
    cublasGemmEx;
        
ffn_add_norm();
	/* ---step 0. layer_norm, add output_bias to "query"--- */
	ker_norm_layer_resual_launcher       
  /* ---step 1. first ffn layer--- */
    cublasGemmEx
    if (_tw._use_gelu) {
    	ker_bias_gelu_launcher<_DataType>();
    } else {
    	ker_bias_relu_launcher<_DataType>();
    }
  /* ---step 2. second ffn layer--- */
	cublasGemmEx
// last layer norm
ker_norm_layer_launcher
        
```



decoder 代码流程

```c++
Decoder->run_one_infer(batch_size, batch_seq_len);
						/* ---step1. init--- */
						/* ---step2. autoregressive decoding--- */
						run_step
                        /* ---step3. output the decoding result--- */
                        ker_write_topk_result   
```

```c++
//decoder.cc.cu
run_step();  
	embedding();

    decoder_stack();

	/* --- Project hidden states to vocab logits--- */
	cublasGemmEx		
        
    if (_tw._sampling_method == "topk") {
    return sample(); {ker_topk_sample_launcher / ker_topp_sample_launcher}
  } else if (_tw._sampling_method == "topp") {
    return sample();
  } else if (_tw._sampling_method == "topk_greedy") {
    return topk_greedy_search();
  } else if (_tw._sampling_method == "beam_search") {
    return beam_sea  rch();
  } else {
    throw std::runtime_error("not supported sampling_method");
  } 

   
        
Decoder<OpType_>::embedding()
    ker_multilg_dec_emb_launcher / ker_dec_embedding_launcher

    
void Decoder<OpType_>::decoder_stack() 
    self_attention();

    encdec_attention();
          /* ---step 0. layer_norm, add output_bias to "query"--- */
                ker_norm_layer_resual_launcher
          /* ---step 1. new_q = ori_q * q_wei + bias, reshape new_q for multi-head
           * gemm--- */
                cublasGemmEx
                ker_arrange_encdec_q_launcher
          /* ---step 2. correlation = q * k, perform softmax on correlation--- */
               cublasGemmStridedBatchedEx 
               ker_correlation_softmax_encdec_launcher
          /* ---step 3. new_q = correlation * v--- */
               cublasGemmStridedBatchedEx
               ker_arrange_atten_output_launcher
          /* ---step 4. new_q = ori_q + new_q * output_wei--- */
       			cublasGemmEx;
            
    ffn_add_norm();


    
```



