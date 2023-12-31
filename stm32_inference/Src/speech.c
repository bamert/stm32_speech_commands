/**
  ******************************************************************************
  * @file    speech.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Thu Dec 28 17:38:45 2023
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "speech.h"
#include "speech_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_speech
 
#undef AI_SPEECH_MODEL_SIGNATURE
#define AI_SPEECH_MODEL_SIGNATURE     "11f6a564cf0fa41c29aaf24072483781"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Thu Dec 28 17:38:45 2023"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_SPEECH_N_BATCHES
#define AI_SPEECH_N_BATCHES         (1)

static ai_ptr g_speech_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_speech_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  _model_Relu_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1792, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _model_pool3_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 448, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv4_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 320, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _model_Relu_3_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 320, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _model_pool4_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _model_global_avg_pool_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _model_fc1_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  output_softmax_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  output_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 36, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv1_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 320, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv1_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv4_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv4_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _model_fc1_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2304, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _model_fc1_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 8000, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv1_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16000, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _model_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16000, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _model_pool1_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4000, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3936, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _model_Relu_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3936, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _model_pool2_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 960, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _model_conv3_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1792, AI_STATIC)
/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _model_Relu_2_output_0_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 28), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_Relu_2_output_0_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _model_pool3_MaxPool_output_0_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 7), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_pool3_MaxPool_output_0_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv4_Conv_output_0_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 5), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_conv4_Conv_output_0_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _model_Relu_3_output_0_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 5), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_Relu_3_output_0_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _model_pool4_MaxPool_output_0_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_pool4_MaxPool_output_0_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _model_global_avg_pool_GlobalAveragePool_output_0_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_global_avg_pool_GlobalAveragePool_output_0_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _model_fc1_Gemm_output_0_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 36, 1, 1), AI_STRIDE_INIT(4, 4, 4, 144, 144),
  1, &_model_fc1_Gemm_output_0_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  output_softmax_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 36, 1, 1), AI_STRIDE_INIT(4, 4, 4, 144, 144),
  1, &output_softmax_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  output_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 36, 1, 1), AI_STRIDE_INIT(4, 4, 4, 144, 144),
  1, &output_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv1_Conv_output_0_weights, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 10, 32), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_conv1_Conv_output_0_weights_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv1_Conv_output_0_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_conv1_Conv_output_0_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv2_Conv_output_0_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 3, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 4096),
  1, &_model_conv2_Conv_output_0_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv2_Conv_output_0_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_conv2_Conv_output_0_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv3_Conv_output_0_weights, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 3, 64), AI_STRIDE_INIT(4, 4, 128, 8192, 8192),
  1, &_model_conv3_Conv_output_0_weights_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv3_Conv_output_0_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_conv3_Conv_output_0_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv4_Conv_output_0_weights, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_model_conv4_Conv_output_0_weights_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv4_Conv_output_0_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_conv4_Conv_output_0_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _model_fc1_Gemm_output_0_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 64, 36, 1, 1), AI_STRIDE_INIT(4, 4, 256, 9216, 9216),
  1, &_model_fc1_Gemm_output_0_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _model_fc1_Gemm_output_0_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 36, 1, 1), AI_STRIDE_INIT(4, 4, 4, 144, 144),
  1, &_model_fc1_Gemm_output_0_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 8000), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &input_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv1_Conv_output_0_output, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 500), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_conv1_Conv_output_0_output_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _model_Relu_output_0_output, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 500), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_Relu_output_0_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _model_pool1_MaxPool_output_0_output, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 125), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_pool1_MaxPool_output_0_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv2_Conv_output_0_output, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 123), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_conv2_Conv_output_0_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _model_Relu_1_output_0_output, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 123), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_Relu_1_output_0_output_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _model_pool2_MaxPool_output_0_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 30), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_model_pool2_MaxPool_output_0_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _model_conv3_Conv_output_0_output, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 28), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_model_conv3_Conv_output_0_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  output_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_softmax_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  output_layer, 17,
  NL_TYPE, 0x0, NULL,
  nl, forward_log,
  &output_chain,
  NULL, &output_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  output_softmax_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_fc1_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_softmax_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  output_softmax_layer, 16,
  NL_TYPE, 0x0, NULL,
  nl, forward_sm,
  &output_softmax_chain,
  NULL, &output_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_fc1_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_global_avg_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_fc1_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_model_fc1_Gemm_output_0_weights, &_model_fc1_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_fc1_Gemm_output_0_layer, 15,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_model_fc1_Gemm_output_0_chain,
  NULL, &output_softmax_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_global_avg_pool_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_pool4_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_global_avg_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_global_avg_pool_GlobalAveragePool_output_0_layer, 13,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &_model_global_avg_pool_GlobalAveragePool_output_0_chain,
  NULL, &_model_fc1_Gemm_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_pool4_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_Relu_3_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_pool4_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_pool4_MaxPool_output_0_layer, 12,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &_model_pool4_MaxPool_output_0_chain,
  NULL, &_model_global_avg_pool_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 4), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 4), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_Relu_3_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_conv4_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_Relu_3_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_Relu_3_output_0_layer, 11,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_model_Relu_3_output_0_chain,
  NULL, &_model_pool4_MaxPool_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_conv4_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_pool3_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_conv4_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_conv4_Conv_output_0_weights, &_model_conv4_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_conv4_Conv_output_0_layer, 10,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_conv4_Conv_output_0_chain,
  NULL, &_model_Relu_3_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_pool3_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_Relu_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_pool3_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_pool3_MaxPool_output_0_layer, 9,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &_model_pool3_MaxPool_output_0_chain,
  NULL, &_model_conv4_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 4), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 4), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_Relu_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_conv3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_Relu_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_Relu_2_output_0_layer, 8,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_model_Relu_2_output_0_chain,
  NULL, &_model_pool3_MaxPool_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_conv3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_pool2_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_conv3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_conv3_Conv_output_0_weights, &_model_conv3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_conv3_Conv_output_0_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_conv3_Conv_output_0_chain,
  NULL, &_model_Relu_2_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_pool2_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_pool2_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_pool2_MaxPool_output_0_layer, 6,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &_model_pool2_MaxPool_output_0_chain,
  NULL, &_model_conv3_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 4), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 4), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_Relu_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_conv2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_Relu_1_output_0_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_model_Relu_1_output_0_chain,
  NULL, &_model_pool2_MaxPool_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_conv2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_pool1_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_conv2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_conv2_Conv_output_0_weights, &_model_conv2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_conv2_Conv_output_0_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_conv2_Conv_output_0_chain,
  NULL, &_model_Relu_1_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_pool1_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_pool1_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_pool1_MaxPool_output_0_layer, 3,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &_model_pool1_MaxPool_output_0_chain,
  NULL, &_model_conv2_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 4), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 4), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_conv1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_Relu_output_0_layer, 2,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_model_Relu_output_0_chain,
  NULL, &_model_pool1_MaxPool_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _model_conv1_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_model_conv1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_model_conv1_Conv_output_0_weights, &_model_conv1_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _model_conv1_Conv_output_0_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_model_conv1_Conv_output_0_chain,
  NULL, &_model_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 16), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 97424, 1, 1),
    97424, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 64288, 1, 1),
    64288, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SPEECH_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SPEECH_OUT_NUM, &output_output),
  &_model_conv1_Conv_output_0_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 97424, 1, 1),
      97424, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 64288, 1, 1),
      64288, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SPEECH_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SPEECH_OUT_NUM, &output_output),
  &_model_conv1_Conv_output_0_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool speech_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_speech_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_output_array.data = AI_PTR(g_speech_activations_map[0] + 32288);
    input_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 32288);
    
    _model_conv1_Conv_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 0);
    _model_conv1_Conv_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 0);
    
    _model_Relu_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 0);
    _model_Relu_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 0);
    
    _model_pool1_MaxPool_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 0);
    _model_pool1_MaxPool_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 0);
    
    _model_conv2_Conv_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 16000);
    _model_conv2_Conv_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 16000);
    
    _model_Relu_1_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 0);
    _model_Relu_1_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 0);
    
    _model_pool2_MaxPool_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 15744);
    _model_pool2_MaxPool_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 15744);
    
    _model_conv3_Conv_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 0);
    _model_conv3_Conv_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 0);
    
    _model_Relu_2_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 7168);
    _model_Relu_2_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 7168);
    
    _model_pool3_MaxPool_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 0);
    _model_pool3_MaxPool_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 0);
    
    _model_conv4_Conv_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 1792);
    _model_conv4_Conv_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 1792);
    
    _model_Relu_3_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 0);
    _model_Relu_3_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 0);
    
    _model_pool4_MaxPool_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 1280);
    _model_pool4_MaxPool_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 1280);
    
    _model_global_avg_pool_GlobalAveragePool_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 0);
    _model_global_avg_pool_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 0);
    
    _model_fc1_Gemm_output_0_output_array.data = AI_PTR(g_speech_activations_map[0] + 256);
    _model_fc1_Gemm_output_0_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 256);
    
    output_softmax_output_array.data = AI_PTR(g_speech_activations_map[0] + 0);
    output_softmax_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 0);
    
    output_output_array.data = AI_PTR(g_speech_activations_map[0] + 144);
    output_output_array.data_start = AI_PTR(g_speech_activations_map[0] + 144);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool speech_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_speech_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _model_conv1_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_conv1_Conv_output_0_weights_array.data = AI_PTR(g_speech_weights_map[0] + 0);
    _model_conv1_Conv_output_0_weights_array.data_start = AI_PTR(g_speech_weights_map[0] + 0);
    
    _model_conv1_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_conv1_Conv_output_0_bias_array.data = AI_PTR(g_speech_weights_map[0] + 1280);
    _model_conv1_Conv_output_0_bias_array.data_start = AI_PTR(g_speech_weights_map[0] + 1280);
    
    _model_conv2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_conv2_Conv_output_0_weights_array.data = AI_PTR(g_speech_weights_map[0] + 1408);
    _model_conv2_Conv_output_0_weights_array.data_start = AI_PTR(g_speech_weights_map[0] + 1408);
    
    _model_conv2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_conv2_Conv_output_0_bias_array.data = AI_PTR(g_speech_weights_map[0] + 13696);
    _model_conv2_Conv_output_0_bias_array.data_start = AI_PTR(g_speech_weights_map[0] + 13696);
    
    _model_conv3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_conv3_Conv_output_0_weights_array.data = AI_PTR(g_speech_weights_map[0] + 13824);
    _model_conv3_Conv_output_0_weights_array.data_start = AI_PTR(g_speech_weights_map[0] + 13824);
    
    _model_conv3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_conv3_Conv_output_0_bias_array.data = AI_PTR(g_speech_weights_map[0] + 38400);
    _model_conv3_Conv_output_0_bias_array.data_start = AI_PTR(g_speech_weights_map[0] + 38400);
    
    _model_conv4_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_conv4_Conv_output_0_weights_array.data = AI_PTR(g_speech_weights_map[0] + 38656);
    _model_conv4_Conv_output_0_weights_array.data_start = AI_PTR(g_speech_weights_map[0] + 38656);
    
    _model_conv4_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_conv4_Conv_output_0_bias_array.data = AI_PTR(g_speech_weights_map[0] + 87808);
    _model_conv4_Conv_output_0_bias_array.data_start = AI_PTR(g_speech_weights_map[0] + 87808);
    
    _model_fc1_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _model_fc1_Gemm_output_0_weights_array.data = AI_PTR(g_speech_weights_map[0] + 88064);
    _model_fc1_Gemm_output_0_weights_array.data_start = AI_PTR(g_speech_weights_map[0] + 88064);
    
    _model_fc1_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _model_fc1_Gemm_output_0_bias_array.data = AI_PTR(g_speech_weights_map[0] + 97280);
    _model_fc1_Gemm_output_0_bias_array.data_start = AI_PTR(g_speech_weights_map[0] + 97280);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_speech_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_SPEECH_MODEL_NAME,
      .model_signature   = AI_SPEECH_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 818760,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_speech_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_SPEECH_MODEL_NAME,
      .model_signature   = AI_SPEECH_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 818760,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_speech_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_speech_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_speech_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_speech_create(network, AI_SPEECH_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_speech_data_params_get(&params) != true) {
        err = ai_speech_get_error(*network);
        return err;
    }
#if defined(AI_SPEECH_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_SPEECH_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_speech_init(*network, &params) != true) {
        err = ai_speech_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_speech_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_speech_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_speech_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_speech_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= speech_configure_weights(net_ctx, params);
  ok &= speech_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_speech_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_speech_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_SPEECH_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

