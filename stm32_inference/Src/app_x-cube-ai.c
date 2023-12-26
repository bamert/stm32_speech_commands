
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\<user_name>\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "speech.h"
#include "speech_data.h"

/* USER CODE BEGIN includes */
#include "arm_math.h"
const char* const speech_classes[] = {
    "background_noise_",
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
};
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_SPEECH_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_SPEECH_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_SPEECH_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_SPEECH_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_SPEECH_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_SPEECH_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_SPEECH_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_SPEECH_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_SPEECH_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle speech = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_speech_create_and_init(&speech, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_speech_create_and_init");
    return -1;
  }

  ai_input = ai_speech_inputs_get(speech, NULL);
  ai_output = ai_speech_outputs_get(speech, NULL);

#if defined(AI_SPEECH_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_SPEECH_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_SPEECH_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_SPEECH_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_SPEECH_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_SPEECH_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_speech_run(speech, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_speech_get_error(speech),
        "ai_speech_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */
int acquire_and_process_data(ai_i8* data[])
{
  /* fill the inputs of the c-model */
  float* float_data = (float*)data[0]; 
  for (int j = 0; j < AI_SPEECH_IN_1_SIZE; j++) {
      float_data[j] = 0.2e-2f; // Fill with float zeros
  }
  return 0;
}
int copy_from_dma_buffer_and_convert(uint16_t* buf, int length) {
  float* float_data = (float*)data_ins[0]; 
  for (int j = 0; j < length; j++) {
      float_data[j] = buf[j]; // Convert to float, preserving scale
  }
  return 0;
}
int start_inference(void){
    // Set the busy flag. Prevents buffer from being overwritten
    // The process method below will scale the inputs and start inference
    // start the inference
    printf("Setting model busy\n\r");
    model_busy = true;
    return 0;
}
float standardize_data(float* data, uint32_t length) {
    float mean, stddev;
    
    // Calculate the mean
    arm_mean_f32(data, length, &mean);

    // Calculate the standard deviation
    arm_std_f32(data, length, &stddev);

    // Standardize the data
    for (uint32_t i = 0; i < length; i++) {
        data[i] = (data[i] - mean) / stddev;
    }
    return stddev;
}
void run_inference(){
    // Do input scaling
    //float* float_data = (float*)data_ins[0]; 
    float stddev = standardize_data(data_ins[0], 8000);
    // Run inference
    ai_run();
    printf("Starting inference stddev %.02f\n\r", stddev);
    int res = ai_run();
    /* 3- post-process the predictions */
    if (res == 0){
        float* output_data = (float*)data_outs[0]; 
        res = post_process(output_data);
        printf("inference complete\n\r");
    }
    model_busy = false;

}

uint32_t VectorMaximum(float* vector){
  ai_float max=-100.;
  uint32_t idx=0;
  
  // find maximum output
  for(int i=0;i<AI_SPEECH_OUT_1_SIZE;i++){
   if(vector[i] > max){
     idx=i;
     max = vector[i];
   }
  }
  return idx;
}
int post_process(float* data)
{
  /* process the predictions
  for (int idx=0; idx < AI_SPEECH_OUT_NUM; idx++ )
  {
      data[idx] = ....
  }

  */
   float maxValue;
   uint32_t maxIndex;

    // Find the maximum value and its index
    //printf("Finding maximum of %i outputs\r\n", AI_SPEECH_OUT_1_SIZE);
    //arm_max_f32(data, AI_SPEECH_OUT_1_SIZE, &maxValue, &maxIndex);
    maxIndex = VectorMaximum(data);
    maxValue = data[maxIndex];
    uint32_t maxScore = (int)(maxValue * 100);
   if  (maxIndex < 0 || maxIndex > AI_SPEECH_OUT_1_SIZE -1 ) {
       printf("Invalid max index\r\n");
       return 0;
   } else if ( maxScore < 40){
       printf("Inference under score threshold: %lu\n\r", maxScore);

   } else {
      printf("score: %lu, index=%lu, class %s\r\n", maxScore, maxIndex, speech_classes[maxIndex], maxValue);
   }

    return 0;
}
/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  printf("\n\rTEMPLATE - initialization\n\r");

  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
  int res = 0;

  printf("TEMPLATE - run - main loop\n");

  int i=0;
  if (speech) {
      for(;;){
          // Do nothing. Dma interrupt directly calls start_inference
          if(model_busy){
            run_inference();
          }
      }
  }

  if (res) {
    ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
    ai_log_err(err, "Process has FAILED");
  }
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif
