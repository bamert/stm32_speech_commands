
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

static volatile bool model_busy;
volatile static uint32_t lastChunkTime = 0;

static volatile int write_offset=0;
static volatile int read_offset=0;
static volatile bool buffer_filled=false;
#define RINGBUFFER_SIZE AI_SPEECH_IN_1_SIZE
#define INFERENCE_PERIOD RINGBUFFER_SIZE / 4
static volatile int num_samples_until_inference=RINGBUFFER_SIZE; 
volatile static int16_t ringbuffer[RINGBUFFER_SIZE] __attribute__((aligned(4)));
int post_process(int inference_time, float volume_stddev);
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
int double_buffer_chunk(int16_t* buf, uint32_t length) {
  uint32_t current = HAL_GetTick();
  uint32_t chunkDuration = current - lastChunkTime;
  lastChunkTime = current;
  //printf("Last Audio Chunk %u ms ago. This chunk %u elements \r\n", chunkDuration, length);
 
  
  if (RINGBUFFER_SIZE % length != 0) {
      printf("Chunk size needs to be multiple of ringbuffer size\r\n");
      return 1;
  }
  if (num_samples_until_inference % length != 0) {
      printf("Chunk size needs to be multiple of num_samples_until_inference\r\n");
      return 1;
  }
  // Copy new contents into write offset
  memcpy(&ringbuffer[write_offset], buf, length * sizeof(int16_t));
  // Advance write pointer
  write_offset += length;
  if (write_offset == RINGBUFFER_SIZE) {
      write_offset = 0;
  }
  num_samples_until_inference-=length;
  // Once buffer filled for the first time, start doing overlapping
  // inferences on arrival of each 2000 elements (~250ms of audio).
  if (num_samples_until_inference == 0) {
    int length_1 = RINGBUFFER_SIZE-read_offset;
    int length_2 = RINGBUFFER_SIZE-length_1;
    // Copy first segment to target & convert to float
    float* target_ptr = (float*)(ai_input[0].data);
    for (uint32_t j = 0; j < length_1; j++) {
          target_ptr[j] = (float) ringbuffer[read_offset+j];
    }
    // Copy second segment (with wraparound) to target & convert to float
    for (uint32_t j = 0; j < length_2; j++) {
          target_ptr[length_1+j] = (float) ringbuffer[j];
    }
    // Advance readoffset by 
    read_offset+= INFERENCE_PERIOD; 
    num_samples_until_inference = INFERENCE_PERIOD;
    if (model_busy){
          // Ignore new data while inference is running
          printf("New data ignored due to slow inference\r\n");
          return 1;
    }
    start_inference();
  }
  if (read_offset == RINGBUFFER_SIZE) {
      read_offset = 0;
  }
  return 0;

}
int start_inference(void){
    // Set the busy flag. Prevents buffer from being overwritten
    // The process method below will scale the inputs and start inference
    // start the inference
    model_busy = true;
    return 0;
}
float standardize_data(float* data, uint32_t length) {
    float mean, stddev;
    
    // Calculate the mean
    arm_mean_f32(data, length, &mean);

    // Calculate the standard deviation
    // arm_std_f32(data, length, &stddev);
    // Fix downscaling to match nearby speaker
    // Avoid unnecessarily upscaling noise
    stddev = 250.;

    // Standardize the data
    for (uint32_t i = 0; i < length; i++) {
        data[i] = (data[i] - mean) / stddev;
    }
    return stddev;
}
void run_inference(){
    // Do input scaling
    float* input_ptr = (float*)(ai_input[0].data);
    if (input_ptr == NULL){
        printf("ERROR. input data pointer has not been setup\r\n");
    }
    float volume_stddev = standardize_data(&input_ptr[0], 8000);
    // Run inference
    uint32_t start = HAL_GetTick();
    int res = ai_run();
    uint32_t inference_time_ms = HAL_GetTick() - start;
    /* 3- post-process the predictions */
    if (res == 0){
        res = post_process(inference_time_ms, volume_stddev);
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
int compareFloats(const void * a, const void * b) {
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}
void findSecondLargest(const float array[], int size, int maxIndex, float *secondLargest) {
    *secondLargest = FLT_MIN;

    for (int i = 0; i < size; ++i) {
        if (i != maxIndex && array[i] > *secondLargest) {
            *secondLargest = array[i];
        }
    }
}
int post_process(int inference_time, float volume_stddev)
{
   float* outdat = (float*)(ai_output[0].data);
   float maxScore;
   uint32_t maxIndex;

   // Softmax probabilities
   for (int i = 0; i < AI_SPEECH_OUT_1_SIZE; i++) {
      outdat[i] = exp(outdat[i]);
   }
   // Find maximum probability and its index
   arm_max_f32(outdat, AI_SPEECH_OUT_1_SIZE, &maxScore, &maxIndex);
   
   // Find second largest probability
   float secondLargestScore;
   findSecondLargest(outdat, AI_SPEECH_OUT_1_SIZE, maxIndex, &secondLargestScore);
   float certaintyMargin = maxScore - secondLargestScore;

   if  (maxIndex < 0 || maxIndex > AI_SPEECH_OUT_1_SIZE -1 ) {
       printf("Invalid max index\r\n");
       return 0;
   } else if ( certaintyMargin > 0.8){
      printf("Prediction: class %s, score %0.2f, inference time %u ms\r\n", speech_classes[maxIndex], maxScore, inference_time);
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
