/**
  ******************************************************************************
  * @file    speech_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    Mon Dec 25 17:59:25 2023
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef SPEECH_DATA_PARAMS_H
#define SPEECH_DATA_PARAMS_H
#pragma once

#include "ai_platform.h"

/*
#define AI_SPEECH_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_speech_data_weights_params[1]))
*/

#define AI_SPEECH_DATA_CONFIG               (NULL)


#define AI_SPEECH_DATA_ACTIVATIONS_SIZES \
  { 64056, }
#define AI_SPEECH_DATA_ACTIVATIONS_SIZE     (64056)
#define AI_SPEECH_DATA_ACTIVATIONS_COUNT    (1)
#define AI_SPEECH_DATA_ACTIVATION_1_SIZE    (64056)



#define AI_SPEECH_DATA_WEIGHTS_SIZES \
  { 106384, }
#define AI_SPEECH_DATA_WEIGHTS_SIZE         (106384)
#define AI_SPEECH_DATA_WEIGHTS_COUNT        (1)
#define AI_SPEECH_DATA_WEIGHT_1_SIZE        (106384)



#define AI_SPEECH_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_speech_activations_table[1])

extern ai_handle g_speech_activations_table[1 + 2];



#define AI_SPEECH_DATA_WEIGHTS_TABLE_GET() \
  (&g_speech_weights_table[1])

extern ai_handle g_speech_weights_table[1 + 2];


#endif    /* SPEECH_DATA_PARAMS_H */
