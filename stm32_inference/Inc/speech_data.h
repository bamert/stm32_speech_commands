/**
  ******************************************************************************
  * @file    speech_data.h
  * @author  AST Embedded Analytics Research Platform
  * @date    Tue Dec 26 10:06:56 2023
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

#ifndef SPEECH_DATA_H
#define SPEECH_DATA_H
#pragma once

#include "speech_config.h"
#include "speech_data_params.h"

AI_DEPRECATED
#define AI_SPEECH_DATA_ACTIVATIONS(ptr_)  \
  ai_speech_data_activations_buffer_get(AI_HANDLE_PTR(ptr_))

AI_DEPRECATED
#define AI_SPEECH_DATA_WEIGHTS(ptr_)  \
  ai_speech_data_weights_buffer_get(AI_HANDLE_PTR(ptr_))


AI_API_DECLARE_BEGIN


extern const ai_u64 s_speech_weights_array_u64[13298];



/*!
 * @brief Get network activations buffer initialized struct.
 * @ingroup speech_data
 * @param[in] ptr a pointer to the activations array storage area
 * @return an ai_buffer initialized struct
 */
AI_DEPRECATED
AI_API_ENTRY
ai_buffer ai_speech_data_activations_buffer_get(const ai_handle ptr);

/*!
 * @brief Get network weights buffer initialized struct.
 * @ingroup speech_data
 * @param[in] ptr a pointer to the weights array storage area
 * @return an ai_buffer initialized struct
 */
AI_DEPRECATED
AI_API_ENTRY
ai_buffer ai_speech_data_weights_buffer_get(const ai_handle ptr);

/*!
 * @brief Get network weights array pointer as a handle ptr.
 * @ingroup speech_data
 * @return a ai_handle pointer to the weights array
 */
AI_DEPRECATED
AI_API_ENTRY
ai_handle ai_speech_data_weights_get(void);


/*!
 * @brief Get network params configuration data structure.
 * @ingroup speech_data
 * @return true if a valid configuration is present, false otherwise
 */
AI_API_ENTRY
ai_bool ai_speech_data_params_get(ai_network_params* params);


AI_API_DECLARE_END

#endif /* SPEECH_DATA_H */

