"""
 ---------------------------------------------------------------------------------------------------------------------
 Copyright (c) 1986-2024, CAG(Computer Architecture Group), Institute of AI and Robotics, Xi'an Jiaotong University.
 Proprietary and Confidential All Rights Reserved.
 ---------------------------------------------------------------------------------------------------------------------
 NOTICE: All information contained herein is, and remains the property of CAG, Institute of AI and Robotics,  Xi'an
 Jiaotong University. The intellectual and technical concepts contained herein are proprietary to CAG team, and may be
 covered by P.R.C. and Foreign Patents, patents in process, and are protected by trade secret or copyright law.# 
 This work may not be copied, modified, re-published, uploaded, executed, or distributed in any way, in any time, in
 any medium, whether in whole or in part, without prior written permission from CAG, Institute of AI and Robotics,
 Xi'an Jiaotong University.# 
 The copyright notice above does not evidence any actual or intended publication or disclosure of this source code,
 which includes information that is confidential and/or proprietary, and is a trade secret of CAG.
 ---------------------------------------------------------------------------------------------------------------------
 FILE NAME  : inference_sim.py
 DEPARTMENT : Computer Architecture Group
 AUTHOR     : zhiwang
 AUTHOR'S EMAIL : zhiwanghuo@stu.xjtu.edu.cn
 ---------------------------------------------------------------------------------------------------------------------
 Ver 1.0  2024-03-17 initial version.
 ---------------------------------------------------------------------------------------------------------------------
"""
# pubilic library
import torch
import copy

# private library
from . import model_tools
from . import quant_tools




def inference_sim(model_name, dataset_name, chunk_size, pair_size, num_images=1000):

    # [Sources]
    project_path        = '/home/zhiwang/projects/shifting-cifar10/' + 'c' + str(chunk_size) + 'p' + str(pair_size)
    info_path           = dataset_name  + '-c' + str(chunk_size) + 'p' + str(pair_size)
    # [Simulation]
    model_int8_pth      = project_path + '/stat-result/stat-pth/' + info_path + '/model-int8/'

    # [Simulation] inference
    device      = torch.device("cpu") # ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. Generate model_fp32 and test dataset
    model, transform, test_dataloader = model_tools.model_set(model_name, dataset_name, device)
    model_fp32 = copy.deepcopy(model)
    fp32_dict = model_fp32.state_dict()
    # Evaluate the original model
    print("Float32 model:")
    model_tools.evaluate_model(model_fp32, test_dataloader, device, model_name+'_fp32', project_path, num_images)
    
    
    # 2. Generate model_int8 base on PTQ
    model_int8 = quant_tools.model_ptsq_quant(model, model_name, dataset_name, 
                                              model_int8_pth, transform, device='cpu', num_images=1000)
    
    
    ptq_state_dict      = torch.load(f'{model_int8_pth}{dataset_name}_{model_name}_int8.pth')
    dequantize_dict_int8= quant_tools.dequantize_model(model.state_dict(), ptq_state_dict)

    # Evaluate the quantized model
    print("Quantized model:")
    model.load_state_dict(dequantize_dict_int8)
    model_tools.evaluate_model(model, test_dataloader, device, model_name+' int8', project_path, num_images)

    model_tools.compare_state_dicts(fp32_dict, dequantize_dict_int8)

    # Evaluate the quantized model
    print("Quantized model:")
    model_tools.evaluate_model(model, test_dataloader, device, model_name+' int8', project_path, num_images)
