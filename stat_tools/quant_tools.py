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
 FILE NAME  : quant_tools.py
 DEPARTMENT : Computer Architecture Group
 AUTHOR     : zhiwang
 AUTHOR'S EMAIL : zhiwanghuo@stu.xjtu.edu.cn
 ---------------------------------------------------------------------------------------------------------------------
 Ver 1.0  2023-12-20 initial version.
 ---------------------------------------------------------------------------------------------------------------------
"""
import os
import torch

from . import model_tools




# [Simulation]
def dequantize_model(dequantized_state_dict, quantized_state_dict):
    """
    用于推理验证
    """
    for key in quantized_state_dict:
        if 'weight' in key:
            weight      = quantized_state_dict[key]
            dequantized_state_dict[key] = torch.dequantize(weight)
    return dequantized_state_dict


def calibrate_model(model, dataloader, device, num_images=1000):
    model.eval()
    count = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            _ = model(data)
            count += len(data)
            if count >= num_images:
                break



def model_quant(model, dataloader, device, num_images=1000):

    model.eval()

    # 1. Insert stubs
    # model = add_quant_dequant(model)

    # 2. Specify quantization configuration for static quantization
    backend = 'fbgemm'
    # backend = 'qnnpack'
    torch.backends.quantized.engine = backend
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    model.qconfig = torch.quantization.QConfig(
          activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), 
          weight=torch.quantization.default_weight_observer)
    
    # 3. Fuse
    # model             = torch.quantization.fuse_modules(model, [['conv', 'relu']])
    # modules_to_fuse   = get_fuseable_modules(model)
    # model             = torch.quantization.fuse_modules(model, modules_to_fuse) #, inplace=True)
    model_prepare   = torch.quantization.prepare(model) #, inplace=False)
    
    # 4. Calibrate the model with the dataset
    calibrate_model(model, dataloader, device, num_images)
    
    # 5. Convert the model to a quantized version
    model_int8 = torch.quantization.convert(model_prepare) #, inplace=False)
    return model_int8



def model_ptsq_quant(model, model_name, dataset_name, model_int8_pth, transform, device, num_images=1000):
    train_dataset   = model_tools.get_dataset(dataset_name, transform, train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    # Quantize the model statically
    quantized_model = model_quant(model, train_dataloader, device, num_images)
    # Save the quantized model with model name
    if not os.path.exists(model_int8_pth):
        os.makedirs(model_int8_pth)
    torch.save(quantized_model.state_dict(), f'{model_int8_pth}{dataset_name}_{model_name}_int8.pth')

    return quantized_model


