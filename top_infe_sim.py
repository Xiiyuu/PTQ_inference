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
 FILE NAME  : top_inference.py
 DEPARTMENT : Computer Architecture Group
 AUTHOR     : zhiwang
 AUTHOR'S EMAIL : zhiwanghuo@stu.xjtu.edu.cn
 ---------------------------------------------------------------------------------------------------------------------
 Ver 1.0  2023-12-20 initial version.
 ---------------------------------------------------------------------------------------------------------------------
"""
# private package
import stat_tools.inference_sim as infe



if __name__ == "__main__":

    infe.inference_sim('resnet18',     'IMAGENET',     chunk_size=4,   pair_size=2,    num_images=1000)
    # infe.inference_sim('resnet18',     'CIFAR10',     chunk_size=4,   pair_size=2,    num_images=1000)
    # infe.inference_sim('mobilenet',    'CIFAR10',     chunk_size=4,   pair_size=2,    num_images=None)
    # infe.inference_sim('shufflenet',   'CIFAR10',     chunk_size=4,   pair_size=2,    num_images=None)
    # infe.inference_sim('inception',    'CIFAR10',     chunk_size=4,   pair_size=2,    num_images=None)
    # infe.inference_sim('densenet',     'CIFAR10',     chunk_size=4,   pair_size=2,    num_images=None)
    # infe.inference_sim('googlenet',    'CIFAR10',     chunk_size=4,   pair_size=2,    num_images=None)

    ### The following models have been run correctly.
    # infe.inference_sim('resnet18',     'CIFAR10',      chunk_size=4,   pair_size=2,    num_images=1000)
    # infe.inference_sim('resnet18',     'IMAGENET',     chunk_size=4,   pair_size=2,    num_images=1000)
    # infe.inference_sim('vgg16',        'IMAGENET',     chunk_size=4,   pair_size=2,    num_images=None)
    # infe.inference_sim('mobilenet',    'IMAGENET',     chunk_size=4,   pair_size=2,    num_images=None)
    # infe.inference_sim('shufflenet',   'IMAGENET',     chunk_size=4,   pair_size=2,    num_images=None)
    # infe.inference_sim('inception',    'IMAGENET',     chunk_size=4,   pair_size=2,    num_images=None)
    # infe.inference_sim('densenet',     'IMAGENET',     chunk_size=4,   pair_size=2,    num_images=None)
    # infe.inference_sim('googlenet',    'IMAGENET',     chunk_size=4,   pair_size=2,    num_images=None)
