import os
import time
import torch
import torch.quantization
import torchvision
import torchvision.transforms as transforms



# dataset
imagenet_path       = '/home/zhiwang/dataset/ILSVRC2012/val'
coco_path           = ''
dataset_path        = '/home/zhiwang/projects/data'

# parameters dictionary
model_cifar10_path  = '/home/zhiwang/projects/data/cifar-10-pth/'
model_cifar100_path = '/home/zhiwang/projects/data/cifar-100-pth/'
model_coco_path     = ''






def evaluate_model(model, dataloader, device, model_name, project_path, num_images=None):
    time_start = time.time()

    model.eval()
    correct = 0
    total = 0
    count = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            count += len(images)
            if num_images is not None and count >= num_images:
                break
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %d %%' % accuracy)
    
    time_end = time.time()
    print('time: ', time_end-time_start)

    model_shifting_infe_info    = os.path.join(project_path, 'stat-result')
    model_shift_result_file     = os.path.join(model_shifting_infe_info, 'model-shift-result.txt')

    with open(model_shift_result_file, 'a') as model_info:
        model_info.write(f"{model_name}\n\n")
        model_info.write(f"The execution time is {time_end - time_start}\n")
        model_info.write(f"The correct number is {correct}\n")
        model_info.write(f"The accuracy is {100 * correct / total}\n\n")



def model_set(model_name='resnet18', dataset_name='IMAGENET', device='cpu'):
    # 1. Select model
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
    elif 'mobilenet' in model_name:
        model = torchvision.models.mobilenet_v2(pretrained=True)
    # 不一定非得使用torchvision自带的网络模型，可以单独建一个文件夹存放模型
    elif 'shufflenet' in model_name:
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif 'inception'in model_name:
        model = torchvision.models.inception_v3(pretrained=True)
    elif 'efficientnet' in model_name:
        model = torchvision.models.efficientnet_b0(pretrained=True)
    elif 'densenet'in model_name:
        model = torchvision.models.densenet121(pretrained=True)
    elif 'googlenet'in model_name:
        model = torchvision.models.googlenet(pretrained=True)
    else:
        raise ValueError("Unsupported model. Supported models are: 'resnet18', 'vgg16', \
            'mobilenet', 'shufflenet', 'inception', 'efficientnet', 'densenet', 'googlenet'.")
    
    # 2. Select dataset and reshape IO layers
    model, input_size = reshape_IO_layers(model, model_name, dataset_name)

    # 3. Modify model for the specific dataset
    model.to(device)
    # Define your dataset and dataloader for calibration and evaluation
    if dataset_name == 'IMAGENET':
        transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        test_dataset = torchvision.datasets.ImageFolder(root=imagenet_path, transform=transform)
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ])
        test_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    # TODO: 设置transfrom参数，设置错误会对准确率有影响
    # elif dataset_name == 'CIFAR100':
    #     transform = transforms.Compose([transforms.Resize((input_size, input_size)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                     ])
    #     test_dataset = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transform)
    # elif dataset_name == 'COCO':
    #     transform = transforms.Compose([transforms.Resize((input_size, input_size)),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                     ])
    #     test_dataset = torchvision.datasets.CocoDetection(root='path_to_coco_dataset', annFile='path_to_annotations', transform=transform)
    else:
        raise ValueError("Unsupported dataset. Supported datasets are: 'CIFAR10', 'CIFAR100', 'IMAGENET', 'COCO'.")
    
    test_dataloader     = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    return model, transform, test_dataloader





def reshape_IO_layers(model, model_name, dataset_name):
    
    # 1. size of different dataset
    if dataset_name == 'IMAGENET':
        input_size = 224
        num_classes = 1000
    elif dataset_name == 'CIFAR10':
        input_size = 32
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        input_size = 32
        num_classes = 100
    elif dataset_name == 'COCO':
        input_size = 224
        num_classes = 80
    
    if dataset_name is not 'IMAGENET':  # IMAGENET is the default dataset
        ## 2. reshape output layers of different models
        modify_model(model, input_size, num_classes)
        ## 3. load fp32 pth
        if dataset_name == 'CIFAR10':
            model.load_state_dict(torch.load(f'{model_cifar10_path}{model_name}.pt'))
        elif dataset_name == 'CIFAR100':
            model.load_state_dict(torch.load(f'{model_cifar100_path}{model_name}.pth'))
        elif dataset_name == 'COCO':
            model.load_state_dict(torch.load(f'{model_coco_path}{model_name}.pth'))

    return model, input_size



def modify_model(model, input_size, num_classes):
    if isinstance(model, torchvision.models.ResNet):
        # Modify the input layer and the last fully connected layer of ResNet
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif isinstance(model, torchvision.models.VGG):
        # Modify the input layer and the last fully connected layer of VGG
        model.features[0] = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif isinstance(model, torchvision.models.MobileNetV2):
        # Modify the input layer and the last fully connected layer of MobileNetV2
        model.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    
    # （1）resnet, vgg已经跑通，且准确率正确。
    #  mobilenet跑通，但准确率小于10%，不太确定原因

    # TODO: 依据不同数据集修改模型部分层的尺寸
    # （2）以下5个网络按照GPT建议修改后，依然无法跑通，报错原因是模型参数和模型结构不匹配
    # elif isinstance(model, torchvision.models.shufflenetv2.ShuffleNetV2):
    #     # Modify the input layer and the last fully connected layer of ShuffleNetV2
    # elif isinstance(model, torchvision.models.Inception3):
    #     # Modify the input layer and the last fully connected layer of Inception
    # elif isinstance(model, torchvision.models.EfficientNet):
    #     # Modify the input layer and the last fully connected layer of EfficientNet
    # elif isinstance(model, torchvision.models.densenet.DenseNet):
    #     # Modify the input layer and the last fully connected layer of DenseNet
    # elif isinstance(model, torchvision.models.googlenet.GoogLeNet):
    #     # Modify the input layer and the last fully connected layer of GoogLeNet
    # else:
    #     raise ValueError("Unsupported model. Supported models are: 'ResNet', 'VGG', \
    #         'MobileNetV2', 'ShuffleNetV2', 'Inception3', 'EfficientNet', 'DenseNet',\
    #         'GoogLeNet'.")



def get_dataset(dataset_name, transform, train=True):
    if dataset_name == 'CIFAR10':
        return torchvision.datasets.CIFAR10(root=dataset_path, train=train, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        return torchvision.datasets.CIFAR100(root=dataset_path, train=train, download=True, transform=transform)
    elif dataset_name == 'IMAGENET':
        return torchvision.datasets.ImageFolder(root=imagenet_path, transform=transform)
    # # TODO: 需要下载COCO数据集，或者使用你毕设中的检测数据集也可以
    # elif dataset_name == 'COCO':  # You should set coco_path and path_to_annotations
    #     return torchvision.datasets.CocoDetection(root=coco_path, annFile='path_to_annotations', transform=transform)
    else:
        raise ValueError("Unsupported dataset. Supported datasets are: 'CIFAR10', 'CIFAR100', 'IMAGENET', 'COCO'.")




def compare_state_dicts(dict1, dict2):
    # 检查两个字典的键是否相同
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    # 检查是否有键只出现在一个字典中
    only_in_dict1 = keys1 - keys2
    only_in_dict2 = keys2 - keys1
    if only_in_dict1:
        print("Keys only in dict1:", only_in_dict1)
    if only_in_dict2:
        print("Keys only in dict2:", only_in_dict2)

    # 检查共同的键对应的值是否相等
    common_keys = keys1.intersection(keys2)
    for key in common_keys:
        val1 = dict1[key]
        val2 = dict2[key]
        if val1 is not None and val2 is not None:
            if not torch.equal(val1, val2):
                print("Values for key '{}' are different.".format(key))
            # measure the degree of difference
            if not torch.allclose(val1, val2, atol=1e-25):
                print(f"Parameter {key} is different.")

            # if isinstance(val1, tuple) and isinstance(val2, tuple):
            #     for t1, t2 in zip(val1, val2):
            #         if t1 != t2:
            #             print("Values for key '{}' are different.".format(key))
            # elif isinstance(val1, torch.dtype) and isinstance(val2, torch.dtype):
            #     continue
            # # elif val1.data is not None and val2.data is not None:
            # else:
            #     if not torch.equal(val1, val2):
            #         print("Values for key '{}' are different.".format(key))
    if only_in_dict1 or only_in_dict2:
        print("Parameter dictionaries are not identical.")
    else:
        print("Parameter dictionaries are identical.")
