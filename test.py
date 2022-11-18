# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import config
import model
from dataset import CUDAPrefetcher, ImageDataset
from utils import load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter
from torchvision.datasets.folder import find_classes
model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def build_model() -> nn.Module:
    resnet_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)
    resnet_model = resnet_model.to(device=config.device, memory_format=torch.channels_last)

    return resnet_model


def load_dataset() -> CUDAPrefetcher:
    test_dataset = ImageDataset("/ssd_scratch/cvit/shaon/ImageNet_test/test",
                                config.image_size,
                                config.model_mean_parameters,
                                config.model_std_parameters,
                                "Test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=5000,
                                 shuffle=True,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return test_prefetcher


def main() -> None:
    # Initialize the model
    resnet_model = build_model()
    print(f"Build `{config.model_arch_name}` model successfully.")

    # Load model weights
    resnet_model, _, _, _, _, _ = load_state_dict(resnet_model, "results/resnet50-Mnist1_6M/last.pth.tar")
    #print(f"Load `{config.model_arch_name}` "
          #f"model weights `{os.path.abspath(config.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    resnet_model.eval()

    # Load test dataloader
    test_prefetcher = load_dataset()

    # Calculate how many batches of data are in each Epoch
    batches = len(test_prefetcher)
    print(batches)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"Test: ")

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()
    datafile_path = "/home2/shaon/predicted_labels_new"
    # Get the initialization test time
    end = time.time()
    count = 1
    image_dir = "/ssd_scratch/cvit/shaon/ImageNet_test/test"
    _,class_to_idx = find_classes(image_dir)
    a = list(np.arange(0,10))
    b = list(np.arange(65,91))
    c = list(np.arange(97,123))
    d = a+b+c
    d.append("empty")
    dicts = {}
    keys = range(63)
    
    for i in keys:
        if i< 10:
            dicts[i] = str(d[i])
        elif i == 62:
            dicts[i] = d[i]
        else:
            dicts[i] = chr(d[i])


    with torch.no_grad():
        while batch_data is not None:
            map_list = []
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, non_blocking=True)
            target = batch_data["target"].to(device=config.device, non_blocking=True)
            image_name = batch_data["path"]
            # Get batch size
            batch_size = images.size(0)

            # Inference
            output = resnet_model(images)
            pred = F.softmax(output,dim = 1)
            pred_label = torch.argmax(pred, dim = 1)
            pred_label = pred_label.detach().cpu().numpy()
            pred_label = list(pred_label)

            keys = list(class_to_idx.keys())
            vals = list(class_to_idx.values())

            for i in range(0, len(pred_label)):
                map_val = keys[vals.index(pred_label[i])]
                results = dicts.get(int(map_val))
                line = image_name[i]+"----->"+str(map_val)+"---->"+results+ "\n"
                map_list.append(line)
            filename = str(count)+ ".txt"
            fp = open(os.path.join(datafile_path,filename),"w")
            fp.writelines(map_list)
            fp.close()


            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5))
            acc1.update(top1[0].item(), batch_size)
            acc5.update(top5[0].item(), batch_size)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            #if batch_index % config.test_print_frequency == 0:
            progress.display(batches)

            # Preload the next batch of data
            batch_data = test_prefetcher.next()

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1
            count+=1
            image_name = 0
            
    print(count)

    # print metrics
    print(f"Acc@1 error: {100 - acc1.avg:.2f}%")
    print(f"Acc@5 error: {100 - acc5.avg:.2f}%")


if __name__ == "__main__":
    main()
