import torch
import args_config
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os, time
import torch.backends.cudnn as cudnn
import dill
import pickle

from quant_net import *
from quant_resnet import *
from training_utils import *
import tracemalloc
import math
import gc

"""
Main training script: builds the dataset, model, and optimizer from command-line arguments, 
then runs the train/test loop, saving the best checkpoint.
"""
def main():

    torch.manual_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    args = args_config.get_args()
    print("********** SNN simulation parameters **********")
    print(args)

    # ── Dataset ───────────────────────────────────────────────────────────────
    #Build train/test loaders for specified datasets
    if args.dataset == 'cifar10':
        # CIFAR-10: 32x32 images, 10 classes
        # augmentation: random crop + horizontal flip
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=True,
            transform=transform_train,
            download=True)
            
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=False,
            transform=transform_test,
            download=True)

        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True, 
            num_workers=4,
            pin_memory=True)
        
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False, 
            num_workers=4,
            pin_memory=True)

        num_classes = 10

    elif args.dataset == 'svhn':
    # SVHN: 32x32 street house number images, 10 classes
    # Augmentation: random crop + horizontal flip
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.SVHN(
            root=args.dataset_dir,
            split='train',
            transform=transform_train,
            download=True)
        test_dataset = torchvision.datasets.SVHN(
            root=args.dataset_dir,
            split='test',
            transform=transform_test,
            download=True)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True, 
            num_workers=0,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False, 
            num_workers=0,
            pin_memory=True)

        num_classes = 10
    
    elif args.dataset == 'tiny':
        # TinyImageNet: 64x64 images, 200 classes
        # Augmentation: random rotation + horizontal flip.
        traindir = os.path.join('/gpfs/gibbs/project/panda/shared/tiny-imagenet-200/train')
        valdir = os.path.join('/gpfs/gibbs/project/panda/shared/tiny-imagenet-200/val')
            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                  std=[0.229, 0.224, 0.225])
        train_transforms = transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])

        train_dataset = torchvision.datasets.ImageFolder(traindir, train_transforms)
        test_dataset = torchvision.datasets.ImageFolder(valdir, test_transforms)

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,pin_memory=True)

        num_classes = 200
    elif args.dataset == 'dvs':
        # DVS (CIFAR10-DVS): event-based 2-channel frames, 10 classes, 
        # no augmentation, pre-processed into tensors
        train_dataset_dvs=torch.load("./train_dataset_dvs_8.pt",pickle_module=dill)
        test_dataset_dvs=torch.load("./test_dataset_dvs_8.pt",pickle_module=dill)

        train_data_loader = torch.utils.data.DataLoader(train_dataset_dvs,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset_dvs,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True)
        num_classes = 10
        # print(type(train_dataset_dvs))
        # print(len(train_dataset_dvs))
        # print(train_dataset_dvs[0])

        # print(type(test_dataset_dvs))
        # print(len(test_dataset_dvs))
        # print(test_dataset_dvs[0])
        # exit()

        # # check the test and train sets 
        # train_indices = set(train_dataset_dvs.indices)
        # test_indices = set(test_dataset_dvs.indices)

        # # The intersection should be an empty set, if they have no common elements
        # common_indices = train_indices.intersection(test_indices)
        # print(f"Common indices between train and test datasets: {common_indices}")
        # exit()
        
    # ── Model ─────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    if args.arch == 'vgg16':
        model = Q_ShareScale_VGG16(args.T,args.dataset).cuda()
    elif args.arch == 'vgg9':
        model = Q_ShareScale_VGG9(args.T,args.dataset).cuda()
    elif args.arch == 'res19':
        model = ResNet19(num_classes, args.T).cuda()
    # print(model)
    
    # ── Optimizer and scheduler ───────────────────────────────────────────────
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, 0.9, weight_decay=5e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,weight_decay=1e-4)
    else:
        print ("Current does not support other optimizers other than sgd or adam.")
        exit()
    
    # Cosine annealing: smoothly decays lr from args.lr to 0 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= args.epoch, eta_min= 0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch)

    #── Training loop ─────────────────────────────────────────────────────────
    best_accuracy = 0
    # tracemalloc.start()
    for epoch_ in range(args.epoch):
        # snap1 = tracemalloc.take_snapshot()
        # time1 = time.time()
        loss = 0
        accuracy = 0
        
        loss = train(args, train_data_loader, model, criterion, optimizer, epoch_)

        accuracy= test(model, test_data_loader, criterion)

        scheduler.step()
        # time2 = time.time()
        # print("Training time for one epoch: ", time2-time1)
        
        # Save checkpoint whenever a new best accuracy is achieved
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkdir(f"{os.getcwd()}/model_dumps/{args.arch}/{args.dataset}/{args.rst}/T10/baseline")
            torch.save(model, f"{os.getcwd()}/model_dumps/{args.arch}/{args.dataset}/{args.rst}/T10/baseline/final_dict.pth.tar")
            

        if (epoch_+1) % args.test_display_freq == 0:
            print(f'Train Epoch: {epoch_}/{args.epoch} Loss: {loss:.6f} Accuracy: {accuracy:.3f}% Best Accuracy: {best_accuracy:.3f}%')
            
        # gc.collect()
        # snap2 = tracemalloc.take_snapshot()
        # top_stats=snap1.compare_to(snap2, "lineno")
        # for stat in top_stats[:50]:
        #     line = str(stat)
        #     if("muless-int-snn" in line):
        #         print(line)

"""
Training loop
"""
def train(args, train_data, model, criterion, optimizer, epoch):
    model.train()

    for batch_idx, (imgs, targets) in enumerate(train_data):
        train_loss = 0.0
        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()

        # Forward pass: model returns a list of outputs, one per timestep
        output = model(imgs)

        # Average the cross-entropy loss across all timesteps.
        train_loss = sum([criterion(s, targets) for s in output]) / args.T
        
        train_loss.backward()
        
        # ── MINT gradient scaling for beta ────────────────────────────────────
        # scale beta's gradient with 1/sqrt(Nw * s(n))
        # Nw: nr of weights in layer
        # s(n): 2^(n-1)-1
        if args.share:
            for m in model.modules():
                if isinstance(m,QConvBN2dLIF):
                    # print(m.scaling.grad)
                    m.beta[0].grad.data = m.beta[0].grad/math.sqrt(torch.numel(m.conv_module.weight)*(2**(m.num_bits_w-1)-1))
                elif isinstance(m,QConvBN2d):
                    m.beta[0].grad.data = m.beta[0].grad/math.sqrt(torch.numel(m.conv_module.weight)*(2**(m.num_bits_w-1)-1))
        # for a in model.alpha_list:
        #     a.grad.data = a.grad/1000
        optimizer.step()
   
    return train_loss.item()

if __name__ == '__main__':
    main()


