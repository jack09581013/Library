import os
import traceback
import datetime
from colorama import Style, Fore
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import jacky_utils
from config import Config


def get_config():
    return Config().set_mode('train_model_7gpu')


class ExampleDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        x = None  # torch.Tensor (B, C, ...)
        y = None  # torch.Tensor (B, C, ...)
        info = {}
        return x, y, info


def main():
    torch.backends.cudnn.benchmark = True
    config = get_config()

    if config.multi_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.ddp_cuda_devices
        # world_size = torch.cuda.device_count()  # number of gpu
        world_size = config.get_multi_gpu_device_count()  # number of gpu
        assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"
        run_ddp(train_model, world_size, config)

    else:
        train_model(None, None, config)


def train_model(rank, world_size, config):
    is_print = (config.multi_gpu and rank == 0) or not config.multi_gpu
    version, model = config.profile.load_model(rank)

    if config.multi_gpu:
        print(f'Model distributed on cuda:{rank}')
        setup(config, rank, world_size)
        dist.barrier()
        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=config.find_unused_parameters)
        config.profile.set_used_model(ddp_model)
        device = 'cuda:' + str(rank)
    else:
        config.profile.set_used_model(model)
        device = config.device

    config.profile.init_losses(device)
    exception_count = 0

    train_dataset = ExampleDataset()
    test_dataset = ExampleDataset()

    if config.multi_gpu:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=True)

    optimizer = Adam(params=model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))

    if is_print:
        number_of_parameters = sum(p.numel() for p in model.parameters())
        print(f'Batch size: {config.batch_size}')
        print(f'Number of parameters: {number_of_parameters:,}')
        print(f'Model size: {number_of_parameters / 1024 / 1024 * 4:.2f} MB')
        print(f'Model name: {config.profile}')
        print('Number of training data:', len(train_dataset))
        print('Number of testing data:', len(test_dataset))
        print()

    current_version = version + 1
    while current_version < config.max_version + 1:
        try:
            epoch_start_time = datetime.datetime.now()
            loss_history = config.profile.load_history(
                current_version - 1 if current_version > 1 else None)[1]

            if is_print and not config.is_debug:
                print('Exception count:', exception_count)

            if config.multi_gpu:
                train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=config.batch_size,
                                          num_workers=config.num_workers, pin_memory=True, drop_last=True)

                test_loader = DataLoader(dataset=test_dataset, sampler=test_sampler, batch_size=config.batch_size,
                                         num_workers=config.num_workers, pin_memory=True, drop_last=True)
            else:
                train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                                          num_workers=config.num_workers, pin_memory=True, drop_last=True)

                test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=True,
                                         num_workers=config.num_workers, pin_memory=True, drop_last=True)

            train_loss = []
            test_loss = []
            if is_print:
                print(f'Start training, version = {current_version}/{config.max_version}')
                tbar = tqdm(train_loader)
            else:
                tbar = train_loader

            model.train()
            for batch_index, (X, Y, dataset_info) in enumerate(tbar):
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)

                optimizer.zero_grad()                
                total_loss = config.profile.train_model(X, Y, dataset_info, current_version)
                total_loss.backward()
                optimizer.step()

                train_loss.append(float(total_loss))
                if is_print:
                    tbar.set_description(f'train loss = {Fore.BLUE}{float(total_loss):.3f}{Style.RESET_ALL}')

            train_loss = float(torch.tensor(train_loss).mean())
            if is_print:
                print(f'Avg train loss = {Fore.BLUE}{train_loss:.3f}{Style.RESET_ALL}')
                print(f'Start testing, version = {current_version}/{config.max_version}')
                tbar = tqdm(test_loader)
            else:
                tbar = test_loader

            model.eval()
            for batch_index, (X, Y, dataset_info) in enumerate(tbar):
                with torch.no_grad():
                    X = X.to(device, non_blocking=True)
                    Y = Y.to(device, non_blocking=True)

                    total_loss = config.profile.eval_model(X, Y, dataset_info, current_version)
                    test_loss.append(float(total_loss))

                    if is_print:
                        tbar.set_description(f'test loss = {Fore.BLUE}{float(total_loss):.3f}{Style.RESET_ALL}')

            test_loss = float(torch.tensor(test_loss).mean())
            if is_print:
                print(f'Avg test loss = {Fore.BLUE}{test_loss:.3f}{Style.RESET_ALL}')
                config.profile.save_model(model, current_version)
                epoch_end_time = datetime.datetime.now()

                print(f'[{jacky_utils.Timer.timespan_str(epoch_end_time - epoch_start_time)}] version = {current_version}')
                print()

            loss_history['train'].append(train_loss)
            loss_history['test'].append(test_loss)
            config.profile.save_history(loss_history, current_version, rank)

            if config.multi_gpu:
                dist.barrier()
                if is_print:
                    config.profile.merge_all_histories(current_version)
                dist.barrier()
            else:
                config.profile.merge_all_histories(current_version)

            current_version += 1

        except Exception as err:
            # traceback.format_exc()  # Traceback string
            traceback.print_exc()
            exception_count += 1
            current_version -= 1
            if config.is_debug or exception_count >= 50:
                if config.multi_gpu:
                    cleanup()
                exit(-1)

    if config.multi_gpu:
        cleanup()


def setup(config, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10130'

    # initialize the process group
    dist.init_process_group(
        config.ddp_backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_ddp(train_func, world_size, config):
    mp.spawn(train_func,
             args=(world_size, config),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main()
