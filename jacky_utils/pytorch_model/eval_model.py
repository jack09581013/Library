import os
import matplotlib.pyplot as plt
import torch.nn
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm

from config import Config


def get_config():
    return Config().set_mode('eval_model_gpu')


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


def eval_model():
    torch.backends.cudnn.benchmark = True
    config = get_config()
    version, model = config.profile.load_model()
    config.profile.set_used_model(model)
    device = config.device

    test_dataset = ExampleDataset()
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=True,
                             num_workers=config.num_workers, pin_memory=True, drop_last=True)

    if config.save_image:
        root = f'result/{config.profile}'
        os.makedirs(root, exist_ok=True)

    MSE_loss = torch.nn.MSELoss()
    loss_mse_eval = 0
    loss_psnr_eval = 0

    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {number_of_parameters:,}')  # 47,554,738
    print(f'Model size: {number_of_parameters / 1024 / 1024 * 4:.2f} MB')
    print(f'Model name: {config.profile}')
    print('Number of evaluation data:', len(test_dataset))

    print(f'Start evaluating, version = {version}')
    tbar = tqdm(test_loader)    
    model.eval()
    for batch_index, (X, Y, dataset_info) in enumerate(tbar):
        with torch.no_grad():
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)

            enhanced = config.profile.produce_result(X, dataset_info)
            loss_mse_temp = MSE_loss(enhanced, Y).item()
            loss_mse_eval += loss_mse_temp
            loss_psnr_temp = 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))
            loss_psnr_eval += loss_psnr_temp

            if config.display_image:
                if config.save_image:
                    # save image
                    pass
                else:
                    # display image
                    pass

            tbar.set_description(f'mse: {loss_mse_temp:.4f}')

    NUMBER_OF_BATCH = len(test_dataset) / config.batch_size
    loss_mse_eval = loss_mse_eval / NUMBER_OF_BATCH
    loss_psnr_eval = loss_psnr_eval / NUMBER_OF_BATCH
    print(f'mse: {loss_mse_eval:.4f}, psnr: {loss_psnr_eval:.4f}')


if __name__ == '__main__':
    eval_model()
