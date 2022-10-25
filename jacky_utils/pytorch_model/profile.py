import os
import torch
import numpy as np
import jacky_utils


class Profile:
    """Profile object manages models.

    import jacky_utils.pytorch_model.loss as loss
    class Losses:
        def __init__(self, device):
            self.VGG_19 = loss.vgg_19(device)
            self.MSE_loss = torch.nn.MSELoss()
            self.SmoothL1Loss = torch.nn.SmoothL1Loss()
            self.L1_Loss = torch.nn.L1Loss()
            self.MS_SSIM = loss.MS_SSIM()
            self.SSIM = loss.SSIM()


    class UNet_v1(Profile):
        def get_model(self):
            return unet_v1.model.UNet_v1()

        def train_model(self, X, Y, dataset_info, current_version):
            enhanced = self.used_model(X)
            total_loss = self.losses.SmoothL1Loss(enhanced, Y)
            return total_loss

        def produce_result(self, X, dataset_info):
            return self.used_model(X)
    """
    def __init__(self, config):
        os.makedirs(self.version_file_path(), exist_ok=True)
        self.config = config
        self.used_model = None  # It could be nn.Module or ddp module (parallel computing)
        self.losses = None
        assert '-' not in str(self)

    def load_model(self, rank=None):
        version = self.config.version
        verbose = self.config.verbose

        if rank is None:
            # model = torch.nn.DataParallel(self.get_model()).to(self.config.device)
            model = self.get_model().to(self.config.device)
        else:
            model = self.get_model().to(rank)

        if version is None:
            if verbose:
                print('Find latest version')
            version = Profile.get_latest_version(self.version_file_path(), '.pt')

        if version is None:
            if verbose:
                print('Can not find any version')
            version = 0
        else:
            if verbose:
                print('Using version:', version)
            pt_file = self.model_file_name(version)

            if os.path.exists(pt_file):
                if verbose:
                    print('Load version model:', pt_file)
                model.load_state_dict(torch.load(
                    pt_file, map_location=self.config.device))
            else:
                raise FileNotFoundError(f'Cannot find neural network file: {pt_file}')

        return version, model

    def load_history(self, version, rank=None, load_merge=True):
        verbose = self.config.verbose
        loss_history = {
            'train': [],
            'test': []
        }

        if version is None:
            if verbose:
                print('Find latest version')
            version = Profile.get_latest_version(self.version_file_path(), '.ht')

        if version is None:
            if verbose:
                print('Can not find any version')
            version = 0
        else:
            if verbose:
                print('Using version:', version)
            ht_file = self.history_file_name(version, rank, load_merge)

            if os.path.exists(ht_file):
                if verbose:
                    print('Load version history:', ht_file)
                loss_history = jacky_utils.load(ht_file)
            else:
                raise FileNotFoundError(f'Cannot find history file: {ht_file}')

        return version, loss_history

    def save_model(self, model, version):
        torch.save(model.state_dict(), self.model_file_name(version))

    def save_history(self, history, version, rank):
        jacky_utils.save(history, self.history_file_name(version, rank, load_merge=False))

    def merge_all_histories(self, version):
        total_loss_history = {
            'train': [],
            'test': []
        }

        train_loss = []
        test_loss = []
        remove_file_list = []

        for rank in range(self.config.get_multi_gpu_device_count()):
            version, loss_history = self.load_history(version, rank, load_merge=False)
            train_loss.append(loss_history['train'])
            test_loss.append(loss_history['test'])
            remove_file_list.append(self.history_file_name(version, rank, load_merge=False))

        train_loss = np.array(train_loss).mean(axis=0)
        test_loss = np.array(test_loss).mean(axis=0)

        total_loss_history['train'] = list(train_loss)
        total_loss_history['test'] = list(test_loss)

        save_path = self.history_file_name(version)
        jacky_utils.save(total_loss_history, save_path)

        for file in remove_file_list:
            os.remove(file)

    def model_file_name(self, version):
        return os.path.join(self.version_file_path(), f'{self}-{version}.pt')

    def history_file_name(self, version, rank=None, load_merge=True):
        if load_merge:
            return os.path.join(self.version_file_path(), f'{self}-{version}.ht')
        else:
            if rank is None:
                return os.path.join(self.version_file_path(), f'{self}-{version}.0.ht')
            else:
                return os.path.join(self.version_file_path(), f'{self}-{version}.{rank}.ht')

    def version_file_path(self):
        return f'./models/{self}'

    def set_used_model(self, model):
        self.used_model = model

    def init_losses(self, losses):
        self.losses = losses

    def __str__(self):
        return type(self).__name__

    def get_model(self):
        raise NotImplementedError()

    def train_model(self, X, Y, dataset_info, current_version):
        raise NotImplementedError()

    def eval_model(self, X, Y, dataset_info, current_version):
        return self.train_model(X, Y, dataset_info, current_version)

    def train_color_chart(self, X, Y, dataset_info, current_version):
        raise NotImplementedError()

    def produce_result(self, X, dataset_info):
        raise NotImplementedError()

    @staticmethod
    def get_latest_version(file_path, file_extension):
        version_codes = []
        for file in os.listdir(file_path):
            if file_extension in file:
                version_codes.append(Profile.version_code(file))
        if len(version_codes) > 0:
            version_codes.sort()
            return version_codes[-1]
        else:
            return None

    @staticmethod
    def version_code(file):
        a = file.index('-')
        b = file.index('.')
        return int(file[a + 1:b])
