import jacky_utils.dataset as dataset

class Config:
    """Config settings' description.

    profile: model's Profile object from profile.py, which control model behavior.
    computer: which computer the project run. it affect Config.get_dataset_directory() to get currect dataset directory.
    version: which model version want to load. Set None to get latest version.
        Model is saved every epoch to a version. version start from 1.
    verbose: print more information on console.
    max_version: When training "max_version" epoches, model's version matches max_version, training is stop automatically.
    batch_size: batch size when training and evaluating model.
    normalize_X: normalized method for input image (X). Input image should be 4 channels bayer RGGB image.
    normalize_Y: normalized method for target image (Y). Target image should be 3 channels RGB image.
    use_data_size: Number of data wanted to used. Set "None" when using full dataset.
    learning_rate: Adam optimizer's learning rate.
    is_debug: Set true to let program stop when any exception raised.
    device: Set to 'cpu' to use cpu. Set to 'cuda:0' to use GPU No.0 when multi_gpu is False.
        Set to 'cuda:0' to use GPU No.0 for ddp server when multi_gpu is True. (ddp: Distributed Data Parallel)
    multi_gpu: Using multiple GPUs or not.
    num_workers: num_workers setting for torch.utils.data.DataLoader.
    ddp_backend: ddp server backend. All options: gloo, mpi, nccl (ddp: Distributed Data Parallel)
    ddp_cuda_devices: parameter to set os.environ['CUDA_VISIBLE_DEVICES']. It's the visible GPUs device can be used.
        Set it to "0,1,2,3" mean GPUs No.0, No.1, No.2, No.3 can be used.
        Set it to "0,2,6,7" mean GPUs No.0, No.2, No.6, No.7 can be used.
        See config.set_mode('train_model_multi_gpu') in get_config docstrings to get more information.
    find_unused_parameters: find_unused_parameters is set to True if there are some output nodes are not used in the model.
        torch.nn.parallel.DistributedDataParallel module will hint to use find_unused_parameters or not.
        First, it can be set to False, then set it to True if receiving error.

    """

    def __init__(self):
        self.profile = None
        self.computer = None
        self.version = None
        self.verbose = False
        self.max_version = 500
        self.batch_size = 8
        self.normalize_X = dataset.ZeroToOneNormalization()
        self.normalize_Y = dataset.ZeroToOneNormalization()
        self.use_data_size = None
        self.learning_rate = 0.001
        self.is_debug = True
        self.device = 'cpu'
        self.multi_gpu = False
        self.num_workers = 8
        self.ddp_backend = 'nccl'
        self.ddp_cuda_devices = '0,1,2,3,4,5,6,7'
        self.find_unused_parameters = False

    def get_multi_gpu_device_count(self):
        if self.multi_gpu:
            return len(self.ddp_cuda_devices.split(','))
        else:
            return 1