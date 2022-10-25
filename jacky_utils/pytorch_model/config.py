import jacky_utils.pytorch_model.dataset as dataset

class Config:
    """Config settings' description.

    # extending Config class
    class Config_ToyModel(Config):
        def __init__(self):
            super().__init__()
            self.profile = profile.ToyModel(self)
            # self.find_unused_parameters = True

    profile: model's Profile object from profile.py, which control model behavior.
    computer: which computer the project run. it affect Config.get_dataset_directory() to get currect dataset directory.
    version: which model version want to load. Set None to get latest version.
        Model is saved every epoch to a version. version start from 1.
    verbose: print more information on console.
    max_version: When training "max_version" epoches, model's version matches max_version, training is stop automatically.
    batch_size: batch size when training and evaluating model.
    normalize_X: normalized method for input image (X).
    normalize_Y: normalized method for target image (Y).
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
        self.display_image = False
        self.save_image = False

    def get_multi_gpu_device_count(self):
        if self.multi_gpu:
            return len(self.ddp_cuda_devices.split(','))
        else:
            return 1

    def set_mode(self, mode: str):
        """Using it for micro-managing config.

        Args:
            mode (str): Mode name.
                'train_model_cpu_test': Used when testing model is runnable with cpu and one image.
                'train_model_single_gpu': Used when running model with single gpu.
                'train_model_multi_gpu_test': Used when testing model is runnable with multiple gpu and 2 images.
                'train_model_multi_gpu': Used when running model with multiple gpu, should be used on normally training a model.
                'train_model_4gpu': Used when running model with multiple gpu, should be used on normally training a model (4 gpu).
                'train_model_7gpu': Used when running model with multiple gpu, should be used on normally training a model with faster speed (7 gpu).
                'eval_model_gpu': Used when evaluating model.
                'eval_model_gpu_save_image': Used when evaluating model with single gpu and save inference result.
                'eval_model_gpu_plot_image': Used when evaluating model with single gpu and plot (matplotlib) inference result.
                'plot_history': Used when plotting history file (.ht) by matplotlib from "fine_tune_version".
                'plot_history_save_image': Used when plotting history file (.ht) by matplotlib then save to png image from "fine_tune_version".
        Raises:
            Exception: Unknown config mode.
        """
        if mode == 'train_model_cpu_test':
            self.verbose = True
            self.max_version = 3
            self.batch_size = 1
            self.use_data_size = 1
            self.device = 'cpu'
            self.multi_gpu = False
            self.num_workers = 0
            self.display_image = False
            self.save_image = False

        elif mode == 'train_model_single_gpu':
            self.verbose = False
            self.batch_size = 8
            self.use_data_size = None
            self.is_debug = True
            self.device = 'cuda:0'
            self.multi_gpu = False
            self.num_workers = 8
            self.ddp_cuda_devices = '0,1,2,3,4,5,6,7'
            self.display_image = False
            self.save_image = False

        elif mode == 'train_model_multi_gpu_test':
            self.verbose = True
            self.max_version = 3
            self.batch_size = 1
            self.use_data_size = 2
            self.is_debug = True
            self.device = 'cuda:0'
            self.multi_gpu = True
            self.num_workers = 0
            self.ddp_backend = 'nccl'
            self.ddp_cuda_devices = '0,1'
            self.display_image = False
            self.save_image = False

        elif mode == 'train_model_multi_gpu':
            self.verbose = False
            self.batch_size = 8
            self.use_data_size = None
            self.is_debug = True
            self.device = 'cuda:0'
            self.multi_gpu = True
            self.num_workers = 8
            self.ddp_backend = 'nccl'
            self.ddp_cuda_devices = None
            self.display_image = False
            self.save_image = False

        elif mode == 'train_model_4gpu':
            self.verbose = False
            self.batch_size = 8
            self.use_data_size = None
            self.is_debug = True
            self.device = 'cuda:0'
            self.multi_gpu = True
            self.num_workers = 8
            self.ddp_backend = 'nccl'
            self.ddp_cuda_devices = '0,1,2,3'
            self.display_image = False
            self.save_image = False

        elif mode == 'train_model_7gpu':
            self.verbose = False
            self.batch_size = 8
            self.use_data_size = None
            self.is_debug = True
            self.device = 'cuda:0'
            self.multi_gpu = True
            self.num_workers = 8
            self.ddp_backend = 'nccl'
            self.ddp_cuda_devices = '0,1,2,3,4,5,6'
            self.display_image = False
            self.save_image = False

        elif mode == 'eval_model_gpu':
            self.verbose = True
            self.batch_size = 8
            self.use_data_size = None
            self.is_debug = True
            self.device = 'cuda:7'
            self.multi_gpu = False
            self.num_workers = 8
            self.ddp_cuda_devices = '0,1,2,3,4,5,6,7'
            self.display_image = False
            self.save_image = False

        elif mode == 'eval_model_gpu_save_image':
            self.verbose = True
            self.batch_size = 1
            self.device = 'cuda:7'
            self.multi_gpu = False
            self.num_workers = 0
            self.ddp_cuda_devices = '0,1,2,3,4,5,6,7'
            self.display_image = True
            self.save_image = True

        elif mode == 'eval_model_gpu_plot_image':
            self.verbose = True
            self.batch_size = 1
            self.device = 'cuda:7'
            self.multi_gpu = False
            self.num_workers = 0
            self.ddp_cuda_devices = '0,1,2,3,4,5,6,7'
            self.display_image = True
            self.save_image = False

        elif mode == 'plot_history':
            self.verbose = True
            self.save_image = False

        elif mode == 'plot_history_save_image':
            self.verbose = True
            self.save_image = True

        else:
            raise NotImplementedError('Unknown config mode: "' + mode + '"')
