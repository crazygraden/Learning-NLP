class Config(object):
    def __init__(self, args):
        print('aldata5')
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        # self.final_out_channels = 128   # (3conv)
        self.final_out_channels = 256   # (4conv)
        # self.final_out_channels = 512
        self.num_classes = 5
        self.dropout = 0.2
        self.features_len = 131
        # self.features_len = 268  # original
        # training configs
        self.num_epoch = 60

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 2e-3

        # data parameters
        self.drop_last = True
        self.batch_size = 16
        self.dataname = args.selected_dataset
        self.fuzzy_flag = args.fuzzy
        self.model_select = args.model_select
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC(self.model_select)
        self.augmentation = augmentations(self.fuzzy_flag)
        self.num_heads = 8


class augmentations(object):
    def __init__(self, fuzzy_flag):
        # self.jitter_scale_ratio = 0.0009
        # self.jitter_ratio = 0.005
        # self.max_seg = 11
        self.jitter_scale_ratio = 0.05
        self.jitter_ratio = 0.008
        self.max_seg = 10
        self.fuzzy = fuzzy_flag  # true 使用datatransfor_fuzzy;否则使用datatransfor


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self, model_select):
        self.hidden_dim = 100
        self.timesteps = 15
        self.windows = 9
        if model_select == 'mlp':
            self.flag = 0   # 1:dilatConv；0:mlp
        else:
            self.flag = 1
