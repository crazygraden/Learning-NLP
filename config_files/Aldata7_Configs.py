class Config(object):
    def __init__(self, args):
        # model configs
        self.input_channels = 1
        self.kernel_size = 16
        self.stride = 1
        # self.final_out_channels = 128   # (3conv)
        self.final_out_channels = 256   # (4conv)

        self.num_classes = 7
        # self.dropout = 0.4
        self.dropout = 0.3
        self.features_len = args.features_len   # (4conv)
        # self.features_len = 313  # (3conv)
        # training configs
        self.num_epoch = 150

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 5e-5

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
        self.hidden_dim = 256
        self.timesteps = 15
        self.windows = 9
        if model_select == 'mlp':
            self.flag = 0   # 1:dilatConv；0:mlp
        else:
            self.flag = 1
