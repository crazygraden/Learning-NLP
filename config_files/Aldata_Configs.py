class Config(object):
    def __init__(self, args):
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        # self.final_out_channels = 128   # (3conv)
        self.final_out_channels = 512   # (4conv)

        self.num_classes = 7
        self.dropout = 0.1
        self.features_len = 200   # (4conv)
        # self.features_len = 403  # (3conv)
        # training configs
        self.num_epoch = 60

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 2e-5

        # data parameters
        self.drop_last = True
        self.batch_size = 32
        self.fuzzy_flag = args.fuzzy
        self.model_select = args.model_select
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC(self.model_select)
        self.augmentation = augmentations(self.fuzzy_flag)


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
        self.hidden_dim = 100
        self.timesteps = 10
        if model_select == 'mlp':
            self.flag = 0   # 1:dilatConv；0:mlp
        else:
            self.flag = 1
