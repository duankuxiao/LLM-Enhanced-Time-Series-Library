import os
import torch
from models import TimeLLM, TimesNet, DLinear, Informer, Transformer, iTransformer, LSTM, PatchTST, LLMformer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'DLinear': DLinear,
            'Informer': Informer,
            'Transformer': Transformer,
            'TimeLLM': TimeLLM,
            'iTransformer': iTransformer,
            'LSTM': LSTM,
            'PatchTST':PatchTST,
            'LLMformer':LLMformer,
        }

        self.f_dim = self.args.c_out
        if args.accelerate:
            from accelerate import Accelerator, DeepSpeedPlugin
            from accelerate import DistributedDataParallelKwargs

            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
            self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin, device_placement=True)
            self.device = self.accelerator.device
        else:
            self.device = args.device if args.device is not None else self._acquire_device()
            self.accelerator = None

        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def train(self, setting):
        pass

    def test(self, setting, test, path):
        pass