import pandas as pd
import torch
import os
import random
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

fix_seed = 1234567  # 4213
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def get_setting(args,ii):
    setting = 'if_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_sd{}_td{}_dm{}_df{}_el{}_dl{}_nh{}_ma{}_factor{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.enc_in,
        args.c_out,
        args.d_model,
        args.d_ff,
        args.e_layers,
        args.d_layers,
        args.n_heads,
        args.moving_avg,
        args.factor, args.loss_method)

    if 'LLM' in args.model:
        setting += '_{}_llmd{}_llmf{}_tk{}'.format(args.llm_model, args.llm_dim, args.llm_layers, args.top_k)
        if args.use_prompt:
            setting += '_prompt'
    if 'RNN' in args.model:
        setting += '_{}'.format(args.rnn_model)
    if args.use_norm:
        setting += '_norm'

    if args.use_forecast:
        setting += '_forecast'
    if args.percent != 100:
        setting = 'few-shot{}_'.format(args.percent) + setting
    if args.scale:
        setting += '_scale'
    return setting


def main(args):
    from exp.exp_probabilistic_forecasting import Exp_Forecast
    Exp = Exp_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            # setting record of experiments
            setting = get_setting(args,ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            _, loss_df = exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            res_df, res_metrics_df = exp.test(setting)
            torch.cuda.empty_cache()
        return res_df, res_metrics_df, loss_df

    else:
        ii = 0
        setting = get_setting(args,ii)

        exp = Exp(args)  # set experiments
        print(' >>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        res_df, res_metrics_df = exp.test(setting, test=1)
        torch.cuda.empty_cache()
        return res_df, res_metrics_df


if __name__ == '__main__':
    from setup_LLMformer import model_hyperparameter_setup
    from configs.electricity_configs import args as default_args
    from copy import deepcopy

    all_results = []

    model_list = ['LSTM', 'Transformer', 'Informer', 'iTransformer', 'TimesNet', 'PatchTST', 'LLMformer']

    for data in ['tokyo']:  # 'Sapporo','Sendai','Fukuoka','Tokyo'   'hokkaido', 'kyushu', 'tohoku'
        for loss_method in ['hybridmu']:  # 'g' 'mse' 'msemu' 'hybridmu'   'adaptive'
            for model in model_list:
                args = deepcopy(default_args)
                args.data_path = '{}.csv'.format(data)
                args.source_data_path = '{}.csv'.format(data)
                args.task_name = 'interval_forecast'  # 'interval_forecast'  'long_term_forecast'
                args.loss_method = loss_method

                args.seq_len = 72
                args.pred_len = 24
                args.label_len = args.seq_len
                args.is_training = 1
                args.use_prompt = False
                args.model_id = 'test'

                args.model = model
                args = model_hyperparameter_setup(args)
                _, res_metrics_df, loss_df = main(args)
                res_metrics_df.insert(0, 'model', model)

                all_results.append(res_metrics_df.iloc[-1:])
                final_metrics_df = pd.concat(all_results, axis=0, ignore_index=False)
                final_metrics_df.to_csv('./results/pf_{}_all_models_comparison.csv'.format(args.model_id))
