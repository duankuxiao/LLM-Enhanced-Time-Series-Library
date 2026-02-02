import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from xgboost import XGBRegressor
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import results_evaluation
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.tools import save_config
from utils.metrics_imputation import calc_mae, calc_mse, results_evaluation_imputation, interpolate_nan_matrix

from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)
        self.loss_method = args.loss_method
        self.loss = args.loss
        if self.args.loss_method == "adaptive":
            self.lamda = nn.Parameter(torch.zeros(1))


    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.loss_method == "adaptive":
            self.lamda.data = self.lamda.data.to(self.device)

        if self.loss_method == "adaptive":
            optimizer = optim.Adam([
                {'params': self.model.parameters()},
                {'params': [self.lamda], 'lr': self.args.learning_rate * 1}
            ], lr=self.args.learning_rate)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return optimizer

    def _select_criterion(self):
        if self.loss == 'MSE':
            criterion = calc_mse
        elif self.loss == 'MAE':
            criterion = calc_mae
        return criterion

    def _select_scheduler(self, model_optim, train_loader):
        train_steps = len(train_loader)
        if self.args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        return scheduler

    def _loss_function(self, criterion, outputs, true, mask):
        mask_ = mask ^ 1
        if self.args.output_ori is False:
            missing_loss = criterion(outputs, true, mask_)
            ori_loss = criterion(outputs, true, mask)
            if self.loss_method == "fix":
                loss = ori_loss + missing_loss
            elif self.loss_method == "adaptive":
                w_ori = 1
                w_missing = 10 * torch.sigmoid(self.lamda)
                loss = ori_loss + w_missing * missing_loss

                return loss, w_ori * ori_loss, w_missing * missing_loss

            elif self.loss_method == "ori":
                loss = criterion(outputs, true)
        else:

            loss = criterion(outputs, true, mask_)
        return loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        if self.loss_method == "adaptive":
            ori_loss, missing_loss = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (inp, inp_inter, mark, mask, x_ori) in enumerate(vali_loader):
                # inp_withmask, mask, inp = mask_custom(batch_x, mask_rate=self.args.mask_rate, method=self.args.mask_method,f_dim=f_dim,seed=self.args.fix_seed,targets_only=self.args.mask_target_only)

                mark = mark.float().to(self.device)
                mask = mask.int().to(self.device)

                # encoder - decoder
                if self.args.input_inter:
                    inp_inter = inp_inter.float().to(self.device)
                    outputs = self.model(inp_inter, mark, None, None, None, mask=mask)
                else:
                    inp = inp.float().to(self.device)
                    outputs = self.model(inp, mark, None, None, None, mask=mask)

                if isinstance(outputs, tuple):
                    outputs = tuple(
                        o[:, :self.args.seq_len, -self.f_dim:].detach().cpu()
                        for o in outputs
                    )
                else:
                    outputs = outputs[:, :self.args.seq_len, -self.f_dim:]
                    outputs = outputs.detach().cpu()

                x_ori = x_ori[:, :self.args.seq_len, -self.f_dim:]
                mask = mask[:, :self.args.seq_len, -self.f_dim:]

                mask = mask.detach().cpu()
                if self.loss_method == "adaptive":
                    loss, ori, missing = self._loss_function(criterion, outputs, x_ori, mask)
                    ori_loss.append(ori.item())
                    missing_loss.append(missing.item())

                else:
                    loss = self._loss_function(criterion, outputs, x_ori, mask)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        if self.loss_method == "adaptive":
            return total_loss, np.average(ori_loss), np.average(missing_loss)
        else:
            return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if self.args.val:
            vali_data, vali_loader = self._get_data(flag='val')
        else:
            vali_data, vali_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting, 'checkpoints')
        if not os.path.exists(path):
            os.makedirs(path)
        save_config(self.args, os.path.join(path, 'configs.pkl'))

        time_now = time.time()
        time_start = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, accelerator=self.accelerator)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(model_optim, train_loader)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Initialize a dictionary to store loss values
        loss_records = {"epoch": [], "time": [], "train_loss": [], "vali_loss": []}
        if self.loss_method == "adaptive":
            loss_records = {"epoch": [], "time": [], "train_loss": [],  "train_ori_loss": [], "train_missing_loss": [], "vali_loss": [],"vali_ori_loss": [],
                            "vali_missing_loss": [], "w_ori": [], "w_missing": []}

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            if self.loss_method == "adaptive":
                ori_loss, missing_loss = [], []

            self.model.train()
            epoch_time = time.time()
            for i, (inp, inp_inter, mark, mask, x_ori) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                mark = mark.float()

                # imputation input
                x_ori = x_ori.float().to(self.device)
                mark = mark.to(self.device)
                mask = mask.int().to(self.device)

                # encoder - decoder
                if self.args.input_inter:
                    inp_inter = inp_inter.float().to(self.device)
                    outputs = self.model(inp_inter, mark, None, None, None, mask=mask)
                else:
                    inp = inp.float().to(self.device)
                    outputs = self.model(inp, mark, None, None, None, mask=mask)

                if isinstance(outputs, tuple):
                    outputs = tuple(
                        o[:, :self.args.seq_len, -self.f_dim:]
                        for o in outputs
                    )
                else:
                    outputs = outputs[:, :self.args.seq_len, -self.f_dim:]
                x_ori = x_ori[:, :self.args.seq_len, -self.f_dim:]
                mask = mask[:, :self.args.seq_len, -self.f_dim:]
                if self.loss_method == "adaptive":
                    loss, ori, missing = self._loss_function(criterion, outputs, x_ori, mask)
                    ori_loss.append(ori.item())
                    missing_loss.append(missing.item())

                else:
                    loss = self._loss_function(criterion, outputs, x_ori, mask)

                train_loss.append(loss.item())

                if self.args.accelerate:
                    self.accelerator.print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.2f}min'.format(speed, left_time / 60))
                else:
                    verbose_interval = (len(train_loader) // 5) if len(train_loader) > 5 else 1
                    if (i + 1) % verbose_interval == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        if speed > 1e-3:
                            print('\tspeed: {:.4f}s/iter; left time: {:.2f}min'.format(speed, left_time / 60))
                        else:
                            print('\tspeed: {:.4f}ms/iter; left time: {:.2f}min'.format(speed * 1000, left_time / 60))

                        iter_count = 0
                        time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if self.args.accelerate:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()
                        model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False, accelerator=self.accelerator)
                    scheduler.step()

            train_loss = np.average(train_loss)
            if self.loss_method == 'adaptive':
                ori_loss, missing_loss = np.average(ori_loss), np.average(missing_loss)
                vali_loss, vali_ori_loss, vali_missing_loss = self.vali(vali_data, vali_loader, criterion)
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion)

            # Record loss values
            loss_records["epoch"].append(epoch + 1)
            loss_records["time"].append(round((time.time() - time_start) / 60, 4))
            loss_records["train_loss"].append(train_loss)
            loss_records["vali_loss"].append(vali_loss)
            if self.loss_method == 'adaptive':
                w_ori = 1
                w_missing = 10 * torch.sigmoid(self.lamda).cpu().detach().numpy()[0]

                loss_records["train_ori_loss"].append(ori_loss)
                loss_records["train_missing_loss"].append(missing_loss)
                loss_records["vali_ori_loss"].append(vali_ori_loss)
                loss_records["vali_missing_loss"].append(vali_missing_loss)
                loss_records["w_ori"].append(w_ori)
                loss_records["w_missing"].append(w_missing)

            # test_loss = self.vali(test_data, test_loader, criterion)
            cost_time = round((time.time() - epoch_time) / 60, 2)
            print(" Epoch: {} cost time: {} min".format(epoch + 1, cost_time))
            print("☆☆☆☆☆Train Loss: {0:.7f} Vali Loss: {1:.7f}".format(train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            left_time = 1 + (self.args.patience - early_stopping.counter) * cost_time
            print("  Left time: {} min".format(round(left_time, 2)))

            if self.args.lradj != 'TST':
                if self.args.lradj == 'COS':
                    scheduler.step()
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)

            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        if self.args.accelerate:
            self.accelerator.wait_for_everyone()

        best_model_path = path + '/' + 'checkpoint'
        if self.args.accelerate:
            self.model = self.accelerator.load_state(best_model_path)
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        # Convert loss records to DataFrame and save as CSV
        folder_path = os.path.join(self.args.checkpoints, setting)
        loss_df = pd.DataFrame(loss_records)
        loss_df.to_csv(os.path.join(folder_path, "loss_records.csv"), index=False)
        print("Loss records saved to:", os.path.join(folder_path, "loss_records.csv"))
        report = torch.cuda.memory_summary(device=self.device, abbreviated=False)
        print(report)
        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        used_bytes = torch.cuda.memory_allocated(self.device)
        with open(os.path.join(folder_path, "memory_summary_{}_{}.txt".format(round(peak_alloc * 1024 / (10 ** 9), 1), round(speed * 1000, 2))), "w") as f:
            f.write(report)
        print("Saved CUDA memory summary to cuda_memory_summary.txt")
        return self.model

    def test(self, setting, test=0, path=None):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            if path is None:
                model_path = os.path.join(self.args.checkpoints, setting, 'checkpoints')
                folder_path = os.path.join(self.args.checkpoints, setting)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            else:
                model_path = os.path.join(path, 'checkpoints')
                folder_path = path
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint')))
        else:
            folder_path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        imputation_trues, imputations, masks = [], [], []

        self.model.eval()

        with torch.no_grad():
            for i, (inp, inp_inter, mark, mask, x_ori) in enumerate(test_loader):
                mark = mark.float().to(self.device)
                mask = mask.int().to(self.device)

                if self.args.input_inter:
                    inp_inter = inp_inter.float().to(self.device)
                    output = self.model(inp_inter, mark, None, None, None, mask=mask)
                else:
                    inp = inp.float().to(self.device)
                    output = self.model(inp, mark, None, None, None, mask=mask)

                if self.args.accelerate:
                    self.accelerator.wait_for_everyone()
                    output = self.accelerator.gather_for_metrics(output)

                mask = mask[:, :self.args.seq_len, -self.f_dim:]
                if isinstance(output, tuple):
                    output = output[-1][:, :self.args.seq_len, -self.f_dim:]
                else:
                    output = output[:, :self.args.seq_len, -self.f_dim:]

                true = x_ori[:, :self.args.seq_len, -self.f_dim:]

                imputation = output.detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()

                true = true.detach().cpu().numpy()
                imputation = mask * true + (1 - mask) * imputation

                if test_data.scale and self.args.inverse:
                    shape = imputation.shape
                    imputation = test_data.inverse_transform(imputation.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    true = test_data.inverse_transform(true.reshape(shape[0] * shape[1], -1)).reshape(shape)

                masks.append(mask)
                imputation_trues.append(true)
                imputations.append(imputation)
        masks = np.concatenate(masks, axis=0)
        imputation_trues = np.concatenate(imputation_trues, axis=0)
        imputations = np.concatenate(imputations, axis=0)

        print('test shape:', imputations.shape, imputation_trues.shape)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        [mse, rmse, nrmse, mae, mape, rae, r2, corr] = results_evaluation(imputation_trues.flatten(), imputations.flatten())
        print('mae:{}, r2:{}'.format(mae, r2))
        f = open(os.path.join('./results', "result_imputation.txt"), 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, r2:{}'.format(mae, r2))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(os.path.join(folder_path, 'metrics_{}_{}.npy'.format(self.args.data, self.args.data_path[:-4])), np.array([mae, mse, rmse, r2, corr]))
        np.save(os.path.join(folder_path, 'imputations_{}_{}.npy'.format(self.args.data, self.args.data_path[:-4])), imputations)
        np.save(os.path.join(folder_path, 'imputation_trues_{}_{}.npy'.format(self.args.data, self.args.data_path[:-4])), imputation_trues)

        res_df, metrics_df, imputation_metrics_df = self.res_evaluation_multi_target(imputations, imputation_trues, masks, trainable_params, folder_path)
        return res_df, metrics_df, imputation_metrics_df

    def res_evaluation_multi_target(self, imputations, trues, mask, trainable_params, path):
        stride = self.args.seq_len
        X_withnan = np.where(mask == 0, np.nan, trues)
        true = trues[::stride, :, :].reshape(-1, len(self.args.target))
        imputations = imputations[::stride, :, :].reshape(-1, len(self.args.target))
        X_withnan = X_withnan[::stride, :, :].reshape(-1, len(self.args.target))
        mask = mask[::stride, :, :].reshape(-1, len(self.args.target))
        mask_int = mask.astype(int)  # 转成整数 0/1
        mask = mask_int ^ 1  # 0↔1 取反
        pred_interpolate = interpolate_nan_matrix(X_withnan, method=self.args.interpolate_method, order=self.args.interpolate_order)

        # Create DataFrame for true and predicted values
        columns = [f"{i}_{col}" for i in self.args.target for col in ["ori", "pred", "pred_inter", "X_withnan"]]
        data_blocks = [true, imputations, pred_interpolate, X_withnan]
        data_parts = [np.hstack([block[:, i].reshape(-1, 1) for block in data_blocks]) for i in range(len(self.args.target))]
        res_df = pd.DataFrame(np.hstack(data_parts), columns=columns)

        # Initialize metrics dictionaries
        metrics = {key: [] for key in ["trainable_params", "mse", "rmse", "nrmse", "mae", "mape", "rae", "r2", "corr", "mse_inter", "rmse_inter", "nrmse_inter", "mae_inter",
                                       "mape_inter", "rae_inter", "r2_inter", "corr_inter"]}
        imputation_metrics = {key: [] for key in ["trainable_params", "mse_imputation", "rmse_imputation", "mae_imputation", "mape_imputation", "mre_imputation",
                                                  "mse_imputation_inter", "rmse_imputation_inter", "mae_imputation_inter", "mape_imputation_inter", "mre_imputation_inter"]}

        # Calculate metrics for each target
        for idx, target in enumerate(self.args.target):
            true_i, imputation_i, inter_i, withnan_i, mask_i = (
                true[:, idx], imputations[:, idx], pred_interpolate[:, idx], X_withnan[:, idx], mask[:, idx])
            # self._show_plot(idx,y_withnan=withnan_i,y_true=true_i,y_imputation=imputation_i,y_inter=inter_i,path=path)

            # Metrics for full sequence
            mse, rmse, nrmse, mae, mape, rae, r2, corr = results_evaluation(true_i, imputation_i)
            mse_inter, rmse_inter, nrmse_inter, mae_inter, mape_inter, rae_inter, r2_inter, corr_inter = results_evaluation(true_i, inter_i)

            metrics['trainable_params'] = trainable_params
            metrics["mse"].append(mse)
            metrics["rmse"].append(rmse)
            metrics["nrmse"].append(nrmse)
            metrics["mae"].append(mae)
            metrics["mape"].append(mape)
            metrics["rae"].append(rae)
            metrics["r2"].append(r2)
            metrics["corr"].append(corr)

            metrics["mse_inter"].append(mse_inter)
            metrics["rmse_inter"].append(rmse_inter)
            metrics["nrmse_inter"].append(nrmse_inter)
            metrics["mae_inter"].append(mae_inter)
            metrics["mape_inter"].append(mape_inter)
            metrics["rae_inter"].append(rae_inter)
            metrics["r2_inter"].append(r2_inter)
            metrics["corr_inter"].append(corr_inter)

            # Metrics for imputation (model output)
            mse_imp, rmse_imp, mae_imp, mape_imp, mre_imp = results_evaluation_imputation(true_i, imputation_i, mask_i)
            imputation_metrics['trainable_params'] = trainable_params
            imputation_metrics["mse_imputation"].append(mse_imp)
            imputation_metrics["rmse_imputation"].append(rmse_imp)
            imputation_metrics["mae_imputation"].append(mae_imp)
            imputation_metrics["mape_imputation"].append(mape_imp)
            imputation_metrics["mre_imputation"].append(mre_imp)

            # Metrics for imputation (interpolation)
            mse_imp_inter, rmse_imp_inter, mae_imp_inter, mape_imp_inter, mre_imp_inter = results_evaluation_imputation(true_i, inter_i, mask_i)

            imputation_metrics["mse_imputation_inter"].append(mse_imp_inter)
            imputation_metrics["rmse_imputation_inter"].append(rmse_imp_inter)
            imputation_metrics["mae_imputation_inter"].append(mae_imp_inter)
            imputation_metrics["mape_imputation_inter"].append(mape_imp_inter)
            imputation_metrics["mre_imputation_inter"].append(mre_imp_inter)

        # Create DataFrames for metrics
        metrics_df = pd.DataFrame(metrics, index=self.args.target)
        imputation_metrics_df = pd.DataFrame(imputation_metrics, index=self.args.target)

        # Add mean row
        metrics_df.loc["mean"] = metrics_df.mean()
        imputation_metrics_df.loc["mean"] = imputation_metrics_df.mean()
        print(imputation_metrics_df.loc["mean"])

        # Save DataFrames
        res_df.to_csv(os.path.join(path, f"pred_res_{self.args.data_path[:-4]}_{self.args.mask_rate}.csv"))
        metrics_df.to_csv(os.path.join(path, f"metrics_df_{self.args.data_path[:-4]}_{self.args.mask_rate}.csv"))
        imputation_metrics_df.to_csv(os.path.join(path, f"imputation_metrics_df_{self.args.data_path[:-4]}_{self.args.mask_rate}.csv"))

        return res_df, metrics_df, imputation_metrics_df

    def xgboost_imputation(self):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        true_all_list, pred_all_list, mask_all_list = [], [], []
        target_index_list = list(range(self.args.enc_in - self.args.c_out, self.args.enc_in))  # 最后4个为目标变量

        for target_idx in target_index_list:
            print(f"🔹 XGBoost Imputation for variable index {target_idx}")

            x_train, y_train = [], []

            # 构造训练样本（只使用非缺失值）
            for inp, inp_inter, mark, mask, x_ori in train_loader:
                inp = inp.numpy()
                mask = mask.numpy()
                x_ori = x_ori.numpy()

                B, T, D = inp.shape
                for b in range(B):
                    for t in range(T):
                        if mask[b, t, target_idx] == 1:
                            feature = np.delete(inp[b, t, :], target_idx)
                            if not np.any(np.isnan(feature)):
                                x_train.append(feature)
                                y_train.append(x_ori[b, t, target_idx])

            if len(x_train) == 0:
                print(f"⚠️  No valid training data for target index {target_idx}, skipping...")
                continue

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            model = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1)
            model.fit(x_train, y_train)

            # 插补
            y_true, y_pred, y_mask = [], [], []

            for inp, inp_inter, mark, mask, x_ori in tqdm(test_loader, desc=f"Imputing var {target_idx}"):
                inp = inp.numpy()
                mask = mask.numpy()
                x_ori = x_ori.numpy()

                B, T, D = inp.shape
                pred = inp[:, :, target_idx].copy()

                for b in range(B):
                    for t in range(T):
                        if mask[b, t, target_idx] == 0:
                            feature = np.delete(inp[b, t, :], target_idx)
                            if not np.any(np.isnan(feature)):
                                pred[b, t] = model.predict(feature.reshape(1, -1))[0]

                y_true.append(x_ori[:, :, target_idx])
                y_pred.append(pred)
                y_mask.append(mask[:, :, target_idx])

            # 拼接完整数组
            true_i = np.concatenate(y_true, axis=0)
            imputation_i = np.concatenate(y_pred, axis=0)
            mask_i = np.concatenate(y_mask, axis=0)
            stride = self.args.seq_len

            true_i = true_i[::stride, :].reshape(-1, 1)
            imputation_i = imputation_i[::stride, :].reshape(-1, 1)
            mask_i = mask_i[::stride, :].reshape(-1, 1)

            true_all_list.append(true_i)
            pred_all_list.append(imputation_i)
            mask_all_list.append(mask_i)
        true = np.concatenate(true_all_list, axis=-1)
        imputations = np.concatenate(pred_all_list, axis=-1)
        mask = np.concatenate(mask_all_list, axis=-1)

        if test_data.scale and self.args.inverse:
            imputations = test_data.inverse_transform(imputations)
            true = test_data.inverse_transform(true)
        X_withnan = np.where(mask == 0, np.nan, true)

        # Create DataFrame for true and predicted values
        columns = [f"{i}_{col}" for i in self.args.target for col in ["ori", "pred", "x_withnan"]]
        data_blocks = [true, imputations, X_withnan]
        data_parts = [np.hstack([block[:, i].reshape(-1, 1) for block in data_blocks]) for i in range(len(self.args.target))]
        res_df = pd.DataFrame(np.hstack(data_parts), columns=columns)

        # Initialize metrics dictionaries
        metrics = {key: [] for key in ["mse", "rmse", "nrmse", "mae", "mape", "rae", "r2", "corr"]}
        imputation_metrics = {key: [] for key in ["mse_imputation", "rmse_imputation", "mae_imputation", "mape_imputation", "mre_imputation"]}
        mask_int = mask.astype(int)  # 转成整数 0/1
        mask = mask_int ^ 1  # 0↔1 取反

        # Calculate metrics for each target
        for idx, target in enumerate(self.args.target):
            true_i, imputation_i, mask_i = (true[:, idx], imputations[:, idx], mask[:, idx])
            imputation_i = np.where(true_i <= 0.0001, 0, imputation_i)
            # true_i = np.where(true_i <= 0.0001, 0, true_i)

            # Metrics for full sequence
            mse, rmse, nrmse, mae, mape, rae, r2, corr = results_evaluation(true_i, imputation_i)

            metrics["mse"].append(mse)
            metrics["rmse"].append(rmse)
            metrics["nrmse"].append(nrmse)
            metrics["mae"].append(mae)
            metrics["mape"].append(mape)
            metrics["rae"].append(rae)
            metrics["r2"].append(r2)
            metrics["corr"].append(corr)

            # Metrics for imputation (model output)
            mse_imp, rmse_imp, mae_imp, mape_imp, mre_imp = results_evaluation_imputation(true_i, imputation_i, mask_i)

            imputation_metrics["mse_imputation"].append(mse_imp)
            imputation_metrics["rmse_imputation"].append(rmse_imp)
            imputation_metrics["mae_imputation"].append(mae_imp)
            imputation_metrics["mape_imputation"].append(mape_imp)
            imputation_metrics["mre_imputation"].append(mre_imp)

        # Create DataFrames for metrics
        metrics_df = pd.DataFrame(metrics, index=self.args.target)
        imputation_metrics_df = pd.DataFrame(imputation_metrics, index=self.args.target)

        # Add mean row
        metrics_df.loc["mean"] = metrics_df.mean()
        imputation_metrics_df.loc["mean"] = imputation_metrics_df.mean()
        print(imputation_metrics_df.loc["mean"])
        path = r'D:\Time-LLM-main\results'

        # Save DataFrames
        res_df.to_csv(os.path.join(path, f"pred_res_{self.args.data_path[:-4]}.csv"))
        metrics_df.to_csv(os.path.join(path, f"metrics_df_{self.args.data_path[:-4]}.csv"))
        imputation_metrics_df.to_csv(os.path.join(path, f"imputation_metrics_df_{self.args.data_path[:-4]}.csv"))

        return res_df, metrics_df, imputation_metrics_df

    def _show_plot(self, i, y_withnan, y_true, y_imputation, y_inter, path=None):
        x_range = np.arange(self.args.num_train - self.args.seq_len * 7, self.args.num_train)
        y_true_plot = y_true[-self.args.seq_len * 7:]
        plt.figure(i + 1, figsize=(20, 5))
        plt.plot(x_range, y_true_plot, "r-", label="True values")

        y_withnan_plot = y_withnan[-self.args.seq_len * 7:]
        plt.plot(x_range, y_withnan_plot, "k-", label="With nan values")

        y_imputation_plot = y_imputation[-self.args.seq_len * 7:]
        plt.plot(x_range, y_imputation_plot, "g--", label="Imputation values")

        y_inter_plot = y_inter[-self.args.seq_len * 7:]
        plt.plot(x_range, y_inter_plot, "b--", label="Interpolate values")
        ymin, ymax = plt.ylim()
        plt.vlines(self.args.num_train - self.args.seq_len * 7, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        plt.ylim(ymin, ymax)
        plt.legend(loc="upper left")
        plt.title('Prediction')
        plt.xlabel("Periods")
        plt.ylabel("Y")
        plt.savefig(os.path.join(path, '{}.png'.format(i)))
        plt.close()
