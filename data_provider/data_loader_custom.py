import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings


class Dataset_Custom(Dataset):
    def __init__(self,configs, root_path, flag='train', size=None,
                 features='S', data_path='PV_power.csv',
                 target=['PV'], scale=True, timeenc=0, freq='h', percent=100,seasonal_patterns=None):
        self.configs = configs
        self.num_train = configs.num_train
        self.num_test = configs.num_test
        self.task_name = configs.task_name

        self.source_data_path = configs.source_data_path
        self.forecast_dim = configs.forecast_dim
        self.feature_cols = configs.feature_cols
        self.c_out = configs.c_out
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        if 'forecast' in self.task_name:
            self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        else:
            self.tot_len = len(self.data_x) - self.seq_len + 1
        if scale:
            self.std_ = self.target_scaler.scale_

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        # s_begin = (index // 24) * 24
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        x_forecast = self.data_forecast[r_begin:r_end, :self.forecast_dim]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, x_forecast

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        try:
            df_source_domain = pd.read_csv(os.path.join(self.root_path, self.source_data_path))
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        except:
            df_source_domain = pd.read_csv(os.path.join(self.root_path, self.source_data_path),encoding='SHIFT-JIS')
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path),encoding='SHIFT-JIS')

        if self.feature_cols is None:
            self.feature_cols = df_raw.columns[1:]
        cols = list(self.feature_cols.copy())

        if 'date' in cols:
            cols.remove('date')

        if self.features == 'M' and self.target is None:
            self.target = self.feature_cols
        if self.target is not None:
            for s in self.target:
                if s in cols:
                    cols.remove(s)
        df_raw = df_raw[['date'] + cols + self.target]
        df_source_domain = df_source_domain[['date'] + cols + self.target]

        num_vali = len(df_raw) - self.num_train - self.num_test
        if 'forecast' in self.task_name:
            border1s = [0, self.num_train - self.seq_len, len(df_raw) - self.num_test - self.seq_len]
            border2s = [self.num_train, self.num_train + num_vali, len(df_raw)]
        else:
            border1s = [0, self.num_train, len(df_raw) - self.num_test]
            border2s = [self.num_train, self.num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_target = df_raw[self.target]

        cols_data_source_domain = df_source_domain.columns[1:]
        df_data_source_domain = df_source_domain[cols_data_source_domain]
        df_target_source_domain = df_source_domain[self.target]

        if self.scale:
            train_data = df_data_source_domain[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            df_target_data = df_target_source_domain[border1s[0]:border2s[0]]
            self.target_scaler.fit(df_target_data.values)

        else:
            data = df_data.values

        self._get_dataset(df_raw, data, border1, border2)

    def _get_dataset(self,df_raw,data,border1,border2):
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.features == 'S':
            self.data_x = data[border1:border2, -1:]
            self.data_y = data[border1:border2, -1:]
        else:
            self.data_x = data[border1:border2, :len(self.feature_cols)]
            self.data_y = data[border1:border2, -len(self.target):]

        self.data_forecast = data[border1:border2, :self.forecast_dim]
        self.data_stamp = data_stamp
