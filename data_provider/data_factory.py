from torch.utils.data import DataLoader
from data_provider.data_loader_custom import Dataset_Custom

def data_provider(args, flag):
    Data = Dataset_Custom
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    drop_last = True
    freq = args.freq
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    batch_size = 1 if (flag == 'test' or flag == 'TEST') else args.batch_size

    data_set = Data(
        configs=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        scale=args.scale,
        seasonal_patterns=args.seasonal_patterns,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
