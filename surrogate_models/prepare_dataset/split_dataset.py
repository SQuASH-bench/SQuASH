import os

from config import PathConfig
from util import split
from util.data_loader import load_data, save_data

if __name__ == '__main__':
    path_config = PathConfig()
    data_set_name = "train_ghz_b_aug_proxy_squash"
    d_suffix = "ghz_b_aug_proxy_squash"

    dataset_path = os.path.join(path_config.paths['gcn_data'], f'{data_set_name}.pt')
    data = load_data(dataset_path)

    train_data, val_data, test_data = split.train_val_test_split(data, random_seed=42)
    print(f"[INFO] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    save_data(os.path.join(
        f'{dataset_path}/train_{d_suffix}.pt'),
        train_data)
    save_data(os.path.join(
        f'{dataset_path}/val_{d_suffix}.pt'),
        val_data)
    save_data(os.path.join(
        f'{dataset_path}/test_{d_suffix}.pt'),
        test_data)
