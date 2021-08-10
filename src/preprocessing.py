from lib.include import *
from CONFIG import GlobalConfig

def prepare_local(cfg):
    folds_df = pd.read_csv(cfg.folds_df_path)
    local_train = Path(cfg.repo_dir) / 'dataset' / 'siim-covid19-train' / 'train'

    local_train_list = glob(f'{str(local_train)}/*.jpg')
    local_train_df = pd.Series(local_train_list).to_frame('local_train_jpg')
    local_train_df['file_name'] = local_train_df['local_train_jpg'].apply(lambda row : row[-16:-4])

    local_mask = Path(cfg.repo_dir) / 'dataset' / 'train_masked' / 'train_masked'
    local_mask_list = glob(f'{str(local_mask)}/*.jpg')
    local_mask_df = pd.Series(local_mask_list).to_frame('local_mask_jpg')
    local_mask_df['file_name'] = local_mask_df['local_mask_jpg'].apply(lambda row : row[-16:-4])

    folds_df = folds_df.merge(local_train_df, on='file_name', how='inner')
    folds_df = folds_df.merge(local_mask_df, on='file_name', how='inner')

    folds_df_local = Path(cfg.repo_dir) / 'dataset' / 'folds_df_local.csv'
    folds_df.to_csv(folds_df_local, index=False)

if __name__ == '__main__':
    cfg = GlobalConfig()
    prepare_local(cfg)
