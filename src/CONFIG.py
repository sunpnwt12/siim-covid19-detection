from lib.include import *
from augmentations import get_train_transforms

class GlobalConfig:
    image_size  = 512
    seed = 42
    batch_size = 4
    fold = 5
    num_classes = 4
    num_workers = 4

    # training conf
    num_epochs = 15
    lr = 0.001
    aug = str(get_train_transforms())
    
    dry_run = True
    run_name = 'effnetb7'
    use_wandblogger = False
    wandb_groupname = 'exp1'
    
    valid_scheduler = False
    train_schedulr = True
    use_apex = False

    grad_accum_step = 8

    folds_df_path = str(Path('../dataset/folds_df.csv'))