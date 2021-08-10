from lib.include import *

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
    
    dry_run = True
    use_wandblogger = False
    run_name = 'effnetb7'
    wandb_groupname = 'exp1'
    
    valid_scheduler = False
    train_scheduler = True
    use_apex = False

    grad_accum_step = 8

    # manage path
    cwd = Path.cwd()
    repo_dir = cwd.parent
    folds_df_path = repo_dir / 'dataset' / 'folds_df.csv'
    outputs_path = repo_dir / 'outputs' / f'{run_name}'
    seed_everything(seed)