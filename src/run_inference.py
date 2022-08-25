from lib.include import *
from model import Effnet
from dataset import get_test_dataset, get_test_loader
from CONFIG import GlobalConfig
from preprocessing import prepare_local


def load_model_state(net, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_stat_dict(checkpoint["model_state_dict"])


def get_test_df():
    path = Path(cfg.repo_dir) / "dataset" / "test_df_local.csv"
    df = pd.read_csv(path)
    return df


def make_prediction_study_df(net, loader):
    # load_model_state(net, checkpoint_path)
    pass


def make_prediction_image_df():
    pass


def run_inference(cfg):

    test_df = get_test_df()
    test_dataset = get_test_dataset(test_df)
    test_loader = get_test_loader(test_dataset)

    net = Effnet()

    pred_study_df = make_prediction_study_df(net, test_loader)
    pred_image_df = make_prediction_image_df()


if __name__ == "__main__":
    cfg = GlobalConfig()
    prepare_local(cfg, test=True)
    run_inference(cfg)
