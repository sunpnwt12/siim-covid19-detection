from threading import local
from lib.include import *
from CONFIG import GlobalConfig


def create_df(type):
    if type == "mask":
        folder = "train_masked"
    else:
        folder = "siim-covid19-train"
    local_path = Path(cfg.repo_dir) / "dataset" / f"{folder}" / f"{type}"
    local_list = glob(f"{str(local_path)}/*.jpg")
    local_df = pd.Series(local_list).to_frame(f"local_{type}_jpg")
    local_df["file_name"] = local_df[f"local_{type}_jpg"].apply(lambda row: row[-16:-4])
    return local_df


def prepare_local(cfg, test=False):
    if not test:

        local_train_df = create_df("train")
        local_mask_df = create_df("mask")

        folds_df = pd.read_csv(cfg.folds_df_path)
        folds_df = folds_df.merge(local_train_df, on="file_name", how="inner")
        folds_df = folds_df.merge(local_mask_df, on="file_name", how="inner")

        folds_df_local = Path(cfg.repo_dir) / "dataset" / "folds_df_local.csv"
        folds_df.to_csv(folds_df_local, index=False)

    else:
        local_test_files_path = Path(cfg.repo_dir) / "siim-covid19-train" / "test"
        local_test_series = pd.Series(
            glob(rf"{local_test_files_path}/*/*/*.dcm"), name="test_path"
        )

        # extract image no.
        local_test_image = local_test_series.str.extract(
            rf"{local_test_files_path}/.*/.*/(.*).dcm"
        )
        local_test_image.columns = ["ImageInstatnceUID"]

        # extract study no.
        local_test_study = local_test_series.str.extract(
            rf"{local_test_files_path}/(.*)/.*/.*.dcm"
        )
        local_test_study.columns = ["StudyInstanceUID"]

        # make test_df
        local_test_df = pd.concat(
            [local_test_study, local_test_image, local_test_series], axis=1
        )

        local_test_df_path = Path(cfg.repo_dir) / "dataset" / "test_df_local.csv"
        local_test_df.to_csv(local_test_df_path, index=False)


if __name__ == "__main__":
    cfg = GlobalConfig()
    prepare_local(cfg)
