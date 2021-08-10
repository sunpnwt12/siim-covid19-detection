from lib.include import *
from augmentations import get_train_transforms, get_valid_transforms
from CONFIG import GlobalConfig



class SiimDataset(Dataset):
    def __init__(self, df, transforms=None):
        super().__init__()
        self.df = df
        self.image_path = self.df['train_jpg']
        self.mask_path = self.df['train_mask']
        self.transforms = transforms

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = self._get_image(idx)
        mask = self._get_mask(idx)
        target = {}
        target['labels'] = self.df[[
            ['Negative for Pneumonia', 
            'Typical Appearance',
            'Indeterminate Appearance',
            'Atypical Appearance']
        ]].iloc[idx].values


        if self.transforms:
            transforms_dict = {
                'image': image,
                'mask': mask,
                'class_labels' : target
            }
            transformed = self.transforms(**transforms_dict)
            image = torch.as_tensor(transformed['image'], dtype=torch.float32)
            mask = torch.as_tensor(transformed['mask'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['class_labels'], dtype=torch.int64)

        return image / 255.0, mask / 255.0, target

    def _get_image(self, idx):
        image = cv2.imread(self.image_path.iloc[idx])
        return image

    def _get_mask(self, idx):
        mask = cv2.imread(self.mask_path.iloc[idx])
        return mask

cfg = GlobalConfig()

folds_df = pd.read_csv(cfg.folds_df_path)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_dataset(fold):
    return SiimDataset(df=folds_df[folds_df['fold'] != fold],
                        transforms=get_train_transforms())

def get_valid_dataset(fold):
    return SiimDataset(df=folds_df[folds_df['fold'] == fold],
                        transforms=get_valid_transforms())


def get_train_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers,
                    pin_memory=True, sampler=RandomSampler(dataset), collate_fn=collate_fn)

def get_valid_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers,
                    pin_memory=True, sampler=SequentialSampler(dataset), collate_fn=collate_fn)