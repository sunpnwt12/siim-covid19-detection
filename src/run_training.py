from timm.models.efficientnet import tf_efficientnet_b7_ns
from lib.include import *
from dataset import get_train_dataset, get_valid_dataset, get_train_loader, get_valid_loader
from model import Effnet, convert_silu_to_mish
from loss import DiceBCELoss, LovaszHinge
from metrics import Metrics, AverageMeter
from CONFIG import GlobalConfig
class Trainer:
    def __init__(self,
        model,
        device,
        config,
        fold,
        dry_run=False,
        wandblogger=False,
    ):
        self.dry_run = dry_run
        self.config = config
        self.model = model
        self.device = device
        self.fold = fold
        self.wandblogger = wandblogger
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_aux1 = nn.BCEWithLogitsLoss()
        self.criterion_aux2 = LovaszHinge()
        self.criterion_aux3 = DiceBCELoss()
        self.optimizer = MADGRAD(self.model.parameters(), lr=self.config.lr)
        self.scheduler_params = dict(
        T_0=5,
        T_mult=2,
        eta_min=0.0001,
        last_epoch=-1,
        verbose=False    
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,  **self.scheduler_params)
        self.scaler = GradScaler(enabled=self.config.use_apex)
        
        self.epoch = 0
        self.best_score = 0
        self.best_score_epoch = 0
        self.best_loss = 10**5
        self.best_loss_epoch = 0

        if self.dry_run:
            print('[DRY RUN]')

        if self.wandblogger and not self.dry_run:
            wandb.login()
            wandb.init(project='siim-covid19-detection',
                        name=f'{self.fold}_{self.run_name}',
                        group=f'{self.config.wandb_groupname}',
                        tags=['study'],
                        )

        self.base_dir = str(Path.cwd() / 'outputs' / f'{self.fold}_{self.run_name}')
        Path(self.base_dir).mkdir(exist_ok=True)

        self.log_path = str(Path(f'self.base_dir') / 'log.txt')

        self._log(f'Train on device: {self.device} ({torch.__version__})')
        self._log('')
        self._log(f'Fold: {self.fold}/{self.config.fold - 1}')
        self._log('')
        self._log(f'Augmentation: {self.config.aug}')
        self._log('')
        self._log(f'Run name: {self.config.run_name}')
        self._log(f'Mixed Precision: {self.config.use_apex}')
        self._log('')
        self._log(f'Gradient Accumulation Step: {self.config.grad_accum_step}')
        self._log(f'Batch_size: {self.config.batch_size} ({self.config.grad_accum_step * self.config.batch_size}')
        self._log('')
        self._log(f'Optimizer: {self.optimizer}')
        self._log(f'LR Scheduler on (Valid: {self.config.valid_scheduler}, Train: {self.config.train_scheduler})')
        self._log(f'')
        self._log(f'Loss: {self.criterion} + ({self.criterion_aux1} + {self.criterion_aux2} + {self.criterion_aux3})')
        
    def fit(self, train_loader, validation_loader):
        for epoch in range(self.config.num_epochs):

            t = time.time()
            train_loss, train_aux_loss, train_metrics = self._train_one_epoch(train_loader)
            train_time = time.time() - t

            self._log(f'[RESULT]: Train. Epoch: {self.epoch + 1}/{self.config.n_epochs}, '
                        f'loss: {train_loss.avg:.5f}, '
                        f'aux_loss: {train_aux_loss.avg:.5f}, '
                        f'acc: {train_metrics.accuracy:.5f}, '
                        f'mAP: {train_metrics.mAP:.5f}, '
                        f'AUROC: {train_metrics.AUROC:.5f}, '
                        f'LR: {self.optimizer.param_groups[0].lr}, '
                        f'time: {train_time:.5f}')
            self._save(f'{self.base_dir}/last-checkpoint.pth', last=True)

            t = time.time()
            valid_loss, valid_metrics = self._validation(validation_loader)
            valid_time = time.time() - t 
            
            self._log(f'[RESULT]: Valid. Epoch: {self.epoch + 1}/{self.config.n_epochs}, '
                        f'summary_loss: {valid_loss.avg:.5f}, '
                        f'acc: {valid_metrics.accuracy:.5f}, '
                        f'mAP: {valid_metrics.mAP:.5f}, '
                        f'AUROC: {valid_metrics.AUROC:.5f}, '
                        f'LR: {self.optimizer.param_groups[0].lr}, '
                        f'time: {valid_time:.5f}')


            if self.wandblogger and not self.dry_run:
                wandb.log({"train/loss": train_loss.avg,
                            "train/aux_loss": train_aux_loss.avg,
                            "train/lr": self.optimizer.param_groups[0].lr,
                            "val/loss": valid_loss.avg,
                            "metrics/mAP": valid_metrics.mAP,
                            "metrics/acc": valid_metrics.accuracy,
                            "metrics/AUROC": valid_metrics.AUROC,
                            })

            if valid_loss.avg < self.best_loss:
                self.best_loss = valid_loss.avg
                self.best_loss_epoch = self.epoch
                self.model.eval()
                self._save(f'{self.base_dir}/best-loss.pth')
                self._log(f'[SAVED] Epoch: {self.epoch + 1} best-loss: {self.best_summary_loss:.5f}')

            if valid_metrics.mAP > self.self.best_score:
                self.best_score = valid_metrics.mAP
                self.best_score.epoch = self.epoch
                self.model.eval()
                self._save(f'{self.base_dir}/best-score.pth')
                self._log(f'[SAVED] Epoch: {self.epoch + 1} best-score: {self.best_score:.5f}')
            
            if self.config.valid_scheduler:
                self.scheduler.step()

            self.epoch += 1

            time_per_epoch = train_time + valid_time
            total_time = (time_per_epoch * self.config.num_epochs)
            
            self.trained_time += time_per_epoch
            self.eta = total_time - self.trained_time

            self._log(f'[TIME]: Trained: {timedelta(seconds=self.trained_time)}, ETA: {timedelta(seconds=self.eta)}')

            if self.dry_run:
                break

            gc.collect()

        self._log('')
        self._log(f'[BEST LOSS] {self.best_summary_loss:.5f} at epoch {self.best_summary_loss_epoch + 1}')
        self._log(f'[BEST SCORE] {self.best_score:.5f} at epoch {self.best_score_epoch + 1} ')
        if self.wandblogger:
            wandb.finish()
        

    def _train_one_epoch(self, train_loader):
        self.model.train()
        train_average_loss = AverageMeter()
        train_average_aux_loss = AverageMeter()
        metrics = Metrics(device=self.device)

        t = time.time()

        for step, (images, masks, targets) in enumerate(train_loader):
            images = torch.stack(images).to(self.device)
            batch_size = images.shape[0]
            masks = torch.unsqueeze(torch.stack(masks)).to(self.device)
            masks = F.interpolate(masks, size=(32, 32), mode='bilinear', align_corners=False)
            labels = torch.stack([target['labels'] for target in targets]).to(self.device).float()
            
            
            with autocast(enabled=self.config.use_apex):
                outputs, aux_outputs = self.model(images)
                loss = self.criterion(outputs, torch.max(labels, 1)[1])
                aux_loss1 = self.criterion_aux1(aux_outputs, masks) # bce
                aux_loss2 = self.criterion_aux2(aux_outputs, masks) # lovasz
                aux_loss3 = self.criterion_aux3(aux_outputs, masks) # dice_bce
                aux_loss = (0.2 * aux_loss1) + (0.7 * aux_loss2) + (0.1 * aux_loss3)

            loss /= self.config.grad_accum_step
            aux_loss /= self.config.grad_accum_step
            self.scaler.scaler(loss + aux_loss).backward()
            
            if ((step + 1) % self.config.grad_accum_step == 0) or ((step + 1) == len(train_loader)):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.config.train_scheduler:
                    self.scheduler.step()
                self.optimier.zero_grad(set_to_none=True)

                train_average_loss.update(loss.detach().item() * self.config.grad_accum_step,
                                        batch_size * self.config.grad_accum_step)
                train_average_aux_loss.update(aux_loss.detach().item() * self.config.grad_accum_step,
                                        batch_size * self.config.grad_accum_step)

                prob = torch.softmax(outputs)

                metrics.update(y_onehot=torch.max(labels, 1)[1],
                                proba=prob,
                                pred_onehot=torch.max(prob, 1)[1])

            if step % self.config.verbose_step == 0:
                print(
                    f'Train Step: {str(step).zfill(len(str(len(train_loader))))}/{len(train_loader)}, '
                    f'loss: {train_average_loss.avg:.5f}, '
                    f'aux_loss: {train_average_aux_loss:.5f}, '
                    f'acc: {metrics.accuracy:.5f}, '
                    f'LR: {self.optimizer.param_groups[0].lr}, '
                    f'time: {(time.time() - t):.2f}'
                    f'[{(step + 1)/(time.time() - t):.2f}it/s]',
                )

            if self.dry_run:
                break
        if not self.dry_run:
            metrics.calculate()

        return train_average_loss, train_average_aux_loss, metrics

    def _validation(self, validation_loader):
        self.model.eval()
        valid_average_loss = AverageMeter()
        metrics = Metrics(device=self.device)
        
        t = time.time()

        for step, (images, _, targets) in enumerate(validation_loader):
            images = torch.stack(images).to(self.device)
            batch_size = images.shape[0]
            labels = torch.stack([target['labels'] for target in targets]).to(self.device).float()
            
            with torch.no_grad():
                outputs, _ = torch.stack(images).to(self.device)
                loss = self.criterion(outputs, torch.max(labels, 1)[1])
            
            valid_average_loss.update(loss, batch_size)

            prob = torch.softmax(outputs)

            metrics.update(y_onehot=torch.max(labels, 1)[1],
                            proba=prob,
                            pred_one_hot=torch.max(prob, 1)[1])

            if step % self.config.verbose_step == 0:
                print(
                    f'Valid Step: {str(step).zfill(len(str(len(validation_loader))))}/{len(validation_loader)}, '
                    f'loss: {valid_average_loss.avg:.5f}, '
                    f'acc: {metrics.accuracy:.5f}, '
                    f'time: {(time.time() - t):.2f}'
                    f'[{(step + 1)/(time.time() - t):.2f}it/s]',
                )

            if self.dry_run:
                break
        if not self.dryp_run:
            metrics.calculate()
        return valid_average_loss, metrics

    def _save(self, path, last=False):
        self.model.eval()
        if last:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'best_score': self.best_score,
                'epoch': self.epoch,
            }, path)
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'best_loss': self.best_loss,
                'best_score': self.best_score,
                'epoch': self.epoch,
            }, path)
        

            
    def load(self, path, last=False):
        checkpoint = torch.load(path, map_location=self.device)
        if last:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_loss = checkpoint['best_loss']
            self.best_score = checkpoint['best_score']
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_loss = checkpoint['best_loss']
            self.best_score = checkpoint['best_score']
        self._log('[LOADED STATE DICT]')
    
    def _log(self, message):
        print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n') 

def run_training(cfg):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    for fold in range(cfg.fold):
        
        model = Effnet(model_name=tf_efficientnet_b7_ns, pretrained=True)

        train_dataset = get_train_dataset(fold)
        valid_dataset = get_valid_dataset(fold)

        train_loader = get_train_loader(train_dataset, cfg.batch_size)
        valid_loader = get_valid_loader(valid_dataset, cfg.batch_size)

        trainer = Trainer(
            model,
            device,
            cfg,
            fold,
            dry_run=cfg.dry_run,
            wandblogger=cfg.use_wandblogger,
        )
        trainer.fit(train_loader, valid_loader)


if __name__ == '__main__':
    cfg = GlobalConfig()
    run_training(cfg)