
import os
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F
import sys
from llsa.dataset import create_dataset, DataKey, create_imagefolder_dataloader
from llsa.trainers.training_cfg import BaseTrainingConfig, load_training_config


def main():
    torch.manual_seed(0)
    cfg_path = sys.argv[1]
    cfgs = load_training_config(cfg_path)
    cfg: BaseTrainingConfig = cfgs.pop('base')

    assert len(sys.argv) >= 3

    fid_state_path = sys.argv[2]

    if len(sys.argv) >= 4:
        sample_resolution = int(sys.argv[3])
    else:
        sample_resolution = cfg.resolution

    device = 'cuda'

    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid.persistent(True)

    cfg.train_batch_size = 32

    if fid_state_path is None or not os.path.exists(fid_state_path):

        if cfg.train_data_dir is not None or cfg.train_metadata_path is not None:
            valid_dataset, valid_dataloader = create_imagefolder_dataloader(
                cfg.resolution,
                cfg.train_data_dir,
                '',
                cfg.train_metadata_path,
                cfg.center_crop,
                cfg.random_flip,
                cfg.downsample_mode,
                cfg.train_batch_size,
                cfg.dataloader_num_workers
            )
        else:
            valid_dataset = create_dataset(cfg.resolution,
                                           cfg.train_dataset_name,
                                           cfg.train_dataset_config_name,
                                           cfg.train_data_dir,
                                           cfg.train_data_files,
                                           cache_dir=cfg.cache_dir,
                                           center_crop=False,
                                           random_flip=False)

            valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=cfg.train_batch_size, shuffle=False,
                num_workers=cfg.dataloader_num_workers
            )
        for i, batch in enumerate(valid_dataloader):
            batch_img = batch[DataKey.IMAGE].to(device)
            batch_img = (batch_img + 1) / 2

            if batch_img.shape[3] != sample_resolution:
                batch_img = F.interpolate(
                    batch_img, (sample_resolution, sample_resolution), mode='bilinear')

            fid.update(batch_img, real=True)

    if fid_state_path is not None:
        torch.save(fid.state_dict(), fid_state_path)


if __name__ == '__main__':
    main()
