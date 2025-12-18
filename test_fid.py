import os
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

from llsa.dataset import create_dataset, DataKey, create_imagefolder_dataloader
from llsa.trainers import BaseTrainingConfig, load_training_config
from llsa.pipelines.flow_matching_pipeline import FlowMatchingPipeline
from tqdm import tqdm

import argparse


def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(
        description="Compute FID for a given model.")
    parser.add_argument("cfg_path", type=str,
                        help="Path to the training configuration file.")
    parser.add_argument("fid_state_path", type=str, nargs="?",
                        default=None, help="Path to save/load FID state.")
    parser.add_argument("input_resolution", type=int, nargs="?",
                        default=None, help="Input resolution for sampling.")
    parser.add_argument("--n_sample_data_batch", type=int,
                        default=20, help="Number of sample data batches.")
    parser.add_argument("--valid_batch_size", type=int,
                        default=50, help="Validation batch size.")
    parser.add_argument("--num_inference_steps", type=int,
                        default=20, help="Number of inference steps.")
    parser.add_argument("--cfg_scale", type=float,
                        default=None, help="CFG scale value.")

    args = parser.parse_args()

    cfgs = load_training_config(args.cfg_path)
    cfg: BaseTrainingConfig = cfgs.pop('base')

    fid_state_path = args.fid_state_path
    input_resolution = args.input_resolution
    n_sample_data_batch = args.n_sample_data_batch
    valid_batch_size = args.valid_batch_size
    num_inference_steps = args.num_inference_steps
    cfg_scale = args.cfg_scale

    device = 'cuda'

    model_dir = cfg.output_dir

    pipeline = FlowMatchingPipeline.from_pretrained(model_dir).to(device)
    pipeline.set_progress_bar_config(disable=True)

    default_resoltuion = pipeline.denoiser.config.sample_size

    if input_resolution is not None:
        sample_resolution = input_resolution
    else:
        sample_resolution = default_resoltuion

    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid.persistent(True)

    pipeline.to(torch.bfloat16)

    if fid_state_path is None or not os.path.exists(fid_state_path):

        if cfg.train_data_dir is not None:
            valid_dataset, valid_dataloader = create_imagefolder_dataloader(
                cfg.resolution,
                cfg.train_data_dir,
                '',
                cfg.center_crop,
                cfg.random_flip,
                cfg.downsample_mode,
                cfg.train_batch_size,
                cfg.dataloader_num_workers,
                shuffle=False
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
    else:
        fid.load_state_dict(torch.load(fid_state_path))

    # run pipeline in inference (sample random noise and denoise)
    for _ in tqdm(range(n_sample_data_batch)):

        images = pipeline(
            batch_size=valid_batch_size,
            sample_shape=sample_resolution,
            num_inference_steps=num_inference_steps,
            output_type="np",
            eta=0,
            cfg=cfg_scale
        ).images
        image_tensor = torch.from_numpy(
            images).permute(0, 3, 1, 2).to(device)

        fid.update(image_tensor, False)

    print('FID:', fid.compute().item())


if __name__ == '__main__':
    main()
