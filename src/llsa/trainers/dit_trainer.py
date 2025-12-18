import os
import timm
import random

from dataclasses import dataclass
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel, compute_density_for_timestep_sampling
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from omegaconf import OmegaConf

from .trainer import register_trainer
from .training_cfg import register_training_config
from .ddpm_trainer import DDPMTrainer
from ..dataset import DataKey

from ..models.rope_dit_transformer_2d import RopeDiTTransformer2DModel
from ..schedulers.flow_training_scheduler import FlowMatchEulerDiscreteTrainingScheduler
from ..pipelines.flow_matching_pipeline import FlowMatchingPipeline

__CLASS_NAME = 'dit'

DIT_CLASS_DICT = {
    'RopeDiTTransformer2DModel': RopeDiTTransformer2DModel
}


@register_training_config(__CLASS_NAME)
@dataclass
class DiTTrainingConfig:
    # Diffuion Models
    model_config: str
    scheduler_config: str
    model_pretrained_path: str = None
    ddpm_beta_schedule: str = 'linear'
    num_inference_steps: int = 100
    dit_cls_name: str = 'base'
    random_resolution: bool = False
    is_flow_matching: bool = False
    reset_parameters: bool = False
    enable_snr: bool = False
    enable_repa: bool = False
    repa_weight: float = 0.5
    multi_res_loss: bool = False
    resolution_weight: bool = False
    enable_cls_label: bool = False
    use_logit_norm: bool = False

    # Validation
    valid_batch_size: int = 1
    valid_loops: int = 100
    valid_resolution: int = None
    valid_fid: bool = False
    valid_cfg_scale: float = None
    fid_path: str = None

    # EMA
    use_ema: bool = False
    ema_max_decay: float = 0.9999
    ema_inv_gamma: float = 1.0
    ema_power: float = 3 / 4

    # AdamW
    scale_lr = False
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08

    # LR Scheduler
    lr_scheduler: str = 'constant'
    lr_warmup_steps: int = 500


@register_trainer(__CLASS_NAME)
class DiTTrainer(DDPMTrainer):
    def __init__(self, weight_dtype, accelerator, logger, cfg: DiTTrainingConfig):
        super().__init__(weight_dtype, accelerator, logger, cfg)

    def init_modules(self,
                     enable_xformer=False,
                     gradient_checkpointing=False):
        class_name = OmegaConf.load(self.cfg.model_config)['_class_name']
        self.model_cls = DIT_CLASS_DICT[class_name]
        config = self.model_cls.load_config(self.cfg.model_config)

        self.model = self.model_cls.from_config(config)
        if self.cfg.model_pretrained_path is not None:
            tmp_model = self.model_cls.from_pretrained(
                self.cfg.model_pretrained_path)
            # self.model.load_state_dict(tmp_model.state_dict(), strict=False)
            self.load_similar_weights(tmp_model.state_dict())
            # self.model.reset_repa_layers()

            if self.cfg.reset_parameters:
                self.model.set_attn_out_to_zero()

        self.sample_size = self.model.sample_size

        # Create EMA for the model.
        if self.cfg.use_ema:
            self.ema_model = EMAModel(
                self.model.parameters(),
                decay=self.cfg.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=self.cfg.ema_inv_gamma,
                power=self.cfg.ema_power,
                model_cls=self.model_cls,
                model_config=self.model.config,
            )

        # if enable_xformer:
        #     self.model.enable_xformers_memory_efficient_attention()

        if self.cfg.is_flow_matching:
            self.noise_scheduler = FlowMatchEulerDiscreteTrainingScheduler.from_config(
                FlowMatchEulerDiscreteTrainingScheduler.load_config(self.cfg.scheduler_config))
        else:
            self.noise_scheduler = DDPMScheduler.from_config(
                DDPMScheduler.load_config(self.cfg.scheduler_config))

        if gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        if self.cfg.enable_repa:
            self.repa_encoder = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitb14').to(self.weight_dtype)
            patch_resolution = 16
            del self.repa_encoder.head
            self.repa_encoder.head = torch.nn.Identity()
            self.repa_encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                self.repa_encoder.pos_embed.data, [
                    patch_resolution, patch_resolution],
            )

    def load_similar_weights(self, ckpt):
        new_state_dict = self.model.state_dict()

        for k, v in ckpt.items():
            if k not in new_state_dict:
                print(f"Skip {k} (not in new model)")
                continue
            if new_state_dict[k].shape != v.shape:
                print(
                    f"Shape mismatch for {k}: {v.shape} -> {new_state_dict[k].shape}")
                if v.dim() == 2:
                    out_dim = min(v.shape[0], new_state_dict[k].shape[0])
                    in_dim = min(v.shape[1], new_state_dict[k].shape[1])
                    new_state_dict[k][:out_dim, :in_dim] = v[:out_dim, :in_dim]
                elif v.dim() == 1:
                    dim = min(v.shape[0], new_state_dict[k].shape[0])
                    new_state_dict[k][:dim] = v[:dim]
                else:
                    pass
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)

    def prepare_modules(self):
        super().prepare_modules()

        if self.cfg.enable_repa:
            self.repa_encoder.to(self.accelerator.device)

    # @line_profiler.profile
    def training_step(self, global_step, batch) -> dict:
        self.model.train()

        weight_dtype = self.weight_dtype
        clean_images = batch[DataKey.IMAGE].to(weight_dtype)
        if self.cfg.enable_cls_label:
            cls_label = batch[DataKey.INT_LABLE]
        else:
            cls_label = None

        if self.cfg.enable_repa:
            repa_input = F.interpolate(clean_images, 224, mode='bicubic')
            gt_repa_feature = self.repa_encoder.forward_features(
                repa_input)['x_norm_patchtokens'].to(weight_dtype)
            gt_repa_feature = torch.nn.functional.normalize(
                gt_repa_feature, dim=-1)

        sample_size = self.sample_size

        cur_weight = 1
        if self.cfg.random_resolution:
            scales_for_sample = [1, 0.75, 0.5, 0.25]
            cur_scale = random.choice(scales_for_sample)
            sample_size = int(sample_size * cur_scale)
            cur_weight = 1 / cur_scale

        if clean_images.shape[3] != sample_size:
            clean_images = F.interpolate(
                clean_images, (sample_size, sample_size), mode='area')

        noise_size = None
        if hasattr(self.noise_scheduler, 'noise_size') and self.noise_scheduler.noise_size is not None:
            noise_size = self.noise_scheduler.noise_size
        elif self.cfg.random_resolution:
            noise_size = self.sample_size

        if noise_size is not None:
            noise_shape = (clean_images.shape[0], clean_images.shape[1],
                           noise_size, noise_size)
            noise = torch.randn(noise_shape,
                                dtype=weight_dtype, device=clean_images.device)

            if noise_size > clean_images.shape[2]:
                resize_mode = self.noise_scheduler.config.noise_resize_mode
                noise = F.interpolate(
                    noise, (clean_images.shape[2], clean_images.shape[3]), mode=resize_mode)
            else:
                noise = F.interpolate(
                    noise, (clean_images.shape[2], clean_images.shape[3]), mode='bilinear')
        else:
            noise = torch.randn(clean_images.shape,
                                dtype=weight_dtype, device=clean_images.device)

        if hasattr(self.noise_scheduler, 'noise_rescale'):
            noise *= self.noise_scheduler.noise_rescale
        if hasattr(self.noise_scheduler.config, 'image_rescale'):
            clean_images *= self.noise_scheduler.config.image_rescale

        bsz = clean_images.shape[0]
        # Sample a random timestep for each image

        if self.cfg.use_logit_norm:
            u = compute_density_for_timestep_sampling(
                "logit_normal",
                bsz,
                0.0,
                1.0,
                device=clean_images.device
            )
            timesteps = (
                u * self.noise_scheduler.config.num_train_timesteps).long()

        else:

            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

        if self.cfg.is_flow_matching:
            noisy_images = self.noise_scheduler.add_noise_indices(
                clean_images, noise, timesteps)
            init_timesteps = timesteps
            timesteps = self.noise_scheduler.init_timesteps[timesteps.to('cpu')].to(
                clean_images.device)

        else:
            noisy_images = self.noise_scheduler.add_noise(
                clean_images, noise, timesteps)

        with self.accelerator.accumulate(self.model):
            # Predict the noise residual
            if self.cfg.enable_repa:
                model_output, repa_feature = self.model(
                    noisy_images, timesteps, class_labels=cls_label, return_repa_feature=True)
            else:
                model_output = self.model(
                    noisy_images, timesteps, class_labels=cls_label).sample
            if self.noise_scheduler.config.pred_x0:
                if self.cfg.no_x0_weight:
                    model_output = (noisy_images - model_output) / \
                        (timesteps[:, None, None, None]/1000).clamp_min(5e-2)
                else:
                    model_output = (noisy_images - model_output) / \
                        (timesteps[:, None, None, None]/1000).clamp_min(5e-2)

            output_list = []
            target_list = []
            weight_list = []

            if self.noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.prediction_type == "v_prediction":
                if self.noise_scheduler.config.pred_x0:
                    target = (noisy_images - clean_images) / \
                        (timesteps[:, None, None, None]/1000).clamp_min(5e-2)
                else:
                    target = noise - clean_images

            if self.cfg.multi_res_loss:
                c_height = clean_images.shape[2]
                while c_height >= 32:
                    c_output = F.interpolate(
                        model_output.float(), c_height, mode='area')
                    c_target = F.interpolate(
                        target.float(), c_height, mode='area')
                    c_weight = 32 / c_height
                    output_list.append(c_output)
                    target_list.append(c_target)
                    weight_list.append(c_weight)
                    c_height //= 2

            else:
                output_list.append(model_output.float())
                target_list.append(target.float())
                weight_list.append(1.0)

            loss = 0
            for output, target, loss_weight in zip(output_list, target_list, weight_list):
                cur_loss = F.mse_loss(output, target, reduction="none")
                if self.cfg.enable_snr:
                    weight = self.noise_scheduler.get_snr_weight(
                        clean_images, init_timesteps)
                    cur_loss *= weight
                cur_loss = cur_loss.mean()
                cur_loss *= loss_weight
                loss += cur_loss

            # if self.noise_scheduler.prediction_type == "epsilon":
            #     # this could have different weights!
            #     loss = F.mse_loss(model_output.float(), noise.float())
            # elif self.noise_scheduler.prediction_type == "v_prediction":
            #     target = noise - clean_images
            #     loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
            #     # rectfied flow
            #     if self.cfg.enable_snr:
            #         weight = self.noise_scheduler.get_snr_weight(clean_images, init_timesteps)
            #         loss *= weight

            #     loss = loss.mean()
            # else:
            #     raise ValueError(
            #         f"Unsupported prediction type: {self.cfg.prediction_type}")

            if self.cfg.enable_repa:
                repa_feature = torch.nn.functional.normalize(
                    repa_feature, dim=-1)
                repa_loss = -(gt_repa_feature *
                              repa_feature).sum(dim=-1).mean()
                loss += repa_loss * self.cfg.repa_weight

            if self.cfg.resolution_weight:
                loss *= cur_weight

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        if self.accelerator.sync_gradients:
            if self.cfg.use_ema:
                self.ema_model.step(self.model.parameters())

        logs = {"loss": loss.detach().item()}
        if self.cfg.use_ema:
            logs["ema_decay"] = self.ema_model.cur_decay_value

        return logs

    def validate(self, epoch, global_step):
        denoiser = self.accelerator.unwrap_model(self.model)
        denoiser.eval()

        if self.cfg.valid_resolution is not None:
            tmp = denoiser.config.sample_size
            denoiser.config.sample_size = self.cfg.valid_resolution

        if self.cfg.use_ema:
            self.ema_model.store(denoiser.parameters())
            self.ema_model.copy_to(denoiser.parameters())

        pipeline = FlowMatchingPipeline(
            denoiser=denoiser,
            scheduler=self.noise_scheduler,
        ).to(denoiser.device)
        pipeline.set_progress_bar_config(disable=True)

        # run pipeline in inference (sample random noise and denoise)
        if self.cfg.valid_fid and not (epoch == 0 and global_step == 0):
            assert self.cfg.fid_path is not None
            fid = FrechetInceptionDistance(
                normalize=True, sync_on_compute=False).to(self.accelerator.device)
            fid.persistent(True)
            fid.load_state_dict(torch.load(self.cfg.fid_path))

            generator = torch.Generator(
                device=pipeline.device).manual_seed(0)

            for _ in range(self.cfg.valid_loops):
                images = pipeline(
                    batch_size=self.cfg.valid_batch_size,
                    num_inference_steps=self.cfg.num_inference_steps,
                    output_type="np",
                    generator=generator,
                    cfg=self.cfg.valid_cfg_scale
                ).images
                image_tensor = torch.from_numpy(
                    images).permute(0, 3, 1, 2).to(self.accelerator.device)
                fid.update(image_tensor, False)

        generator = torch.Generator(
            device=pipeline.device).manual_seed(0)
        images = pipeline(
            generator=generator,
            batch_size=4,
            num_inference_steps=self.cfg.num_inference_steps,
            output_type="np",
            cfg=self.cfg.valid_cfg_scale
        ).images

        denoiser.train()

        if self.cfg.valid_resolution is not None:
            denoiser.config.sample_size = tmp

        if self.cfg.use_ema:
            self.ema_model.restore(denoiser.parameters())

        # denormalize the images and save to tensorboard
        images_processed = (images * 255).round().astype("uint8")

        if self.cfg.valid_fid and epoch > 0:
            msg_dict = {'fid': fid.compute()}
        else:
            msg_dict = {}
        self.accelerator.log(msg_dict, step=global_step)

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img)
                                     for img in images_processed])
                tracker.writer.add_images(
                    "test_samples", np_images, epoch, dataformats="NHWC")
            # if tracker.name == "wandb":
            #     tracker.log(
            #         {
            #             "test_samples": [
            #                 wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
            #             ]
            #         }
            #     )

    def save_pipeline(self, output_dir):
        denoiser = self.accelerator.unwrap_model(self.model)

        if self.cfg.use_ema:
            self.ema_model.store(denoiser.parameters())
            self.ema_model.copy_to(denoiser.parameters())

        pipeline = FlowMatchingPipeline(
            denoiser=denoiser,
            scheduler=self.noise_scheduler,
        )

        pipeline.save_pretrained(output_dir)

        if self.cfg.use_ema:
            self.ema_model.restore(denoiser.parameters())

    def save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            if self.cfg.use_ema:
                self.ema_model.save_pretrained(
                    os.path.join(output_dir, "denoiser_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "denoiser"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(self, models, input_dir):
        if self.cfg.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "denoiser_ema"), self.model_cls)
            self.ema_model.load_state_dict(load_model.state_dict())
            self.ema_model.to(self.accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = self.model_cls.from_pretrained(
                input_dir, subfolder="denoiser")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model
