import inspect
import os

from dataclasses import dataclass
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version
import torch
import torch.nn.functional as F

from .trainer import Trainer
from ..dataset import DataKey


@dataclass
class DDPMTrainingConfig:
    # Diffuion Models
    model_config: str
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = 'linear'
    prediction_type: str = 'epsilon'
    ddpm_num_inference_steps: int = 100

    # Validation
    valid_batch_size: int = 1
    valid_loops: int = 100
    valid_fid: bool = False

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


class DDPMTrainer(Trainer):
    def __init__(self, weight_dtype, accelerator, logger, cfg: DDPMTrainingConfig):
        super().__init__(weight_dtype, accelerator, logger, cfg)

    def init_modules(self,
                     enable_xformer=False,
                     gradient_checkpointing=False):
        config = UNet2DModel.load_config(self.cfg.model_config)
        self.model = UNet2DModel.from_config(config)

        # Create EMA for the model.
        if self.cfg.use_ema:
            self.ema_model = EMAModel(
                self.model.parameters(),
                decay=self.cfg.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=self.cfg.ema_inv_gamma,
                power=self.cfg.ema_power,
                model_cls=UNet2DModel,
                model_config=self.model.config,
            )

        if enable_xformer:
            self.model.enable_xformers_memory_efficient_attention()

        accepts_prediction_type = "prediction_type" in set(
            inspect.signature(DDPMScheduler.__init__).parameters.keys())
        if accepts_prediction_type:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.cfg.ddpm_num_steps,
                beta_schedule=self.cfg.ddpm_beta_schedule,
                prediction_type=self.cfg.prediction_type,
            )
        else:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.cfg.ddpm_num_steps,
                beta_schedule=self.cfg.ddpm_beta_schedule)

        if gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

    def init_optimizers(self, train_batch_size, gradient_accumulation_steps=1):
        if self.cfg.scale_lr:
            self.cfg.learning_rate = (
                self.cfg.learning_rate * gradient_accumulation_steps *
                train_batch_size * self.accelerator.num_processes
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.adam_weight_decay,
            eps=self.cfg.adam_epsilon,
        )

    def init_lr_schedulers(self, gradient_accumulation_steps, num_epochs):
        self.lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps *
            gradient_accumulation_steps,
            num_training_steps=(len(self.train_dataloader)
                                * num_epochs)
        )

    def prepare_modules(self):
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        if self.cfg.use_ema:
            self.ema_model.to(self.accelerator.device)

    def set_dataset(self, dataset, train_dataloader, valid_dataloader):
        super().set_dataset(dataset, train_dataloader, valid_dataloader)

    def models_to_train(self):
        self.model.train()

    def training_step(self, global_step, batch) -> dict:
        weight_dtype = self.weight_dtype
        clean_images = batch[DataKey.IMAGE].to(weight_dtype)
        # Sample noise that we'll add to the images
        noise = torch.randn(clean_images.shape,
                            dtype=weight_dtype, device=clean_images.device)
        bsz = clean_images.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(
            clean_images, noise, timesteps)

        with self.accelerator.accumulate(self.model):
            # Predict the noise residual
            model_output = self.model(noisy_images, timesteps).sample

            if self.cfg.prediction_type == "epsilon":
                # this could have different weights!
                loss = F.mse_loss(model_output.float(), noise.float())
            else:
                raise ValueError(
                    f"Unsupported prediction type: {self.cfg.prediction_type}")

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
        unet = self.accelerator.unwrap_model(self.model)

        if self.cfg.use_ema:
            self.ema_model.store(unet.parameters())
            self.ema_model.copy_to(unet.parameters())

        pipeline = DDPMPipeline(
            unet=unet,
            scheduler=self.noise_scheduler,
        ).to(unet.device)
        pipeline.set_progress_bar_config(disable=True)

        # run pipeline in inference (sample random noise and denoise)
        if self.cfg.valid_fid:
            for _ in range(self.cfg.valid_loops):
                images = pipeline(
                    batch_size=self.cfg.valid_batch_size,
                    num_inference_steps=self.cfg.ddpm_num_inference_steps,
                    output_type="np",
                ).images
                image_tensor = torch.from_numpy(
                    images).permute(0, 3, 1, 2).to(self.accelerator.device)
                self.fid.update(image_tensor, False)

        generator = torch.Generator(
            device=pipeline.device).manual_seed(0)
        images = pipeline(
            generator=generator,
            batch_size=self.cfg.valid_batch_size,
            num_inference_steps=self.cfg.ddpm_num_inference_steps,
            output_type="np",
        ).images

        if self.cfg.use_ema:
            self.ema_model.restore(unet.parameters())

        # denormalize the images and save to tensorboard
        images_processed = (images * 255).round().astype("uint8")

        if self.cfg.valid_fid:
            msg_dict = {'fid': self.fid.compute()}
        else:
            msg_dict = {}
        self.accelerator.log(msg_dict, step=global_step)

        if self.logger == "tensorboard":
            if is_accelerate_version(">=", "0.17.0.dev0"):
                tracker = self.accelerator.get_tracker(
                    "tensorboard", unwrap=True)
            else:
                tracker = self.accelerator.get_tracker("tensorboard")
            tracker.add_images(
                "test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
        elif self.logger == "wandb":
            # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
            import wandb
            self.accelerator.get_tracker("wandb").log(
                {"test_samples": [wandb.Image(
                    img) for img in images_processed], "epoch": epoch},
                step=global_step,
            )

    def save_pipeline(self, output_dir):
        unet = self.accelerator.unwrap_model(self.model)

        if self.cfg.use_ema:
            self.ema_model.store(unet.parameters())
            self.ema_model.copy_to(unet.parameters())

        pipeline = DDPMPipeline(
            unet=unet,
            scheduler=self.noise_scheduler,
        )

        pipeline.save_pretrained(output_dir)

        if self.cfg.use_ema:
            self.ema_model.restore(unet.parameters())

    def save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            if self.cfg.use_ema:
                self.ema_model.save_pretrained(
                    os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(self, models, input_dir):
        if self.cfg.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "unet_ema"), UNet2DModel)
            self.ema_model.load_state_dict(load_model.state_dict())
            self.ema_model.to(self.accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DModel.from_pretrained(
                input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model
