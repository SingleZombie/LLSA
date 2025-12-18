# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class FlowMatchingPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        denoiser ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "denoiser"

    def __init__(self, denoiser, scheduler):
        super().__init__()

        self.register_modules(denoiser=denoiser, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        sample_shape: int = None,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        cfg: float = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
                DDIM and `1` corresponds to DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
                downstream to the scheduler (use `None` for schedulers which don't support this argument).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDIMPipeline
        >>> import PIL.Image
        >>> import numpy as np

        >>> # load model and scheduler
        >>> pipe = DDIMPipeline.from_pretrained("fusing/ddim-lsun-bedroom")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe(eta=0.0, num_inference_steps=50)

        >>> # process image to PIL
        >>> image_processed = image.cpu().permute(0, 2, 3, 1)
        >>> image_processed = (image_processed + 1.0) * 127.5
        >>> image_processed = image_processed.numpy().astype(np.uint8)
        >>> image_pil = PIL.Image.fromarray(image_processed[0])

        >>> # save image
        >>> image_pil.save("test.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        # Sample gaussian noise to begin loop
        if isinstance(self.denoiser.config.sample_size, int):
            image_shape = (
                batch_size,
                self.denoiser.config.in_channels,
                self.denoiser.config.sample_size,
                self.denoiser.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.denoiser.config.in_channels,
                           *self.denoiser.config.sample_size)

        self.denoiser.eval()

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if hasattr(self.scheduler, 'noise_size') and self.scheduler.noise_size is not None:
            noise_shape = (image_shape[0], image_shape[1],
                           self.scheduler.noise_size, self.scheduler.noise_size)
            image = torch.randn(noise_shape, generator=generator,
                                dtype=self.denoiser.dtype, device=self._execution_device)

            if self.scheduler.noise_size > image_shape[2]:
                resize_mode = self.scheduler.config.noise_resize_mode
                image = F.interpolate(
                    image, (image_shape[2], image_shape[3]), mode=resize_mode)
            else:
                image = F.interpolate(
                    image, (image_shape[2], image_shape[3]), mode='bilinear')
        else:
            image = torch.randn(image_shape, generator=generator,
                                dtype=self.denoiser.dtype, device=self._execution_device)

        if hasattr(self.scheduler, 'noise_rescale'):
            image *= self.scheduler.noise_rescale

        if sample_shape is not None:
            image = F.interpolate(image, sample_shape, mode='bilinear')

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        accepts_eta = "eta" in set(inspect.signature(
            self.scheduler.step).parameters.keys())

        extra_kwargs = {"generator": generator}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        use_cfg = False
        if cfg is None:
            emtpy_cls_label = None
        else:
            use_cfg = cfg != 1
            emtpy_cls_label = torch.full(
                (image.shape[0],),  self.denoiser.config.num_embeds_ada_norm, dtype=torch.long, device=image.device)
            guidance_cls_label = torch.randint(0, self.denoiser.config.num_embeds_ada_norm,
                                               (image.shape[0],), generator=generator, dtype=torch.long, device=image.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output

            if use_cfg:
                model_output = self.denoiser(
                    image, t, class_labels=emtpy_cls_label).sample
                model_output_cfg = self.denoiser(
                    image, t, class_labels=guidance_cls_label).sample
                model_output = model_output + cfg * \
                    (model_output_cfg - model_output)
            else:
                model_output = self.denoiser(
                    image, t, class_labels=emtpy_cls_label).sample

            if self.scheduler.config.pred_x0:
                model_output = (image - model_output) / \
                    (t/1000).clamp_min(5e-2)

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, **extra_kwargs
            ).prev_sample

        if hasattr(self.scheduler.config, 'image_rescale'):
            image *= self.scheduler.config.image_rescale

        image = image.to(torch.float32)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
