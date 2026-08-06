"""Native Modly adapter for the full Hunyuan Shape 2.1 model.

This is not the Mini, Turbo, or Turbo Mini model. FP16 weights are loaded from
a lossless safetensors repack and quantized locally with MMGP/Quanto when INT8
or FP8 is selected. The adapter generates geometry only and exports GLB files.
"""

from __future__ import annotations

import io
import gc
import os
import random
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator


MODEL_FILE = "hunyuan_3d_v2.1.safetensors"
MODEL_BYTES = 7_365_943_290
MODEL_SHA256 = "5f21e98a6cb99b13b5e224abaee33929570fff7af2b6a0060001559a04ba9d72"


def release_offloader(offloader) -> None:
    """Release MMGP without letting cleanup hide the original error."""

    if offloader is not None:
        try:
            offloader.release()
        except Exception as exc:
            print(f"[Hunyuan3D21] Warning while releasing MMGP: {exc}")


def trim_process_memory() -> None:
    """Release Python, CUDA and Windows working-set memory when possible."""

    gc.collect()
    try:
        import torch

        torch.set_default_device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import ctypes

            psapi = ctypes.windll.psapi
            kernel32 = ctypes.windll.kernel32
            psapi.EmptyWorkingSet(kernel32.GetCurrentProcess())
        except Exception:
            pass


class Hunyuan3D21LowVRAMGenerator(BaseGenerator):
    MODEL_ID = "hunyuan3d-21-lowvram"
    DISPLAY_NAME = "Hunyuan Shape 2.1 Full - Low VRAM"
    VRAM_GB = 8

    def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
        super().__init__(model_dir, outputs_dir)
        self._shape_pipeline = None
        self._shape_offloader = None
        self._shape_precision = None
        self._rembg_session = None
        self._generation_lock = threading.RLock()

    def is_downloaded(self) -> bool:
        path = self.model_dir / MODEL_FILE
        return path.is_file() and path.stat().st_size == MODEL_BYTES

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_downloaded():
            self._download_weights()
        self._validate_model_file()
        self._validate_cuda()
        # Precision lives in the generation parameters, so the heavy pipeline is
        # constructed lazily in generate() and released again before generate()
        # returns.  This tiny sentinel only tells Modly that the node is ready.
        self._model = {"ready": True, "variant": "Hunyuan3D-2.1-full"}

    def unload(self) -> None:
        self._release_generation_memory()
        super().unload()

    def _release_generation_memory(self) -> None:
        """Drop every heavy CPU/GPU reference owned by this generator."""

        self._release_shape()
        self._rembg_session = None
        trim_process_memory()

    @contextmanager
    def _auto_release_after_generation(self, progress_cb):
        """Guarantee low-RAM cleanup after success, cancellation, or failure."""

        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            try:
                if succeeded:
                    self._report(progress_cb, 99, "Releasing models from RAM and VRAM...")
            finally:
                # Even a broken progress callback must not skip model cleanup.
                self._release_generation_memory()
                print("[Hunyuan3D21] Post-generation cleanup complete; model weights released from RAM/VRAM.")
            if succeeded:
                self._report(progress_cb, 100, "Done - models unloaded")

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        with self._generation_lock, self._auto_release_after_generation(progress_cb):
            if self._model is None:
                self.load()

            precision = self._normalize_precision(params.get("precision", "int8"))
            steps = max(1, min(100, int(params.get("num_inference_steps", 30))))
            octree = int(params.get("octree_resolution", 256))
            if octree not in {256, 320, 384}:
                octree = 256
            num_chunks = max(1000, min(12000, int(params.get("num_chunks", 4000))))
            guidance = max(1.0, min(10.0, float(params.get("guidance_scale", 5.0))))
            remove_background = self._as_bool(params.get("remove_background", "on"))
            seed = int(params.get("seed", -1))
            if seed < 0:
                seed = random.randint(0, 2**32 - 1)
            seed &= 0xFFFFFFFF

            self._report(progress_cb, 2, "Preparing image...")
            image = self._preprocess(image_bytes, remove_background)
            self._check_cancelled(cancel_event)

            self._report(progress_cb, 8, f"Loading full Hunyuan3D 2.1 ({precision.upper()})...")
            self._ensure_shape_pipeline(precision)
            self._check_cancelled(cancel_event)

            shape_end = 88
            mesh = self._generate_shape(
                image=image,
                steps=steps,
                octree=octree,
                num_chunks=num_chunks,
                guidance=guidance,
                seed=seed,
                start_pct=18,
                end_pct=shape_end,
                progress_cb=progress_cb,
                cancel_event=cancel_event,
            )

            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
            final_path = self.outputs_dir / f"{stem}.glb"

            self._report(progress_cb, 94, "Exporting GLB...")
            mesh.export(str(final_path))

            if not final_path.exists() or final_path.stat().st_size < 1024:
                raise RuntimeError("Generation finished without producing a valid GLB file.")
            self._report(progress_cb, 98, "GLB complete...")
            return final_path

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download

        print("[Hunyuan3D21] Downloading the full Hunyuan3D 2.1 safetensors repack...")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="Comfy-Org/hunyuan3D_2.1_repackaged",
            allow_patterns=[MODEL_FILE],
            local_dir=str(self.model_dir),
        )

    def _validate_model_file(self) -> None:
        path = self.model_dir / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"Missing model weight file: {path}")
        if path.stat().st_size != MODEL_BYTES:
            raise RuntimeError(
                f"Incomplete model weight file or unexpected version ({path.stat().st_size} bytes; expected {MODEL_BYTES}). "
                "Delete only this file, then use Repair/Download in Modly."
            )

    def _validate_cuda(self) -> None:
        # MMGP must be imported before safetensors/model code so its low-RAM
        # loader hooks are active during construction.
        from mmgp import offload as _offload  # noqa: F401
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("This extension requires a compatible NVIDIA CUDA GPU, but PyTorch did not detect one.")
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_gb < 7.0:
            raise RuntimeError(f"Only {total_gb:.1f} GB of VRAM was detected; this build requires about 8 GB.")
        print(f"[Hunyuan3D21] GPU: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)")
        try:
            import psutil

            physical_gb = psutil.virtual_memory().total / 1024**3
            pagefile_gb = psutil.swap_memory().total / 1024**3
            if physical_gb < 23 and pagefile_gb < 24:
                raise RuntimeError(
                    f"Detected {physical_gb:.1f} GB of RAM and {pagefile_gb:.1f} GB of page file. "
                    "The WinPortable reference requires 24 GB of RAM; on a 16 GB PC, "
                    "configure a 32-48 GB page file before generating."
                )
            if physical_gb < 23:
                print(
                    f"[Hunyuan3D21] WARNING: {physical_gb:.1f} GB of physical RAM is below the 24 GB minimum; "
                    f"using a {pagefile_gb:.1f} GB page file. Performance will be lower."
                )
        except ImportError:
            pass

    @staticmethod
    def _normalize_precision(value, allow_fp16=True) -> str:
        value = str(value).strip().lower()
        allowed = {"int8", "fp8"}
        if allow_fp16:
            allowed.add("fp16")
        return value if value in allowed else "int8"

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}

    @staticmethod
    def _shape_vendor_path() -> Path:
        return Path(__file__).resolve().parent / "vendor" / "hunyuan3d21" / "hy3dshape"

    def _activate_vendor(self) -> None:
        path = str(self._shape_vendor_path())
        if path not in sys.path:
            sys.path.insert(0, path)

    def _ensure_shape_pipeline(self, precision: str) -> None:
        if self._shape_pipeline is not None and self._shape_precision == precision:
            return
        self._release_shape()
        self._activate_vendor()

        from mmgp import offload, profile_type
        import torch

        torch.set_default_device("cpu")
        ckpt = self.model_dir / MODEL_FILE
        config = Path(__file__).resolve().parent / "configs" / "dit_config_2_1.yaml"
        print(f"[Hunyuan3D21] Building low-RAM pipeline from {ckpt.name}...")
        modules = {
            "transformer": None,
            "shape_vae": None,
            "conditioner": None,
        }
        try:
            pipeline = self._load_shape_pipeline_low_ram(ckpt, config, precision)
            modules.update(
                transformer=pipeline.model,
                shape_vae=pipeline.vae,
                conditioner=pipeline.conditioner,
            )
            manager = offload.profile(
                modules,
                profile_no=profile_type.VerylowRAM_LowVRAM,
                # _load_shape_pipeline_low_ram already quantizes directly from
                # the mmap-backed checkpoint.  Asking profile() to quantize
                # again would retain an avoidable second view of the weights.
                quantizeTransformer=False,
                extraModelsToQuantize=[],
                budgets={"*": 360, "transformer": 320, "shape_vae": 480, "conditioner": 360},
                workingVRAM={"transformer": 2500, "shape_vae": 3300, "conditioner": 2500},
                pinnedMemory=False,
                asyncTransfers=False,
                convertWeightsFloatTo=torch.float16,
                verboseLevel=1,
            )
        except Exception:
            pipeline = None
            trim_process_memory()
            raise

        pipeline.device = torch.device("cuda")
        pipeline.dtype = torch.float16
        self._shape_pipeline = pipeline
        self._shape_offloader = manager
        self._shape_precision = precision
        print(f"[Hunyuan3D21] Pipeline ready in {precision.upper()} with MMGP streaming.")

    @staticmethod
    def _load_shape_pipeline_low_ram(ckpt: Path, config_path: Path, precision: str):
        """Build Shape 2.1 without ever allocating its FP32 initialization.

        Tencent's ``from_single_file`` first constructs every parameter with
        PyTorch's FP32 default, copies the 6.86 GiB FP16 checkpoint into that
        allocation, and only then converts/quantizes it.  That temporary FP32
        model alone is roughly 13.7 GiB and causes heavy paging on 16 GiB PCs.

        MMGP exposes a loader designed for this case: instantiate parameters on
        the ``meta`` device, assign read-only mmap tensors from safetensors, and
        quantize each component in place.  No full FP32 or duplicate FP16 model
        is created.  Buffers stay on CPU because non-persistent buffers are not
        present in the checkpoint.
        """

        import yaml
        from accelerate import init_empty_weights
        from functools import partial
        from mmgp import offload
        from optimum.quanto import qfloat8, qint8
        import torch

        from hy3dshape.pipelines import (
            Hunyuan3DDiTFlowMatchingPipeline,
            instantiate_from_config,
        )

        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        components = {}
        try:
            # include_buffers=False is deliberate: position ids and other
            # non-persistent buffers are absent from safetensors and must be
            # materialized normally rather than left on the meta device.
            with init_empty_weights(include_buffers=False):
                components["model"] = instantiate_from_config(config["model"])
                components["vae"] = instantiate_from_config(config["vae"])
                components["conditioner"] = instantiate_from_config(config["conditioner"])

            quantized = precision != "fp16"
            qtype = qfloat8 if precision == "fp8" else qint8
            load_plan = (
                ("model", "transformer", quantized),
                ("conditioner", "conditioner", quantized),
                ("vae", "shape VAE", False),
            )
            for prefix, label, quantize_component in load_plan:
                print(
                    f"[Hunyuan3D21] Loading {label} through mmap"
                    + (f" and quantizing to {precision.upper()}..." if quantize_component else "...")
                )
                module = components[prefix]
                offload.load_model_data(
                    module,
                    str(ckpt),
                    do_quantize=quantize_component,
                    quantizationType=qtype,
                    # MMGP's generic modelPrefix also matches ".model."
                    # inside conditioner keys, then abandons filtering when it
                    # later reaches the real top-level "model." namespace.
                    # An exact startswith filter avoids leaving meta parameters
                    # such as x_embedder.weight without checkpoint data.
                    preprocess_sd=partial(
                        Hunyuan3D21LowVRAMGenerator._filter_shape_namespace,
                        prefix=prefix,
                    ),
                    writable_tensors=False,
                    verboseLevel=1,
                )
                module.eval().requires_grad_(False)
                # Release file-backed pages from the component just converted
                # before opening the next namespace of the same checkpoint.
                trim_process_memory()

            image_processor = instantiate_from_config(config["image_processor"])
            scheduler = instantiate_from_config(config["scheduler"])
            pipeline = Hunyuan3DDiTFlowMatchingPipeline(
                model=components["model"],
                vae=components["vae"],
                conditioner=components["conditioner"],
                image_processor=image_processor,
                scheduler=scheduler,
                device=None,
                dtype=None,
            )
            # MMGP owns all actual transfers.  These attributes are consumed by
            # the pipeline when it creates inputs and random generators.
            pipeline.device = torch.device("cuda")
            pipeline.dtype = torch.float16
            return pipeline
        except Exception:
            components.clear()
            trim_process_memory()
            raise

    @staticmethod
    def _filter_shape_namespace(state_dict, quantization_map, prefix: str):
        """Select only one top-level namespace from the combined checkpoint."""

        marker = f"{prefix}."
        filtered_state = {
            key[len(marker) :]: value
            for key, value in state_dict.items()
            if key.startswith(marker)
        }
        if not filtered_state:
            raise RuntimeError(f"The checkpoint does not contain the required '{marker}' namespace.")

        if quantization_map is None:
            filtered_quantization = None
        else:
            filtered_quantization = {
                key[len(marker) :]: value
                for key, value in quantization_map.items()
                if key.startswith(marker)
            }
        return filtered_state, filtered_quantization

    def _generate_shape(
        self,
        image: Image.Image,
        steps: int,
        octree: int,
        num_chunks: int,
        guidance: float,
        seed: int,
        start_pct: int,
        end_pct: int,
        progress_cb,
        cancel_event,
    ):
        import torch

        def on_step(step_idx, _timestep, _outputs):
            self._check_cancelled(cancel_event)
            pct = start_pct + int((step_idx + 1) / max(steps, 1) * (end_pct - start_pct - 8))
            self._report(progress_cb, min(pct, end_pct - 8), f"Generating 3D shape - step {step_idx + 1}/{steps}...")

        self._report(progress_cb, start_pct, "Generating 3D shape with the full 2.1 model...")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode():
            outputs = self._shape_pipeline(
                image=image,
                num_inference_steps=steps,
                octree_resolution=octree,
                guidance_scale=guidance,
                num_chunks=num_chunks,
                generator=generator,
                output_type="trimesh",
                enable_pbar=False,
                callback=on_step,
                callback_steps=1,
            )
        self._report(progress_cb, end_pct, "Mesh reconstruction complete...")
        if not outputs:
            raise RuntimeError("The 2.1 pipeline did not return a mesh.")
        return outputs[0]

    def _release_shape(self) -> None:
        release_offloader(self._shape_offloader)
        self._shape_offloader = None
        self._shape_pipeline = None
        self._shape_precision = None
        trim_process_memory()

    def _preprocess(self, image_bytes: bytes, remove_background: bool) -> Image.Image:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        image = image.convert("RGBA")
        alpha = image.getchannel("A")
        already_transparent = alpha.getextrema()[0] < 250
        if not remove_background or already_transparent:
            return image

        os.environ.setdefault("U2NET_HOME", str(self.model_dir / "_rembg"))
        import rembg

        if self._rembg_session is None:
            self._rembg_session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
        return rembg.remove(image.convert("RGB"), session=self._rembg_session).convert("RGBA")
