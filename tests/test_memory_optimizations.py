"""CPU-only regression tests for the low-memory execution paths."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_generator_module():
    base_module = types.ModuleType("services.generators.base")

    class BaseGenerator:
        def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
            self.model_dir = model_dir
            self.outputs_dir = outputs_dir
            self._model = None

        def unload(self) -> None:
            self._model = None

        def _report(self, progress_cb, pct: int, step: str) -> None:
            if progress_cb:
                progress_cb(pct, step)

        @staticmethod
        def _check_cancelled(cancel_event) -> None:
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("cancelled")

    base_module.BaseGenerator = BaseGenerator
    services_module = types.ModuleType("services")
    generators_module = types.ModuleType("services.generators")
    sys.modules["services"] = services_module
    sys.modules["services.generators"] = generators_module
    sys.modules["services.generators.base"] = base_module

    spec = importlib.util.spec_from_file_location("generator_under_test", ROOT / "generator.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)
        self.device = "cpu"
        self.dtype = self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    def to(self, device=None, dtype=None):
        del device
        return FakeTensor(self.array.astype(dtype or self.dtype, copy=False))

    def view(self, shape):
        return FakeTensor(self.array.reshape(shape))

    reshape = view

    def copy_(self, other):
        self.array[...] = other.array
        return self

    def __getitem__(self, key):
        if isinstance(key, FakeTensor):
            key = key.array
        elif isinstance(key, tuple):
            key = tuple(item.array if isinstance(item, FakeTensor) else item for item in key)
        return FakeTensor(self.array[key])

    def __sub__(self, other):
        other = other.array if isinstance(other, FakeTensor) else other
        return FakeTensor(self.array - other)

    def __mul__(self, other):
        other = other.array if isinstance(other, FakeTensor) else other
        return FakeTensor(self.array * other)


def load_volume_decoder_module():
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = FakeTensor
    torch_module.FloatTensor = FakeTensor
    torch_module.float32 = np.float32
    torch_module.max_arange_size = 0

    def no_grad():
        return lambda function: function

    def arange(start, stop, device=None):
        del device
        torch_module.max_arange_size = max(torch_module.max_arange_size, stop - start)
        return FakeTensor(np.arange(start, stop, dtype=np.int64))

    torch_module.no_grad = no_grad
    torch_module.from_numpy = lambda value: FakeTensor(value)
    torch_module.arange = arange
    torch_module.div = lambda value, divisor, rounding_mode=None: FakeTensor(
        np.floor_divide(value.array, divisor)
    )
    torch_module.stack = lambda values, dim=0: FakeTensor(
        np.stack([value.array for value in values], axis=dim)
    )
    torch_module.empty = lambda shape, dtype, device=None: FakeTensor(np.empty(shape, dtype=dtype))

    nn_module = types.ModuleType("torch.nn")
    functional_module = types.ModuleType("torch.nn.functional")
    nn_module.functional = functional_module
    torch_module.nn = nn_module
    sys.modules["torch"] = torch_module
    sys.modules["torch.nn"] = nn_module
    sys.modules["torch.nn.functional"] = functional_module

    einops_module = types.ModuleType("einops")
    einops_module.repeat = lambda value, _pattern, b: FakeTensor(
        np.repeat(value.array[np.newaxis, ...], b, axis=0)
    )
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **_kwargs: iterable
    sys.modules["einops"] = einops_module
    sys.modules["tqdm"] = tqdm_module

    package_names = ("hy3dshape", "hy3dshape.models", "hy3dshape.models.autoencoders")
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    attention_blocks = types.ModuleType("hy3dshape.models.autoencoders.attention_blocks")
    attention_blocks.CrossAttentionDecoder = object
    attention_processors = types.ModuleType("hy3dshape.models.autoencoders.attention_processors")
    attention_processors.FlashVDMCrossAttentionProcessor = object
    attention_processors.FlashVDMTopMCrossAttentionProcessor = object
    utils_module = types.ModuleType("hy3dshape.utils")
    utils_module.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    sys.modules[attention_blocks.__name__] = attention_blocks
    sys.modules[attention_processors.__name__] = attention_processors
    sys.modules[utils_module.__name__] = utils_module

    path = ROOT / "vendor/hunyuan3d21/hy3dshape/hy3dshape/models/autoencoders/volume_decoders.py"
    name = "hy3dshape.models.autoencoders.volume_decoders_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, torch_module


class GeneratorMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_generator_module()

    def test_namespace_filter_mutates_private_mapping_without_tensor_copy(self):
        keep = object()
        state = {"model.weight": keep, "vae.weight": object(), "conditioner.weight": object()}
        quantization = {"model.weight": {"qtype": "int8"}, "vae.weight": {"qtype": "fp16"}}

        result, quantization_result = self.module.Hunyuan3D21LowVRAMGenerator._filter_shape_namespace(
            state, quantization, "model"
        )

        self.assertIs(result, state)
        self.assertIs(quantization_result, quantization)
        self.assertEqual(list(result), ["weight"])
        self.assertIs(result["weight"], keep)
        self.assertEqual(quantization_result, {"weight": {"qtype": "int8"}})

    def test_model_is_released_before_glb_export(self):
        events = []

        class Mesh:
            @staticmethod
            def export(path):
                events.append("export")
                Path(path).write_bytes(b"g" * 1024)

        with tempfile.TemporaryDirectory() as directory:
            gen = self.module.Hunyuan3D21LowVRAMGenerator(Path(directory), Path(directory))
            gen._model = {"ready": True}
            gen._preprocess = lambda *_args: object()
            gen._release_background_remover = lambda: None
            gen._ensure_shape_pipeline = lambda _precision: None
            gen._generate_shape = lambda **_kwargs: Mesh()
            gen._release_shape = lambda *, aggressive=True: events.append(f"release:{aggressive}")
            gen._release_generation_memory = lambda: events.append("final-cleanup")

            output = gen.generate(b"image", {}, cancel_event=threading.Event())

        self.assertTrue(output.name.endswith(".glb"))
        self.assertLess(events.index("release:False"), events.index("export"))
        self.assertEqual(events[-1], "final-cleanup")

    def test_quantized_cache_is_reusable_and_invalidates_when_source_changes(self):
        generator_class = self.module.Hunyuan3D21LowVRAMGenerator

        class TensorMetadata:
            @staticmethod
            def numel():
                return 16

            @staticmethod
            def element_size():
                return 1

        class Module:
            @staticmethod
            def state_dict():
                return {"weight._data": TensorMetadata()}

        class Offload:
            calls = []

            @classmethod
            def save_model(cls, _module, path, **kwargs):
                cls.calls.append((Path(path), kwargs))
                Path(path).write_bytes(b"q" * 2048)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / self.module.MODEL_FILE
            config = root / "config.yaml"
            checkpoint.write_bytes(b"source-checkpoint")
            config.write_bytes(b"model: test\n")
            components = {"model": Module(), "conditioner": Module()}

            generator_class._save_quantized_cache(
                components,
                checkpoint,
                config,
                "int8",
                Offload,
            )

            manifest_path = generator_class._quantized_cache_manifest_path(checkpoint, "int8")
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(len(Offload.calls), 2)
            for temporary_path, kwargs in Offload.calls:
                self.assertFalse(temporary_path.exists())
                self.assertFalse(kwargs["do_quantize"])

            plan = generator_class._load_quantized_cache_plan(checkpoint, config, "int8")
            self.assertEqual(set(plan), {"model", "conditioner"})
            self.assertTrue(all(path.is_file() for path in plan.values()))

            checkpoint.write_bytes(b"changed-source-checkpoint")
            self.assertIsNone(
                generator_class._load_quantized_cache_plan(checkpoint, config, "int8")
            )

    def test_incomplete_cache_never_publishes_a_manifest(self):
        generator_class = self.module.Hunyuan3D21LowVRAMGenerator

        class Module:
            @staticmethod
            def state_dict():
                return {}

        class FailingOffload:
            calls = 0

            @classmethod
            def save_model(cls, _module, path, **_kwargs):
                cls.calls += 1
                if cls.calls == 2:
                    raise OSError("simulated disk failure")
                Path(path).write_bytes(b"q" * 2048)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / self.module.MODEL_FILE
            config = root / "config.yaml"
            checkpoint.write_bytes(b"source-checkpoint")
            config.write_bytes(b"model: test\n")
            manifest_path = generator_class._quantized_cache_manifest_path(checkpoint, "fp8")
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps({"old": True}), encoding="utf-8")

            with self.assertRaisesRegex(OSError, "simulated disk failure"):
                generator_class._save_quantized_cache(
                    {"model": Module(), "conditioner": Module()},
                    checkpoint,
                    config,
                    "fp8",
                    FailingOffload,
                )

            self.assertFalse(manifest_path.exists())
            self.assertIsNone(
                generator_class._load_quantized_cache_plan(checkpoint, config, "fp8")
            )

    def test_cache_manifest_cannot_redirect_component_paths(self):
        generator_class = self.module.Hunyuan3D21LowVRAMGenerator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / self.module.MODEL_FILE
            config = root / "config.yaml"
            checkpoint.write_bytes(b"source-checkpoint")
            config.write_bytes(b"model: test\n")
            manifest_path = generator_class._quantized_cache_manifest_path(checkpoint, "int8")
            manifest_path.parent.mkdir(parents=True)

            manifest = generator_class._quantized_cache_context(checkpoint, config, "int8")
            manifest["components"] = {
                "model": {"file": "../transformer.safetensors", "size": 2048},
                "conditioner": {"file": "conditioner.safetensors", "size": 2048},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            self.assertIsNone(
                generator_class._load_quantized_cache_plan(checkpoint, config, "int8")
            )

    def test_valid_cache_skips_transformer_and_conditioner_quantization(self):
        generator_class = self.module.Hunyuan3D21LowVRAMGenerator

        class FakeModule:
            def __init__(self, name):
                self.name = name

            def eval(self):
                return self

            def requires_grad_(self, _enabled):
                return self

        class Pipeline:
            def __init__(self, **components):
                self.__dict__.update(components)

        calls = []

        class Offload:
            @staticmethod
            def load_model_data(module, path, **kwargs):
                calls.append((module.name, Path(path), kwargs))

        pipelines_module = types.ModuleType("hy3dshape.pipelines")
        pipelines_module.Hunyuan3DDiTFlowMatchingPipeline = Pipeline
        pipelines_module.instantiate_from_config = lambda name: FakeModule(name)
        hy3dshape_module = types.ModuleType("hy3dshape")
        hy3dshape_module.__path__ = []
        accelerate_module = types.ModuleType("accelerate")
        accelerate_module.init_empty_weights = lambda **_kwargs: nullcontext()
        mmgp_module = types.ModuleType("mmgp")
        mmgp_module.offload = Offload
        quanto_module = types.ModuleType("optimum.quanto")
        quanto_module.qfloat8 = object()
        quanto_module.qint8 = object()
        optimum_module = types.ModuleType("optimum")
        optimum_module.__path__ = []
        torch_module = types.ModuleType("torch")
        torch_module.device = lambda value: value
        torch_module.float16 = "float16"

        injected_modules = {
            "accelerate": accelerate_module,
            "hy3dshape": hy3dshape_module,
            "hy3dshape.pipelines": pipelines_module,
            "mmgp": mmgp_module,
            "optimum": optimum_module,
            "optimum.quanto": quanto_module,
            "torch": torch_module,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / self.module.MODEL_FILE
            config = root / "config.yaml"
            checkpoint.write_bytes(b"source-checkpoint")
            config.write_text(
                "model: model\n"
                "vae: vae\n"
                "conditioner: conditioner\n"
                "image_processor: image_processor\n"
                "scheduler: scheduler\n",
                encoding="utf-8",
            )
            cache_dir = generator_class._quantized_cache_dir(checkpoint, "int8")
            cache_dir.mkdir(parents=True)
            component_metadata = {}
            for component, filename in self.module.CACHE_COMPONENT_FILES.items():
                component_path = cache_dir / filename
                component_path.write_bytes(b"q" * 2048)
                component_metadata[component] = {"file": filename, "size": 2048}
            manifest = generator_class._quantized_cache_context(checkpoint, config, "int8")
            manifest["components"] = component_metadata
            generator_class._atomic_write_json(cache_dir / self.module.CACHE_MANIFEST_FILE, manifest)

            with mock.patch.dict(sys.modules, injected_modules):
                pipeline = generator_class._load_shape_pipeline_low_ram(
                    checkpoint,
                    config,
                    "int8",
                )

        self.assertEqual(pipeline.model.name, "model")
        self.assertEqual([name for name, _path, _kwargs in calls], ["model", "conditioner", "vae"])
        self.assertEqual(calls[0][1].name, "transformer.safetensors")
        self.assertEqual(calls[1][1].name, "conditioner.safetensors")
        self.assertEqual(calls[2][1], checkpoint)
        self.assertTrue(all(call_kwargs["do_quantize"] is False for _name, _path, call_kwargs in calls))
        self.assertNotIn("preprocess_sd", calls[0][2])
        self.assertIn("preprocess_sd", calls[2][2])


class VolumeDecoderMemoryTests(unittest.TestCase):
    def test_streamed_grid_matches_dense_ij_grid_and_stays_chunk_bounded(self):
        module, fake_torch = load_volume_decoder_module()
        bounds = [-1.01, -1.01, -1.01, 1.01, 1.01, 1.01]
        resolution = 5
        dense, grid_size, _ = module.generate_dense_grid_points(
            np.array(bounds[:3]), np.array(bounds[3:]), resolution, indexing="ij"
        )
        expected = dense.astype(np.float16).sum(axis=-1, dtype=np.float16).astype(np.float32)

        latents = FakeTensor(np.zeros((1, 4, 2), dtype=np.float16))

        def decoder(*, queries, latents):
            del latents
            return FakeTensor(queries.array.sum(axis=-1, keepdims=True, dtype=np.float16))

        output = module.VanillaVolumeDecoder()(
            latents,
            decoder,
            bounds=bounds,
            num_chunks=7,
            octree_resolution=resolution,
            enable_pbar=False,
        )

        np.testing.assert_array_equal(output.array[0], expected.reshape(grid_size))
        self.assertLessEqual(fake_torch.max_arange_size, 7)


if __name__ == "__main__":
    unittest.main()
