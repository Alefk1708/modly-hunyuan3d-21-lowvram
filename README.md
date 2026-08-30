# Hunyuan3D 2.1 Full Low-VRAM for Modly

Native [Modly](https://github.com/lightningpixel/modly) extension for generating **3D meshes** from an image with the **full Hunyuan Shape 2.1 model (3.3B)** on NVIDIA GPUs with limited VRAM.

- Author and maintainer: **AlefK1708**
- Version: **0.5.0**
- Repository: [AlefK1708/modly-hunyuan3d-21-lowvram](https://github.com/AlefK1708/modly-hunyuan3d-21-lowvram)
- Model: Hunyuan3D Shape 2.1 Full - this is not Mini, Turbo, or Turbo Mini.
- Backend: native Modly Python integration; **no ComfyUI, Gradio, or external server is used**.
- Output: **GLB geometry only**, with no texture generation.

## What this extension does

The **Hunyuan Shape 2.1 - Full Low VRAM** node accepts an image and returns a `mesh` output, so it can be connected normally to other compatible nodes in the Modly Workflow.

Main features:

- Full Hunyuan Shape 2.1 with the complete Safetensors checkpoint.
- INT8 recommended for GPUs with 8 GB of VRAM, experimental FP8, and FP16.
- `meta` initialization, memory-mapped Safetensors loading, and MMGP offloading to reduce RAM peaks.
- Persistent, versioned INT8/FP8 disk cache that skips repeated runtime quantization.
- Chunk-streamed dense-grid reconstruction without allocating a complete XYZ cube in RAM.
- Optional CPU background removal with `rembg`.
- Native GLB export without Blender.
- Attempts to release the heavy pipeline and RAM/VRAM caches after every generation, including errors and cancellations.

The extension returns the geometry produced by Hunyuan Shape without automatic polygon reduction. If you want a lower-poly mesh, connect a dedicated mesh optimization node later in the Modly Workflow.

## v0.5.0

This release adds a safe persistent cache for quantized weights:

- The first use of INT8 or FP8 still converts the original FP16 checkpoint locally, then saves the transformer and conditioner in MMGP/Quanto-aware Safetensors files.
- Later generations load those already-quantized files directly through read-only memory mapping, avoiding the repeated conversion pass and its temporary RAM pressure.
- INT8 and FP8 have separate caches under the model directory: `_quantized_cache/int8/` and `_quantized_cache/fp8/`.
- A manifest binds each cache to the source checkpoint, Hunyuan configuration, precision, cache format, PyTorch, MMGP, and Quanto versions. A mismatch automatically triggers a safe rebuild.
- Component files are written to unique temporary paths and the manifest is published last. Interrupted or incomplete writes are therefore never treated as valid.
- If the model directory is read-only, the disk is full, or a cache cannot be loaded, generation falls back to the original checkpoint. Cache creation is an optimization and does not become a new runtime requirement.

The cache is stored on disk, not permanently kept in RAM. The heavy pipeline continues to be released after every generation. Delete only the relevant `_quantized_cache/int8/` or `_quantized_cache/fp8/` directory if you want to force a rebuild; the original 6.86 GiB checkpoint is preserved.

## v0.4.0

This release reduces peak memory without changing model weights, inference steps, guidance, mesh resolution, or output precision:

- Streams dense-grid coordinates one reconstruction chunk at a time. The previous NumPy `meshgrid` plus `stack` path could temporarily allocate about **1.37 GB** at mesh resolution 384.
- Writes decoder logits directly into their final FP32 field instead of retaining every chunk and creating a second full tensor with `torch.cat`.
- Releases the CPU U2Net background-removal session before loading Hunyuan Shape.
- Releases the Shape pipeline before GLB export buffers are created.
- Uses lightweight cleanup between checkpoint components so hot pages are not evicted and immediately reloaded; aggressive cleanup remains enabled at lifecycle boundaries.
- Adds CPU-only regression tests for grid ordering, chunk bounds, checkpoint namespace filtering, and cleanup order.

The streamed coordinates use the same FP32 axes and `ij` ordering as the original decoder, and the logits are converted to FP32 at the same point in the computation. These changes target allocation overhead rather than reducing quality settings.

## v0.3.1

This version consolidates the **geometry-only** edition and keeps the repository aligned with the current Modly extension format:

- Keeps a single `image -> mesh` node.
- Keeps `manifest.json` and `generator.py`, required by the extension system.
- Uses `setup.py` to create the isolated Python environment, following current official Modly extension patterns.
- Declares Python dependencies in `requirements.txt`.
- Removes diagnostic scripts, test files, caches, development artifacts, and Paint/texture components.
- Removes the previous internal face-count control and leaves mesh optimization to downstream Workflow nodes.
- Keeps the small memory-cleanup routine directly inside `generator.py`.
- Includes only the Tencent Shape 2.1 inference code and configuration required to run the model.
- Preserves the licenses, redistribution notice, and required credits.

This version does not request any Paint checkpoint, Delight model, Paint-specific DINO model, or texture rasterizer.

## Requirements

The primary target for this build is Windows 10/11 with an NVIDIA GPU and approximately **8 GB of VRAM or more**.

| Item | Minimum for this build | Recommended |
|---|---:|---:|
| GPU | NVIDIA CUDA with about **8 GB VRAM** | NVIDIA with 8 GB or more |
| NVIDIA driver | **576.57** | Current stable driver |
| Physical RAM | **24 GB** according to the WinPortable reference | 32 GB or more |
| PC with 16 GB RAM | Large page file on an SSD | 32-48 GB page file |
| Modly Python | 3.10, 3.11, or 3.12 | Python supplied by Modly |
| Free space for a clean installation | About 20 GB | 30 GB or more |

With 16 GB of physical RAM, the page file is a fallback and loading/quantization may temporarily make Windows less responsive. v0.4.0 avoids full FP32 initialization, streams the reconstruction grid, and drops the heavy pipeline before export, but no software configuration can guarantee the complete absence of memory peaks for every input and setting.

## Downloaded model

The only large 3D checkpoint requested by the node is:

| File | Size |
|---|---:|
| `hunyuan_3d_v2.1.safetensors` | 7,365,943,290 bytes (about 6.86 GiB) |

Weight source: `Comfy-Org/hunyuan3D_2.1_repackaged`.

That repository only hosts the **repackaged Safetensors** used by this extension. The generator does not start or depend on ComfyUI.

INT8 and FP8 are produced locally from the same Safetensors checkpoint. Starting with v0.5.0, the first successful conversion is cached beside the model and reused on later runs. Updating the extension does not require a separate download, and an existing complete Shape checkpoint in Modly's model directory can be reused.

The local cache uses several additional GiB of disk space for each precision actually selected. It is created on demand; choosing only INT8 does not create an FP8 cache.

The first background-removal run may download the U2Net model used by `rembg` if it is not already cached.

## Install from GitHub

1. Publish these files at the root of your GitHub repository.
2. In Modly, open **Models -> Install from GitHub**.
3. Paste:

   ```text
   https://github.com/AlefK1708/modly-hunyuan3d-21-lowvram
   ```

4. Wait for `setup.py` to create the `venv` and install PyTorch and the required dependencies.
5. Download the model from the node when Modly requests it.

When updating an older installation, use Modly's update/repair mechanism. The model-weight directory is separate from the extension directory, so a complete Shape checkpoint that is already downloaded should not need to be downloaded again.

## First test on an 8 GB GPU

| Option | Starting value |
|---|---:|
| Precision | INT8 |
| Steps | 20 or 30 |
| Mesh Resolution | 256 |
| Chunks | 2,000 or 4,000 |
| Guidance | 5.0 |
| Remove Background | On |

Use an image containing one complete object that is clearly separated from the background.

## Modly Workflow

Basic example:

```text
Image -> Hunyuan Shape 2.1 - Full Low VRAM -> Mesh Optimizer / Export / Scene
```

The node output type is `mesh`, exactly as declared in `manifest.json`, so it remains compatible with other nodes that accept mesh input.

## RAM and VRAM

The first INT8 or FP8 preparation can use a significant amount of RAM, CPU, and disk activity. The loader processes the checkpoint component by component through memory mapping and uses MMGP/Quanto to avoid keeping a second complete FP32 copy of the model. Once the persistent cache is valid, later runs skip quantization and mmap the smaller prequantized component files. During mesh reconstruction, the decoder keeps only the current coordinate chunk instead of the complete dense XYZ grid.

At the end of generation, the UI should reach the memory-release stage and the log records:

```text
[Hunyuan3D21] Post-generation cleanup complete; model weights released from RAM/VRAM.
```

Windows can still show loaded libraries, file cache, reserved memory, or standby memory for the Modly extension process after cleanup. That reading alone does not mean the full Hunyuan pipeline is still loaded.

The operating system may use otherwise-free RAM as a file cache for the persistent Safetensors. This memory is reclaimable and is different from retaining the Hunyuan pipeline as live Python model objects.

## Common issues

### CUDA out of memory

- Use INT8.
- Use mesh resolution 256.
- Reduce chunks to 2,000.
- Close programs that are using the GPU.
- Generate one model at a time.

### RAM is full or the PC becomes unresponsive while loading

- On a 16 GB PC, make sure a 32-48 GB page file is configured on an SSD.
- Close browsers, games, and other memory-heavy applications.
- Let the first INT8/FP8 run finish writing its cache; later startup should be faster and have less conversion pressure.
- Wait for post-generation cleanup before starting another run.
- For frequent use, 32 GB of physical RAM is much more comfortable.

### Quantized cache is rebuilt or cannot be saved

- A rebuild after changing the extension, checkpoint, precision, or pinned runtime versions is expected and protects against incompatible tensors.
- Make sure the drive containing Modly's model directory has several GiB of free space and is writable.
- Generation can continue from the original checkpoint even when cache creation fails.
- To repair a suspected cache without redownloading the model, delete only `_quantized_cache/int8/` or `_quantized_cache/fp8/`.

### `x_embedder.weight has no data`

The current loader filters the `model.`, `conditioner.`, and `vae.` namespaces of the combined checkpoint separately. Do not redownload the 6.86 GiB checkpoint only because of this error if the file has the exact size shown above.

## Repository files

```text
README.md
manifest.json
generator.py
requirements.txt
setup.py
configs/dit_config_2_1.yaml
vendor/hunyuan3d21/hy3dshape/...
tests/test_memory_optimizations.py
LICENSE
LICENSE-HUNYUAN-3D-2.1.txt
Notice.txt
```

`configs/` and `vendor/` are Shape 2.1 runtime dependencies. The lightweight `tests/` directory is source-only and is not needed during generation. The public ZIP does not contain a `venv`, model weights, caches, Paint code, or diagnostic scripts.

## Licenses and credits

- **AlefK1708** - Modly integration, packaging, and maintenance of this extension.
- **Lightning Pixel** - Modly and its extension contract.
- **Tencent Hunyuan** - Hunyuan3D 2.1 architecture, Shape code, and model. The inference subset included in `vendor/` is derived from `Tencent-Hunyuan/Hunyuan3D-2.1`, commit `82920d643c0dc2f7bfd7255f45f62d386edfe60c`.
- **YanWenKun** - `Hunyuan3D-2-WinPortable`, commit `b913eedb15a63e5eaf7e32f1cae02c560af9a7f8`, used as a reference for the Windows/low-VRAM strategy.
- **deepbeepmeep** - MMGP and low-VRAM offloading techniques used by this adapter.
- **Comfy-Org** - hosting for the combined Safetensors repack; it is not used as a backend.

### Independence notice

This is an independent community extension maintained by **AlefK1708**. Tencent is not affiliated with, associated with, sponsoring, or endorsing this extension or Modly. The Tencent Hunyuan name is used only to identify the upstream model and software on which this extension depends.

The adapter is distributed under GPL-3.0 because of the WinPortable base used during adaptation; see `LICENSE`. Hunyuan3D 2.1 code and weights remain subject to `LICENSE-HUNYUAN-3D-2.1.txt` and `Notice.txt`.

Before public redistribution or commercial use, read the complete Hunyuan license. It has its own territorial and use restrictions. This repository does not grant any additional rights to Tencent's model.
