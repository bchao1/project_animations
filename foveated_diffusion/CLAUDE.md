# Foveated Diffusion — Project Animations

Project to create animated videos for the Foveated Diffusion research project.

Foveated Diffusion is a perceptually-motivated framework for efficient image/video generation using DiT-based models. It exploits the human visual system's fovea — high acuity near gaze center, degrading toward periphery — to reduce the number of tokens processed during diffusion.

## Environment
Use the `hsplat` conda environment.

## Folder Structure

Each animation lives in its own subfolder:

- `tokenization/` — Foveated Tokenization animation (current: `animation_tokenization_v9.py`)
- `overview/` — Full pipeline overview animation (`animation.py`, in progress)

Each subfolder contains its own `storyboard.md` and optionally a `CLAUDE.md` for animation-specific context. Only generate content defined in the subfolder's `storyboard.md`.

Each subfolder may also have its own `data/` folder for animation-specific assets. Shared assets live at the top level and should be referenced by absolute path:
- `data/image.png` — source image for tokenization visualization
- `context.tex` — full algorithm reference
- `pipeline.png` — visual pipeline diagram

## Output Conventions

These rules apply to every animation subfolder:

**Render outputs** go into the subfolder's own `media/` directory. Pass `--media_dir media` (or set `media_dir` in config) when rendering so Manim writes there instead of the project root.

**Render formats** (render in this order):
1. 480p, 15fps first for testing: `manim -ql --media_dir media <script.py> <SceneName>`
2. 1080p, 60fps for the final version: `manim -qh --media_dir media <script.py> <SceneName>`

**Script versioning** — save each new version as `animation_<name>_vN.py` inside the subfolder. Never overwrite an existing file. The current version number is tracked at the top of `storyboard.md`.

**Storyboard discipline** — after every new animation version, update the subfolder's `storyboard.md` to accurately reflect the current implementation: layout parameters, timing, token counts, colors, gap sizes, label text, and any other details that changed. The storyboard is the ground truth for future agents.

## Storyboard Visualization

When the user invokes `/visualize-storyboard` in any subfolder, render actual PNG preview frames using Manim — one frame per layout option being evaluated. Rules:
- Write a self-contained Manim script `storyboard_previews/preview_options.py` (inside the animation subfolder) containing one `Scene` subclass per layout option (e.g. `OptionA`, `OptionB`, …)
- Each scene constructs a static frame of the phase in question using the shared visual style defined above
- Render every scene as a PNG using `manim -s --format=png -r 1920,1080` (save last frame only, no video) with the `hsplat` conda environment
- Output PNGs go into the subfolder's `storyboard_previews/media/images/preview_options/`
- After rendering, list the saved file paths so the user can open them directly

## Shared Visual Style

All animations in this project **must** conform to these style constants:

| Property | Value |
|---|---|
| Font | CMU Serif (`cmunrm.ttf`, `cmunbx.ttf`, `cmunti.ttf`, `cmunbi.ttf` from `/home/brianchc/fonts/cmu-serif/`) |
| Background | Black |
| HR token color | `#448AFF` (blue) |
| LR token color | `#00C853` (green) |
| Token shape | `RoundedRectangle`, corner radius = 10% of side length, exactly square |
| Gap between tokens (1D) | 20% of token side |
| Grid cell margin | 10% of cell width/height |
| Text | White; use `self.add()` for instant appearance (no fade-in animations on text) |

Visuals must never overlap. Text should appear instantly, not animate in.

## Key Concepts

### Foveation Mask
- Binary mask `M ∈ {0,1}^(h×w)` specifying high-resolution (HR) token locations
- `M(i,j)=1`: high-resolution foveal region
- `M(i,j)=0`: low-resolution peripheral region

### Foveated Tokenization
- Standard DiT uses all `h×w` tokens at full resolution
- Foveated Diffusion operates in foveated token space `R^(c×L)` with `L ≪ h·w`
- One low-resolution token covers a 2×2 block of high-resolution tokens (4× savings in periphery)
- Total sequence length: `L = m + (h·w - m) / 4`, where `m` = number of HR tokens in mask

### Foveated Generation (Inference)
1. Sample Gaussian noise `z_1^fov ~ N(0,I)` in foveated token space (length `L`)
2. Iteratively denoise from `t=1` to `t=0` → clean foveated token sequence `z_0^fov`
3. Split: `(z_0^high, z_0^low) = Split(z_0^fov, M)`
4. Decode: `x^high = D(z_0^high)`, `x^low = D(z_0^low)`
5. Blend: `x^fov = M' ⊙ x^high + (1 - M') ⊙ Up(x^low)`, where `M' = Up(M)`

### Foveated Training
- Post-training via LoRA on pretrained DiTs
- For each image `x`, build mixed-resolution-consistent training target:
  - `z_0^high = E(x)` — encode full resolution
  - `z_0^low = E(Down(x))` — encode downsampled version
  - `z_0^fov = Merge(z_0^high, z_0^low, M)` — merged foveated target
- Train with standard flow-matching objective on the foveated token sequence

### Mixed-Resolution RoPE
- Standard RoPE assumes uniform grid; must be adapted for mixed-resolution tokens
- Follows CRPA (Wu et al.): aligns key RoPE phases with query RoPE phases based on token resolution

### Masking Strategies
- **Randomized masks**: gaze-agnostic, supports arbitrary fovea location
- **Saliency-guided masks**: fovea covers visually salient objects
- **Bounding-box masks**: rectangular foveal regions

### Results
- Up to 2× speedup for image generation, 4× for video generation
- Achieves perceptual quality comparable to full high-resolution synthesis

## References
The full algorithm is defined in [context.tex](context.tex). The pipeline is shown in [pipeline.png](pipeline.png).
