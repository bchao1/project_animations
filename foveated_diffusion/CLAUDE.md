# Foveated Diffusion
Project to create an animated video for the Foveated Diffusion Project.

Foveated Diffusion is a perceptually-motivated framework for efficient image/video generation using DiT-based models. It exploits the human visual system's fovea — high acuity near gaze center, degrading toward periphery — to reduce the number of tokens processed during diffusion.

Only generate the content defined in the `storyboard.md` file. Do not generate any other parts of the project.

## Environment
Use the `hsplat` conda environment.

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
The full algorithm of foveated diffusion is defined in the [context.tex](context.tex) file. The pipeline description is defined in the [pipeline.png](./pipeline.png) image.

## Animation generation
- Use assets in [data](./data/) to create the animation
- Use [source image](./data/image.png) to visualize tokenization.
- **After every new animation version**, update `storyboard.md` to accurately reflect the current implementation — layout parameters, timing, token counts, colors, gap sizes, label text, and any other details that changed. The storyboard is the ground truth for future agents.

## Storyboard visualization prompt
When the user invokes `/visualize-storyboard`, render actual PNG preview frames using Manim — one frame per layout option being evaluated. Rules:
- Write a self-contained Manim script `storyboard_previews/preview_options.py` containing one `Scene` subclass per layout option (e.g. `OptionA`, `OptionB`, …)
- Each scene constructs a static frame of the phase in question using the exact same visual style as the main animation: CMU Serif font, `#448AFF` blue HR tokens, `#00C853` green LR tokens, `RoundedRectangle` with 10% corner radius, black background
- Render every scene as a PNG using `manim -s --format=png -r 1920,1080` (save last frame only, no video) with the `hsplat` conda environment
- Save all output PNGs into `storyboard_previews/` (Manim will place them under `storyboard_previews/media/images/preview_options/`)
- After rendering, list the saved file paths so the user can open them directly