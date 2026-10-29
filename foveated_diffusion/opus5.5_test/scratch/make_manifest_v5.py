"""Build slides_v5/sections.json from the v5 1080p60 section mp4s and their QA last frames."""
import json, glob, os, subprocess
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEC = f"{R}/media/videos/animation_foveated_diffusion_v5/1080p60/sections"
QA = f"{R}/slides_v5/qa"
D = {
 "S01_TokenCost": ["Dog image with 16×16 patch grid lines",
   "Image replaced by 256 colored patch tokens",
   "Face token (amber) and door token (purple) boxed with legend",
   "All tokens recolored blue; label: 16×16 = 256 tokens",
   "UNUSED cost beat: attention matrix, 65,536 query–key pairs",
   "UNUSED cost beat: query token and its attention row highlighted",
   "UNUSED cost beat: small square, N tokens / N² pairs",
   "UNUSED cost beat: 2× resolution square, 4N tokens / 16N² pairs"],
 "S02_Vision": ["Visual acuity vs eccentricity curve with fovea band",
   "Acuity plot plus farmer photo with red gaze dot",
   "Farmer photo blurred away from gaze, dashed fovea circle",
   "Gaze dot moved to hat brim; sharp region follows",
   "Foveated rendering equation with five-image visual equation",
   "Question card: can a diffusion model spend tokens this way?"],
 "S03_Tokenization": ["Full 16×16 grid of blue high-res tokens",
   "Mask M panel; periphery dimmed, fovea outlined",
   "Periphery merged into green low-res tokens, fovea stays blue",
   "Token count: m = 48, 52 low-res, L = 100, 39%",
   "Flattened row: one sequence of L = 100 tokens"],
 "S04_RoPE": ["Uniform token row 0–7 with rotating RoPE dials",
   "Tokens 4–7 merged into two green tokens; dials show ?",
   "High-res query row: low-res keys at indices 4, 6",
   "Low-res query row: re-indexed 0, 1, 2, 3",
   "High-res and low-res region brackets under the rows"],
 "S05_Naive": ["Full high-res row: horse, guitar, farmer",
   "Naïve mixed-res row with red failure labels"],
 "S06_Training": ["Training image x (lion)",
   "Bicubic downsampled copy Down(x) below x",
   "VAE encoders produce z_high (16×16) and z_low (8×8)",
   "Mask M merges both into foveated token grid z_fov",
   "Flow-matching equation, z_fov tokens at t = 0.55 slider",
   "DiT+LoRA box maps noisy tokens to velocity v_θ",
   "Flow-matching loss equation below the DiT pipeline",
   "Training details table: models, data, LoRA, masks"],
 "S07_Generation": ["Foveated noise tokens z_1 ∼ N(0, I)",
   "DiT+LoRA loop; tokens denoised to clean z_0",
   "z_0 split into z_high and z_low grids",
   "VAE decoders produce sharp fovea x_high and low-res x_low",
   "Blend equation with five-image visual equation (lion)",
   "Large lion output, fovea outline, 48 + 52 = 100 tokens"],
 "S08_Masks": ["Fig. 9 grid: position, size, disjoint masks, naïve vs ours",
   "Random-mask vs saliency-guided flower columns",
   "Saliency columns plus four application examples",
   "Video frame box with cubic-spline gaze path, control points",
   "White fovea circle back at start of spline path",
   "Spline diagram plus five generated fox video frames"],
 "S09_Results": ["Speedup vs token ratio: image 2.2×, video 4.2×",
   "Speedup plot plus HPSv2.1 quality table",
   "Qualitative grid: full, naïve, ours for four subjects",
   "User-study protocol strip: fixate, image A, image B, choose",
   "Protocol plus preference bars 47.3%, 87.4%, 90.8%, 50% line",
   "Fox video rows full/naïve/ours, VBench scores, 3.5× faster"],
 "S10_Outlook": ["Seam artifact crops with cause and fix text",
   "Three-level foveated token grid (1×, ½×, ¼×)",
   "Three-level grid plus gaze-following token field with red dot",
   "Summary title with idea 1: spend tokens by eccentricity",
   "Summary ideas 1–2: LoRA post-training targets",
   "Summary ideas 1–3: phase-aligned RoPE",
   "Summary plus ≈2× images, ≈4× video, indistinguishable",
   "End card: title, arXiv link, authors"],
}
out = []
for sc, descs in D.items():
    files = sorted(glob.glob(f"{SEC}/{sc}_[0-9][0-9][0-9][0-9]_*.mp4"))
    assert len(files) == len(descs), (sc, len(files), len(descs))
    for f, d in zip(files, descs):
        base = os.path.basename(f)[:-4]
        idx, name = int(base[len(sc) + 1:len(sc) + 5]), base[len(sc) + 6:]
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                             "-of", "csv=p=0", f]).decode())
        lf = f"{QA}/{base}_last.png"
        assert os.path.exists(lf), lf
        assert len(d.split()) <= 12, d
        out.append(dict(scene=sc, index=idx, name=name, file=f, duration_s=round(dur, 2),
                        last_frame=lf, description=d))
json.dump(out, open(f"{R}/slides_v5/sections.json", "w"), indent=2, ensure_ascii=False)
print(len(out), "sections")
