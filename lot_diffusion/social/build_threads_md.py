"""Writes threads.md from tweets.py (single source of truth for post text)."""
from tweets import T, weight

PEOPLE = {
    "A": ("@GordonWetzstein", "Main thread (7 posts). Big picture: the problem, the layout idea, the method, results, numbers."),
    "B": ("@BrianCChao", "Quote of A1 (4 posts). Origin story (Foveated Diffusion → any layout); layouts as control; the two DiT changes; the budget knob."),
    "C": ("@YarivLior", "Quote of A1 or B1 (4 posts). Plans already exist; attention cost; how much a layout saves; quality at the same budget."),
    "D": ("<4th co-author>", "Quote of A5, the video post (4 posts). Video comparison; why the naive approach fails; the projection; image comparison."),
}
SCHEDULE = [
    ("T+0", "A", "Post the 7-post main thread A1–A7."),
    ("T+10 min", "B", "Quote A1 with B1, reply-chain B2–B4 under B1. Repost A1."),
    ("T+45 min", "C", "Quote A1 (or B1) with C1, reply-chain C2–C4. Repost A1 and B1."),
    ("T+2 h", "D", "Quote A5 (the driving-video post) with D1, reply-chain D2–D4. Repost A1."),
    ("T+2–4 h", "A", "Repost B1, C1, D1. Reply to B2 or D5 with one sentence of your own (not a copy of the post)."),
    ("T+4 h", "B, C", "Repost each other's quote and D1."),
    ("Day 1", "A", "Standalone post with the 1-min cut ('In case you missed it…'), linking to A1."),
    ("Day 2", "any", "Reply to A7 with the YouTube link of the 5-min explainer once it is up."),
]


def main():
    out = ["# Level-of-Token Diffusion — X thread plan (4 people)", "",
           "Every thread clip is used exactly once across the four threads, so the threads look different even when",
           "they are reposted or quoted side by side. Post text lives in `tweets.py` (run it to check lengths; X counts a",
           "URL as 23 characters). Clips: `../clips/media/final/1080p60/LoT_<clip>.mp4` (1920×1080, 60 fps, 8–20 s, no audio; each is one beat of the 5-min teaser).",
           "Teasers: `../media/final/LevelOfTokenDiffusion_5min_v2_1080p60.mp4`, "
           "`../teaser_1min/media/final/LevelOfTokenDiffusion_1min_v2_1080p60.mp4`.", "",
           "## Before posting", "",
           "- The paper is an anonymous ICLR 2027 submission. Link an arXiv / project page, not the anonymous supplementary",
           "  site, and check the venue's current policy on promoting submissions under review.",
           "- Fill in the `<…URL>` placeholders, the co-author handles in A7, and who person D is.",
           "- The clips have no sound; X autoplays muted, so nothing is lost.", "",
           "## Who posts what", "", "| | account | role |", "|---|---|---|"]
    for k, (h, r) in PEOPLE.items():
        out.append(f"| {k} | {h} | {r} |")
    out += ["", "## Schedule (T = launch time)", "", "| when | who | action |", "|---|---|---|"]
    for w, who, a in SCHEDULE:
        out.append(f"| {w} | {who} | {a} |")
    for P in "ABCD":
        h, r = PEOPLE[P]
        out += ["", f"## Thread {P} — {h}", "", r, ""]
        for k, (clip, s) in T.items():
            if not k.startswith(P):
                continue
            media = f"`LoT_{clip}.mp4`" if clip.startswith("K") else clip
            out += [f"### {k} · media: {media} · {weight(s)}/280", "", "```", s, "```", ""]
    out += ["## Clip index", "", "| clip | what it shows | thread |", "|---|---|---|"]
    desc = {
        "K01_TokenCost": "corgi image → 16 × 16 patch-colour tokens; a face token vs a sky token cost the same; 256 tokens",
        "K02_Attention": "attention matrix with a moving query row; N → 4N tokens, N² → 16N² pairs",
        "K03_Plan": "plan boxes → agent-painted detail map → generated image; the plan never sets compute",
        "K04_Merge": "256 tokens merge by the detail map (2×2, then 4×4) → count column, L = 133 (toy)",
        "K05_Shapes": "square tokens become rectangles (2×4, 8×4); R_i equation; L = 126 (toy)",
        "K06_Sequence": "the layout flattens to a 126-token sequence; attention area ≈ 24% (toy)",
        "K07_Naive": "paper Fig. 21: direct full-res prediction gives noise; ours is clean",
        "K08_Patch": "noisy latent → one 2×2 patch → A_eᵀ → one ordinary-size token",
        "K09_Flow": "short sequence → DiT (LoRA) → size head → Eq. 9 (colour-coded) → full-res velocity → VAE",
        "K10_RoPE": "RoPE at the patch centre; size embedding s(e) bars → zero-init MLP",
        "K11_Sources": "paper Fig. 4: box / mask / texture / depth layouts → images, compression · speed-up",
        "K12_Budget": "paper Fig. 10: deer at B = 1,024 … 4,096 tokens",
        "K13_MoveBox": "paper Fig. 19: move the fine box, subject follows; Tab. 4 44.9% → 66.8%",
        "K14_Driving": "site driving video: semantic layout + LoT-Wan2.1 video, 1.97× fewer tokens",
        "K15_Speed": "speed-up vs compression curves (Tab. 1, 2); 2.04× images, 3.53× video",
        "K16_Quality": "HPSv3 bars at the same budget: full / ToMe-SD / DDiT / Foveated / ours",
        "K17_ImageCompare": "paper Fig. 7a row with speed-ups and failure notes",
        "K18_VideoCompare": "site video comparison (paper Fig. 7b example), 4 playing videos",
    }
    where = {clip: k for k, (clip, _) in T.items()}
    for c, d in desc.items():
        out.append(f"| `LoT_{c}.mp4` | {d} | {where.get(c, '—')} |")
    open("threads.md", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
