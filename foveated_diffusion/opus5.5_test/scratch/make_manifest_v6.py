"""Build slides_v6/sections.json. v6 re-rendered only S01, S02, S05, S08, S09 (scenes with replaced images).
The other scenes keep their v5 files. Names, order and descriptions match slides_v5/sections.json."""
import json, os, subprocess
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6 = {"S01_TokenCost", "S02_Vision", "S05_Naive", "S08_Masks", "S09_Results"}
SEC6 = f"{R}/media/videos/animation_foveated_diffusion_v6/1080p60/sections"
out = []
for e in json.load(open(f"{R}/slides_v5/sections.json")):
    e = dict(e)
    if e["scene"] in V6:
        base = os.path.basename(e["file"])[:-4]
        e["file"] = f"{SEC6}/{base}.mp4"
        e["last_frame"] = f"{R}/slides_v6/qa/{base}_last.png"
        assert os.path.exists(e["file"]) and os.path.exists(e["last_frame"]), base
        e["duration_s"] = round(float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of", "csv=p=0", e["file"]]).decode()), 2)
        e["version"] = "v6"
    else:
        e["version"] = "v5"
    out.append(e)
json.dump(out, open(f"{R}/slides_v6/sections.json", "w"), indent=1, ensure_ascii=False)
print(len(out), sum(e["version"] == "v6" for e in out))
