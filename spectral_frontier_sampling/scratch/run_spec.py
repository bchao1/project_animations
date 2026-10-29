# Wrapper from findings/2026-10-15_spectrogram-motivation.md: write the npz to a folder this user can write.
import sys
from pathlib import Path
sys.path.insert(0, "/home/brianchc/prompt-adaptive-speed-cvpr-2027")
import experiments.fractional_expansion.spectrogram as S
S.OUT = Path(__file__).resolve().parent / "spec" / "online"
S.main()
