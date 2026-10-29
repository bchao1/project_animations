import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animation_lot_5min_v2 import *


class Still(Scene):
    def construct(self):
        self.add(LoTField(LEAVES, 4.6, (-3.4, 0.35), shade=True).group())
        self.add(LoTField(state_max(1), 4.6, (3.4, 0.35), shade=True).group())
