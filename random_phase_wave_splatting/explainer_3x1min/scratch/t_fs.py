import sys; sys.path.insert(0, '.')
from animation_rpws_explainer_v1 import *
class T(Base):
    def construct(self):
        kt = ValueTracker(0)
        er = FrameSeq(np.load(gen("c_erpws.npy")), 2.5, tracker=kt); er.mob.move_to([3,0,0])
        self.add(er.mob, kt)
        cnt = always_redraw(lambda: tx(f"{kt.get_value():.1f}", 30).move_to([-3,0,0]))
        self.add(cnt)
        self.run(kt, 30, 1.0)
        self.wait(0.2)
