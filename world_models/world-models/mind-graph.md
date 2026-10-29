# Mind Graph
Last updated: 2026-09-26

### Controllable video world models
- **Description**: Video generators you can steer — camera, actions, time, memory — so they behave like explorable worlds.
- **Related topics**: [Camera control], [Interactive / embodied generation], [Space-time control], [World-model memory]
- **Key papers**:
  - [he2025cameractrl2] CameraCtrl II (ICCV 2025) — camera-steered open-ended exploration of dynamic scenes
  - [wu2025spmem] Spatial Memory (NeurIPS 2025) — persistence when revisiting places

### Camera control
- **Description**: Conditioning video diffusion on explicit camera trajectories.
- **Key papers**:
  - [he2025cameractrl] CameraCtrl (ICLR 2025) — Plücker-embedding camera encoder, plug-in module
  - [he2025cameractrl2] CameraCtrl II (ICCV 2025) — dynamic scenes, metric cameras, clip chaining

### Interactive / embodied generation
- **Description**: Worlds that respond to the user's body (head + hands) in real time.
- **Key papers**:
  - [xie2026generatedreality] Generated Reality (CVPR 2026) — hand/head-pose conditioned, real-time XR

### Space-time control
- **Description**: Decoupling scene time from camera motion (freeze / slow time while moving).
- **Key papers**:
  - [wang2026bullettime] BulletTime (CVPR 2026) — Time-RoPE + 4D-RoPE + AdaLN controls

### World-model memory
- **Description**: Keeping generated worlds consistent over long horizons and revisits.
- **Key papers**:
  - [wu2025spmem] Spatial Memory (NeurIPS 2025) — point-cloud spatial memory + episodic keyframes
