# Mind Graph
Last updated: 2026-09-27

### Light Field Displays for Accommodation (VAC)
- **Description**: Displays whose view spacing is smaller than the pupil, so the eye can refocus and the vergence-accommodation conflict disappears
- **Related topics**: [Coded Aperture Imaging and Display], [Digital Refocusing], [Time Multiplexing]
- **Key papers**:
  - [chao2023time] CTDM-LFD (IEEE TIP 2023) — learned coded apertures + encoded views, time-multiplexed
  - [lanman2013near] Near-eye light field displays (TOG 2013) — microlens (space-division) design
- **Other relevant papers**:
  - [wang2021time] Time-multiplexed integral imaging LFDs (SPIE 2021)
  - [zabels2019ar] VAC survey (Appl. Sci. 2019)

### Coded Aperture Imaging and Display
- **Description**: Patterned apertures shape the point spread function (defocus blur = scaled aperture) for capture or projection
- **Related topics**: [Light Field Displays for Accommodation (VAC)], [End-to-end Optics Optimization]
- **Key papers**:
  - [grosse2010coded] Coded aperture projection (TOG 2010) — coded apertures in a display (projector)
  - [liang2008programmable] Programmable aperture photography (TOG 2008) — multiplexed light field capture, same NTU lab
- **Other relevant papers**:
  - [inagaki2018learning] Learned coded-aperture light field capture (ECCV 2018)
  - [maruyama2019display] Coded-aperture camera to tensor display via CNN (ICIP 2019)
  - [chao2023time] CTDM-LFD — learned coded apertures on the display side

### Digital Refocusing
- **Description**: Refocused images from light fields (shift-and-add, Fourier slice); sparse views alias in defocused regions
- **Related topics**: [Light Field Displays for Accommodation (VAC)]
- **Key papers**:
  - [levoy1996light] Light field rendering (SIGGRAPH 1996)
  - [ng2005fourier] Fourier slice photography (SIGGRAPH 2005)
- **Other relevant papers**:
  - [kalantari2016learning] Learning-based view synthesis (SIGGRAPH Asia 2016) — SIGGRAPH16 dataset

### End-to-end Optics Optimization
- **Description**: Differentiable image-formation models let optics (apertures) and processing (CNNs) be learned jointly
- **Related topics**: [Coded Aperture Imaging and Display]
- **Key papers**:
  - [chao2023time] CTDM-LFD — Fourier-optics PSFs of learned apertures + light field encoding CNN, focal-stack loss
- **Other relevant papers**:
  - [inagaki2018learning] learned coded aperture + CNN for capture
