Slide 2 (media2.mp3): Gaussian wave splatting is a novel computer-generated holography algorithm that seamlessly transforms Gaussian scene representations derived from real-world scenes using Gaussian splatting into wave-based holograms.

Slide 3 (media3.mp3): During this transformation, each Gaussian primitive is converted into a wave front and propagated to the hologram plane.

Slide 10 (media4.mp3): However, Gaussian waves splatting initializes wave fronts with a smooth phase distribution, which concentrates most of the wavefront energy near the center of the angular spectrum of the hologram plane. This results in a limited spatial bandwidth that leads to a small eye box and unnatural defocus blur.

Slide 15 (media8.mp3): a naive extension, which is simply modulating each primitive wave front with random phase and then running Gaussian wave splatting as is provably fails, reconstructing incorrect appearance.

Slide 16 (media11.mp3): Moreover, as its name implies, Gaussian wave splatting is restricted to Gaussian primitives. How to extend this paradigm to work with other emerging translucent primitives used for novel view synthesis, such as soft edge triangles, remains unclear.

Slide 17 (media12.mp3): To address these issues, we introduce a novel alpha blending and wave compositing framework for random phase wave fronts, which substantially expands the IBOX and reconstructs natural defocus blur outperforming Gaussian wave splatting. Furthermore, OutMethod generalizes beyond Gaussians to other translucent primitives, including soft edge triangles. We call this approach random phase wave splatting.

Slide 26 (media13.mp3): In random phase wavesplating, each primitive, not constrained to be Gaussians, is initialized with a spatially varying random phase. This random phase scatters the wavefront to a large spatial region and distributes wavefront energy uniformly across the angular spectrum, increasing the spatial bandwidth of the hologram.

Slide 33 (media14.mp3): We introduce a novel intensity-based alpha blending rule for random phase primitives with continuous opacity, and show that time multiplexing is the exact mathematical mechanism enabling physically correct blending for accurate appearance reconstruction, not merely a speckle reduction heuristic.

Slide 37 (media15.mp3): Furthermore, to enable accurate occlusion modeling under high angular spread, we devise a novel back-to-front wave compositing strategy. At each step, the current primitive is used to mask the wave fronts of the primitives behind it, and its own wave front is then added. This recursive process allows us to reconstruct the final hologram with physically correct occlusion effects.

Slide 43 (media16.mp3): and its own wave front is then added. This recursive process allows us to construct the final hologram with physically correct occlusion effects.

Slide 52 (media19.mp3): compared to Gaussian wave splatting. Our method synthesizes natural defocus blur and sharp and focus image content in reconstructed 3D focal stacks.

Slide 54 (media35.mp3): Our method also generates holograms with a uniform spectrum, demonstrating large parallax effects across the expanded iBox.

Slide 55 (media37.mp3): While naively applying random phase in Gaussian wave, splatting provably fails, our method accurately reconstruct appearance using our novel alpha blending and wave compositing strategy.

Slide 56 (media38.mp3): Furthermore, our method naturally works with other primitives aside from Gaussians for novel view synthesis, such as soft edge triangles with spatially varying opacity.

Slide 57 (media54.mp3): Our method also generates holograms which are much more uniform spectrum compared to smooth phase Gaussian wave spliting holograms, demonstrating large parallax effects across the expanded I-box.

Slide 58 (media59.mp3): Our random phase Gaussian's formulation also directly benefits from time multiplexing. The focal stack reconstruction quality drastically improves as the number of time multiplex frames increases.

Slide 59 (media61.mp3): We experimentally capture 3D focal stacks and light field parallax on a holographic display prototype, and the results will match our simulations.

Slide 64 (media75.mp3): Our random phase wave, splatting algorithm, synthesizes photorealistic holograms from state-of-the-art primitive-based scene representations, accurately reconstructing natural defocus blur and large parallax, unlocking the full potential of next-generation holographic displays for perceptually realistic and comfortable VR experiences.

Slide 71 (media94.mp3): In recent years, holographic near-eye displays have unlocked breakthroughs in the design of spatial computing systems. The use of holograms has enabled novel ultra-compact augmented and virtual reality glass designs. Additionally, these displays have been used to produce unprecedented 3D image quality with virtual content that accurately reproduces the focus cues that would be seen by the real world. At the core of emerging holographic displays is computer-generated holography, or CGH. CGH refers to algorithms that convert 3D scenes into 2D interference patterns that can be displayed on a spatial light modulator. Over the years, various CGH methods have been developed primarily for virtual content, including point clouds, polygon meshes, image layers, and light fields. Recently, Gaussian wave splatting has been proposed as a method to reproduce realistic holograms from real-world 3D scenes using the state-of-the-art neural rendering technique, Gaussian Splatting. Each Gaussian primitive is converted into a wave front and propagated to the hologram plane. However, Gaussian wave splatting initializes with a smooth-phase distribution, which results in similar wave front phases at the hologram plane. This limits the spatial bandwidth of a hologram, causing a small eye-box and unnatural depth blur. To address these issues, we propose random-phase Gaussian wave splatting. In our method, each Gaussian is initialized with a spatially-varying random phase. This random phase scatters the wave front across the angular spectrum, significantly increasing the spatial bandwidth of the hologram. To further improve the visual quality, we mitigate constructive and destructive interference between overlapping wave fronts by adopting a time-multiplex approach, by rapidly displaying multiple holograms with different random-phase patterns and relying on temporal integration by the human eye. We achieve an intensity-average reconstruction that suppresses interference artifacts and speckle. Furthermore, to handle occlusion correctly under high-angular spread, we adopt a back-to-front compositing strategy. At each step, the current Gaussian is used to mask the wavefront of those behind it, and its own wavefront is then added. This recursive process allows us to construct the final hologram with physically correct occlusion effects.

Slide 106 (media16.mp3): and its own wave front is then added. This recursive process allows us to construct the final hologram with physically correct occlusion effects.

