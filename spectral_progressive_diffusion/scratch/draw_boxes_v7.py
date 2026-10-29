# Draw the paper's inset boxes (make_figure_grid.py selection, colors, width) on the clean source images.
# Writes data/<name>_box.png and keeps data/<name>.png clean.
import importlib.util, os
from PIL import Image, ImageDraw
spec = importlib.util.spec_from_file_location('mfg', '/home/brianchc/wavelet-diffusion/figure_scripts/make_figure_grid.py')
mfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(mfg)
D = '/home/brianchc/project_animations/spectral_progressive_diffusion/data/'
def draw(name, boxes, ds, bw):
    im = Image.open(D + name + '.png').convert('RGB'); d = ImageDraw.Draw(im)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # paper: rectangle [x/ds ...] with width bw on the ds-downsampled cell; same footprint at native res
        d.rectangle([x1, y1, x2 + ds - 1, y2 + ds - 1], outline=mfg.INSET_COLORS[i % 2], width=bw * ds)
    im.save(D + name + '_box.png')
    return [tuple(b) for b in boxes]
log = []
lat = ['flux1x', 'ralu158', 'ours166', 'flux484', 'ralu498', 'ours577']
for subj in ['buddha', 'canyon']:      # main_qual.sh: --downsample 4, padding 0.01, consensus over the 6 columns
    ims = [Image.open(f'{D}{subj}_{k}.png').convert('RGB') for k in lat]
    cell = 1024 // 4; bw = max(2, max(1, int(0.01 * cell)) // 2)
    boxes = mfg.sharpness_based_inset_boxes_consensus(ims, cell // 2)
    for k in lat: log.append((f'{subj}_{k}', draw(f'{subj}_{k}', boxes, 4, bw)))
for subj in ['city', 'aurora']:        # pixelgen panel: --downsample 2, boxes picked per image
    for k in ['pixelgen1x', 'ours16']:
        im = Image.open(f'{D}{subj}_{k}.png').convert('RGB')
        cell = 512 // 2; bw = max(2, max(1, int(0.01 * cell)) // 2)
        log.append((f'{subj}_{k}', draw(f'{subj}_{k}', mfg.sharpness_based_inset_boxes(im, cell // 2), 2, bw)))
pt = ['full', 'd0001', 'd001', 'd01', 'd05', 'd1']
for subj in ['crow', 'phone', 'camera', 'bus']:   # ablation_qual.sh: --downsample 2, consensus over the 6 columns
    ims = [Image.open(f'{D}pt_{subj}_{k}.png').convert('RGB') for k in pt]
    cell = 1024 // 2; bw = max(2, max(1, int(0.01 * cell)) // 2)
    boxes = mfg.sharpness_based_inset_boxes_consensus(ims, cell // 2)
    for k in pt: log.append((f'pt_{subj}_{k}', draw(f'pt_{subj}_{k}', boxes, 2, bw)))
for n, b in log: print(n, b)
