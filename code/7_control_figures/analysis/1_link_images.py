'''
Lays out the images of the control experiment so that the scripts of steps 1 and 3 can be used on them unchanged.

The scripts of steps 1 and 3 read the tiles of an acquisition in {EXP_PATH}Live/{CONDITION}/non_merged/, named
*_t{TT}_s{SS}_z{ZZ}_ch00.tif. This script creates the folder control/analysis/ (the EXP_PATH of run_control.sh) with
one CONDITION per acquisition, whose non_merged/ folder holds links to the original images, renamed to this
convention (only the plane used for the segmentation, given the index z00). The original images are not copied
and not modified.

The cellpose masks already computed for these images are linked too, in non_merged/finetune_z2/ with the
names written by cellpose (*_cp_masks.png): with them, the segmentation of run_control.sh (SEGMENTATION) can be skipped.

    CONDITION                     acquisition                        original folder
    step0_{density}               colour barcode before the movie    BarcodeBeforeTimelapse/canal_{density}/   1 frame
    step1_{density}               movie (21 frames)                  Live/canal_{density}/                      plane z02
    afterUV_{density}             colour barcode after fixation      BarcodeAfterUV/canal_{density}/            1 frame
    step3_{density}               long movie after UV                LongMovieAfterUV/canal_{density}/          plane z02, 1 tile

    Cellpose masks already computed: in the original folder (step0, afterUV), in its subfolder
    non_merged/finetune_z2/ (step1) or z_slice/ (step3)

density: highdensity or lowdensity. The long movie exists for the high density only.

Run: python3 -m 7_control_figures.analysis.1_link_images   (from code/; the folder of the experiment is
control.exp_path of config.yaml)
'''

import contextlib
import io
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # code/, for tools/
with contextlib.redirect_stdout(io.StringIO()):
    from tools.read_yaml import ROOTPATH, config

CONTROL = pathlib.Path(ROOTPATH + config['control']['exp_path'])
ANALYSIS = CONTROL / 'analysis' / 'Live'
Z_PLANE = 'z02'   # plane of the movies used for the segmentation

# (condition, original folder, regular expression of the original names of the brightfield images,
#  subfolder of the original folder holding the cellpose masks)
ACQUISITIONS = []
for density in ['highdensity', 'lowdensity']:
    ACQUISITIONS += [
        (f'step0_{density}', f'BarcodeBeforeTimelapse/canal_{density}', r'^(?P<prefix>.+)_s(?P<s>\d+)_ch00', '.'),
        (f'step1_{density}', f'Live/canal_{density}', r'^(?P<prefix>.+)_t(?P<t>\d+)_s(?P<s>\d+)_' + Z_PLANE + r'_ch00',
         'non_merged/finetune_z2'),
        (f'afterUV_{density}', f'BarcodeAfterUV/canal_{density}', r'^(?P<prefix>.+)_s(?P<s>\d+)_ch00', '.'),
        (f'step3_{density}', f'LongMovieAfterUV/canal_{density}', r'^(?P<prefix>.+)_t(?P<t>\d+)_' + Z_PLANE + r'_ch00',
         'z_slice'),
    ]


def link_files(source, target, pattern, suffix):
    '''Links in target the files of source whose name is pattern + suffix, renamed *_t{TT}_s{SS}_z00_ch00 + suffix'''
    if not source.is_dir():
        return 0
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(source.iterdir()):
        match = re.match(pattern + re.escape(suffix) + '$', path.name)
        if path.name.startswith('.') or match is None:
            continue
        t = int(match.groupdict().get('t') or 0)
        s = int(match.groupdict().get('s') or 0)
        link = target / f"{match.group('prefix')}_t{t:02d}_s{s:02d}_z00_ch00{suffix}"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(path, target), link)
        n += 1
    return n


for condition, folder, pattern, mask_folder in ACQUISITIONS:
    source = CONTROL / folder
    if not source.is_dir():
        print(f'{condition}: no folder {source}, skipped')
        continue
    target = ANALYSIS / condition / 'non_merged'
    n = link_files(source, target, pattern, '.tif')
    n_masks = link_files(source / mask_folder, target / 'finetune_z2', pattern, '_cp_masks.png')
    print(f'{condition}: {n} images and {n_masks} cellpose masks linked in {target}')
