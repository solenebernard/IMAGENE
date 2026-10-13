'''
Merge the cellpose masks of the tiles into one global mask per time frame.
Adapted from 260918_Oslo/4_merge_masks.py

Tiles are merged sequentially in ascending tile index. When two cells overlap
(same cell seen in two tiles), the priority rule is:
  - one cropped, one not  ->  keep the non-cropped segmentation
  - both non-cropped      ->  keep the first (lower tile index)
  - both cropped          ->  keep the first (lower tile index)
"Cropped" means any pixel of the cell is within BORDER_MARGIN pixels of the tile edge.

Saved in ROOTPATH/EXP_PATH/Live/CONDITION/global_masks/
    - global_mask_t*.png: the global mask of each time frame (ID not tracked yet)
    - global_id_map.csv: for each cell of the global masks, its tile (s) and its ID in the tile (local_id)
'''

# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *


def cropped_labels(tile, margin=3):
    """Labels with at least one pixel within `margin` pixels of any tile border."""
    h, w = tile.shape
    border_zone = np.zeros((h, w), dtype=bool)
    border_zone[:margin, :] = True
    border_zone[-margin:, :] = True
    border_zone[:, :margin] = True
    border_zone[:, -margin:] = True
    result = set(np.unique(tile[border_zone]).tolist())
    result.discard(0)
    return result


def stitch_masks(tiles, tile_coords, shape, border_margin=3):
    """
    Build the global mask for one time frame.
    tiles: (s,x,y) masks of the tiles
    tile_coords: list of (x0, x1, y0, y1), position of each tile in the global mask
    shape: shape of the global mask

    Returns (global_mask, records) where records is a list of
    {global_id, local_id, s} dicts, one per placed cell.
    """
    global_mask = np.zeros(shape, dtype=np.uint32)
    global_is_cropped = {}
    records = {}
    next_id = 1

    for s_idx, tile in enumerate(tiles):
        tile = tile.astype(np.uint32)
        if tile.max() == 0:
            continue

        x0, x1, y0, y1 = tile_coords[s_idx]
        canvas_view = global_mask[y0:y1, x0:x1]

        tile_cropped = cropped_labels(tile, margin=border_margin)

        for lbl in np.unique(tile):
            if lbl == 0:
                continue
            cell_mask = tile == lbl
            current_cropped = lbl in tile_cropped

            canvas_labels = set(np.unique(canvas_view[cell_mask]).tolist())
            canvas_labels.discard(0)

            if not canvas_labels:
                # No overlap: new cell
                canvas_view[cell_mask] = next_id
                global_is_cropped[next_id] = current_cropped
                records[next_id] = {"global_id": next_id, "local_id": int(lbl), "s": s_idx}
                next_id += 1
                continue

            if current_cropped:
                # Current cell is cropped: the global mask always wins, skip
                continue

            # Current cell is NOT cropped: check whether all the overlapping cells are cropped
            if any(not global_is_cropped.get(cl, False) for cl in canvas_labels):
                # At least one is non-cropped: both non-cropped -> first wins
                continue

            # All the overlapping cells are cropped, current is not: current wins.
            # Erase all the cropped versions and place the non-cropped one.
            for cl in canvas_labels:
                global_mask[global_mask == cl] = 0
                del global_is_cropped[cl]
                del records[cl]
            canvas_view[cell_mask] = next_id
            global_is_cropped[next_id] = False
            records[next_id] = {"global_id": next_id, "local_id": int(lbl), "s": s_idx}
            next_id += 1

    return global_mask, list(records.values())


def main(rootpath, exp_path, condition):
    folder_save = f'{rootpath}{exp_path}Live/{condition}/global_masks/'
    os.makedirs(folder_save, exist_ok=True)
    tile_coords = compute_snake_tile_coords(LIVE_TILE_SIZE, LIVE_OVERLAP_PIXELS, LIVE_N_ROWS, LIVE_N_COLS)

    all_records = []
    for t in range(N_FRAMES):
        tiles = load_masks_tile_cellpose(rootpath, exp_path, condition, t, n_tiles=LIVE_N_TILES, z=Z_FINETUNE)
        global_mask, records = stitch_masks(tiles, tile_coords, (LIVE_HEIGHT, LIVE_WIDTH), border_margin=BORDER_MARGIN)
        for r in records:
            r["t"] = t
        all_records.extend(records)
        Image.fromarray(global_mask.astype(np.uint16)).save(folder_save + f'global_mask_t{str(t).zfill(2)}.png')
        print(f't={t}: {len(records)} cells')

    pd.DataFrame(all_records, columns=["t", "global_id", "local_id", "s"]).to_csv(folder_save + 'global_id_map.csv', index=False)


if __name__ == "__main__":
    main(ROOTPATH, EXP_PATH, CONDITION)
