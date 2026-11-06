'''
Step 2.4 (manual, interactive): choose the intensity threshold of the clumps of the smFISH images.

Clumps are bright aggregates that are not single RNA. For a tile, a round and a channel, the app shows the MIP with an
adjustable contrast, the detected spots, the binary image above the threshold (regions smaller than
fish.clumps_min_size removed), and the mask of the cells without the cells that overlap a clump.
The button saves the threshold of the image in ROOTPATH/EXP_PATH/Fish/clumps_detection/CONDITION_thresholds.csv,
read by step 2.5.

Run from the folder code/: streamlit run 2_fish_analysis/4_clumps_detector_streamlit.py
'''


# Import local tools
import sys
sys.path.append('tools/')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import tifffile
from read_yaml import CLUMPS_MIN_SIZE, CONDITION, EXP_PATH, FISH_N_CHANNELS, FISH_N_TILES, ROOTPATH, ROUNDS
from tools_fish import show_tile, remove_small_regions
import streamlit as st


# --- Cached image generation ---
@st.cache_data
def generate_image(condition, s, t, ch):
    img = show_tile(ROOTPATH, EXP_PATH, condition, s, t, ch)
    mip = np.max(img, axis=0)
    t_, s_, ch_ = str(t).zfill((2)), str(s).zfill((2)), str(ch).zfill((2))
    mask = tifffile.imread(ROOTPATH + EXP_PATH + 'Fish/masks/' + condition + f'/mosaic_mask_reassigned_s{s_}.tiff')
    # Spots
    folder_spots = ROOTPATH + EXP_PATH + 'Fish/bigfish/' + condition + '/'
    spots = np.load(folder_spots + 't{}_s{}_ch{}.npy'.format(t_, s_, ch_))
    return mip, mask, spots


def main():
    '''Page of the app'''
    MINSIZE = CLUMPS_MIN_SIZE

    st.title("🧠 Interactive Image Control")
    st.set_page_config(layout='wide')

    # --- Parameters ---
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        s = st.selectbox("Select s value", options=np.arange(FISH_N_TILES))
    with col2:
        t = st.selectbox("Select t value", options=ROUNDS)
    with col3:
        ch = st.selectbox("Select ch value", options=np.arange(FISH_N_CHANNELS))
    contrast_q = st.slider("Adjust contrast (percentage of max value)", 990., 999., 999.99, 0.01)
    threshold_q = st.slider("Binary threshold", 990., 999., 999.99, 0.01)

    # --- Generate base image ---
    image, mask, spots = generate_image(CONDITION, s, t, ch)

    contrast = np.percentile(image, q=contrast_q / 10)
    threshold = np.percentile(image, q=threshold_q / 10)

    # --- Apply contrast ---
    contrast_image = np.clip(image, 0, contrast)
    contrast_image /= contrast_image.max()

    # --- Apply threshold ---
    binary_image = (image > threshold).astype(float)
    binary_image_filtered = remove_small_regions(binary_image, min_size=MINSIZE)

    new_mask = np.copy(mask)
    value_mask = mask[binary_image_filtered > 0]
    remove_id_cells = np.unique(value_mask[value_mask > 0])
    for id_cell in remove_id_cells:
        new_mask[new_mask == id_cell] = 0

    # --- Display images side by side ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.subheader("Original / Contrast Image")
        st.image(contrast_image, caption=f"Contrast = {contrast:.2f}", use_container_width=True)

    with col2:
        st.subheader("Original / Contrast Image with Spots")
        fig, ax = plt.subplots()
        ax.imshow(contrast_image, cmap='gray', origin='upper', vmax=1, vmin=0)
        ax.scatter(spots[:, 2], spots[:, 1], s=1, c='red', marker='o')  # (x, y)
        ax.axis("off")
        st.pyplot(fig)

    with col3:
        st.subheader("Binary Image")
        st.image(binary_image_filtered, caption=f"Threshold = {threshold:.2f}", use_container_width=True)

    with col4:
        st.subheader("Mask")
        st.image((mask > 0).astype(float), use_container_width=True)

    with col5:
        st.subheader("Filtered mask")
        st.image((new_mask > 0).astype(float), use_container_width=True)

    df = pd.DataFrame([{'s': s, 't': t, 'ch': ch, 'thresh': threshold}])
    df = df.set_index(['s', 't', 'ch'])

    # --- Button to save threshold ---
    if st.button("💾 Save Threshold"):
        path_csv = ROOTPATH + EXP_PATH + f'Fish/clumps_detection/{CONDITION}_thresholds.csv'
        if pathlib.Path(path_csv).exists():
            file = pd.read_csv(path_csv, index_col=['s', 't', 'ch'])
        else:
            file = pd.DataFrame()
        # Update the file, and replace when the row with specific index already exists
        file = pd.concat([file, df[~df.index.isin(file.index)]])
        file.update(df)
        file.to_csv(path_csv)
        st.success(f"Threshold value {threshold:.2f} saved to file.")
        st.dataframe(file)


if __name__ == "__main__":
    main()
