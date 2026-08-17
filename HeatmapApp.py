# ==========================================
# Revision History
# Rev 1.0.0 : Initial version.
# Rev 1.0.1 : Modified pd.read_csv logic to support headerless CSV files.
# Rev 1.0.2 : Added Panel Masking feature (Threshold based).
# Rev 1.0.3 : Added 'Auto-Detect Shape' to Panel Masking using OpenCV contour/hierarchy 
#             analysis to automatically exclude round corners and camera holes while 
#             preserving intentional dead pixels inside the panel.
# Rev 1.0.4 : Increased precision of mask threshold input to 5 decimal places.
# Rev 1.0.5 : Changed Threshold masking to use Percentage (%) of the Maximum value 
#             instead of an absolute value for better generalization across datasets.
# Rev 1.0.6 : Added a login page to restrict access. ID must end with '@yitoa.co.jp' and Password must match ID.
# Rev 1.0.7 : Fixed RuntimeWarning 'Mean of empty slice' when measuring uniformity over completely masked (NaN) areas.
# ==========================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.ndimage as ndimage
import cv2  # Added for contour detection
import os
import warnings # Added to suppress expected math warnings

# Page configuration
st.set_page_config(page_title="Demura Heatmap Analyzer", layout="wide")

# --- Login System ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    # ログイン画面上部にもロゴを表示（存在する場合）
    if os.path.exists("yitoa.png"):
        st.image("yitoa.png", width=250)
        
    st.title("🔒 System Login")
    st.write("Please log in to access the Demura Heatmap Analyzer.")
    
    with st.form("login_form"):
        user_id = st.text_input("ID")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            # 他のAppと共通のログインロジック: @yitoa.co.jpドメイン ＆ IDとパスワードが一致
            if user_id.endswith("@yitoa.co.jp") and user_id == password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid ID or Password.")
                
    # 未ログイン時はここで処理を停止させ、メインアプリを描画しない
    st.stop()

# --- Sidebar Logo and Copyright ---
if os.path.exists("yitoa.png"):
    st.sidebar.image("yitoa.png", width="stretch")
else:
    st.sidebar.markdown("<div style='text-align: center; color: red; font-size: 12px;'>[yitoa.png not found]</div>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 12px; color: #666; margin-bottom: 20px; line-height: 1.4;">
        Copyright(c) YITOA Technology.<br>
        All rights reserved.<br>
        Rev 1.0.7
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar for Settings ---
st.sidebar.header("⚙️ Display Settings")
st.sidebar.write("Adjust the heatmap size dynamically.")
fig_width = st.sidebar.slider("Heatmap Width", min_value=1.0, max_value=10.0, value=2.7, step=0.1)
fig_height = st.sidebar.slider("Heatmap Height", min_value=1.0, max_value=15.0, value=3.4, step=0.1)
st.sidebar.markdown("---")

# --- Sidebar for Panel Masking (UPDATED to %) ---
st.sidebar.header("🔲 Panel Masking")
st.sidebar.write("Exclude dead areas (corners, camera holes) from stats.")
enable_mask = st.sidebar.checkbox("Enable Masking", value=True)

mask_mode = "Threshold (%)"
mask_threshold_pct = 0.0

if enable_mask:
    mask_mode = st.sidebar.radio("Masking Mode", ["Auto-Detect Shape (OpenCV)", "Threshold (%)"])
    if mask_mode == "Threshold (%)":
        mask_threshold_pct = st.sidebar.number_input("Mask Threshold (% of Max Value)", min_value=0.0, max_value=100.0, value=1.0, step=0.1, format="%.2f")
        st.sidebar.caption("Pixels at or below this percentage of the maximum value will be ignored.")
    else:
        st.sidebar.caption("Automatically detects panel outlines and camera holes based on geometry, keeping random internal dead pixels intact.")

st.sidebar.markdown("---")

# --- Sidebar for Spatial Filter ---
st.sidebar.header("🎛️ Spatial Filter")
st.sidebar.write("Apply filters to extract specific Mura frequencies.")
filter_type = st.sidebar.selectbox("Filter Type", ["None", "Gaussian (Smooth)", "Median (Noise Reduction)"])

filter_param = None
if filter_type == "Gaussian (Smooth)":
    filter_param = st.sidebar.slider("Gaussian Sigma (Strength)", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
elif filter_type == "Median (Noise Reduction)":
    filter_param = st.sidebar.slider("Median Size (Window)", min_value=3, max_value=21, value=3, step=2)

st.sidebar.markdown("---")

# --- Sidebar for Uniformity Measurement ---
st.sidebar.header("📏 Uniformity Measurement")
st.sidebar.write("Calculate Uniformity (Lmin/Lmax*100%) by sampling grid points.")
enable_uniformity = st.sidebar.checkbox("Enable Uniformity", value=False)

grid_x, grid_y, margin, spot_type, spot_param, show_points = 9, 15, 0, "1 Pixel (Point)", 0, False

if enable_uniformity:
    grid_x = st.sidebar.number_input("Points X (Columns)", min_value=2, max_value=100, value=9)
    grid_y = st.sidebar.number_input("Points Y (Rows)", min_value=2, max_value=100, value=15)
    margin = st.sidebar.number_input("Edge Margin (pixels to exclude)", min_value=0, max_value=2000, value=0)
    
    spot_type = st.sidebar.selectbox("Spot Size Type", ["1 Pixel (Point)", "Square (N x N)", "Circle (Radius R)", "Probe (Auto Calculate)"])
    if spot_type == "Square (N x N)":
        spot_param = st.sidebar.number_input("Square Size (N)", min_value=3, max_value=101, value=5, step=2)
    elif spot_type == "Circle (Radius R)":
        spot_param = st.sidebar.number_input("Circle Radius (R)", min_value=1, max_value=1000, value=3)
    elif spot_type == "Probe (Auto Calculate)":
        st.sidebar.markdown("**Panel Specifications**")
        panel_inch = st.sidebar.number_input("Panel Size (inch)", min_value=1.0, max_value=100.0, value=6.1, step=0.1)
        res_x = st.sidebar.number_input("Resolution X", min_value=100, max_value=10000, value=1080)
        res_y = st.sidebar.number_input("Resolution Y", min_value=100, max_value=10000, value=2532)
        probe_phi = st.sidebar.selectbox("Probe Diameter (φ mm)", [2.1, 4.0, 10.0, 27.0], index=2)
        
        diag_pixels = np.sqrt(res_x**2 + res_y**2)
        ppi = diag_pixels / panel_inch
        pixel_pitch_mm = 25.4 / ppi
        r_pixels = int(round((probe_phi / 2) / pixel_pitch_mm))
        spot_param = max(1, r_pixels)
        st.sidebar.caption(f"**Evaluated Radius (R): {spot_param} px**")
        
    show_points = st.sidebar.checkbox("Show Measurement Points on Heatmap", value=False)

def apply_spatial_filter(data, f_type, f_param):
    if f_type == "Gaussian (Smooth)":
        return ndimage.gaussian_filter(data, sigma=f_param)
    elif f_type == "Median (Noise Reduction)":
        return ndimage.median_filter(data, size=f_param)
    return data

def create_shape_mask(data):
    """OpenCVを使用して、パネルのコーナーとカメラホールの幾何学形状を認識しマスクを作成する"""
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    
    if data_max == data_min or np.isnan(data_max):
        return np.ones_like(data, dtype=bool)
        
    img_norm = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours_ext, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_ext:
        return thresh > 0
        
    main_contour = max(contours_ext, key=cv2.contourArea)
    base_mask = np.zeros_like(thresh, dtype=np.uint8)
    
    cv2.drawContours(base_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
    
    contours_all, hierarchy = cv2.findContours(thresh_cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = base_mask.copy()
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        main_idx = -1
        max_area = -1
        for i, cnt in enumerate(contours_all):
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                main_idx = i
                
        for i, h in enumerate(hierarchy):
            if h[3] == main_idx:
                hole_area = cv2.contourArea(contours_all[i])
                if max_area * 0.0005 < hole_area < max_area * 0.15:
                    cv2.drawContours(final_mask, contours_all, i, 0, thickness=cv2.FILLED)

    return final_mask > 0

def calculate_uniformity(data, gx, gy, m, s_type, s_param):
    h, w = data.shape
    if h <= m * 2 or w <= m * 2:
        return np.nan
        
    y_coords = np.linspace(m, h - 1 - m, gy, dtype=int)
    x_coords = np.linspace(m, w - 1 - m, gx, dtype=int)
    sampled_means = []
    
    # 測定スポットが完全にNaN(マスク領域)に落ちた場合の警告を抑制する
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for cy in y_coords:
            for cx in x_coords:
                if s_type == "1 Pixel (Point)":
                    sampled_means.append(data[cy, cx])
                elif s_type == "Square (N x N)":
                    half = s_param // 2
                    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
                    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
                    sampled_means.append(np.nanmean(data[y0:y1, x0:x1]))
                elif s_type in ["Circle (Radius R)", "Probe (Auto Calculate)"]:
                    r = s_param
                    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
                    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
                    Y, X = np.ogrid[y0-cy:y1-cy, x0-cx:x1-cx]
                    mask = X**2 + Y**2 <= r**2
                    region = data[y0:y1, x0:x1]
                    sampled_means.append(np.nanmean(region[mask]))
    
    # NaNになっているスポットを除外してUniformityを計算
    valid_means = [v for v in sampled_means if not np.isnan(v)]
    if not valid_means:
        return np.nan
        
    lmin = np.nanmin(valid_means)
    lmax = np.nanmax(valid_means)
    
    if lmax == 0 or np.isnan(lmax):
        return np.nan
        
    return (lmin / lmax) * 100.0

def draw_measurement_points(ax, h, w, gx, gy, m, s_type, s_param):
    if h <= m * 2 or w <= m * 2:
        return
    y_coords = np.linspace(m, h - 1 - m, gy, dtype=int)
    x_coords = np.linspace(m, w - 1 - m, gx, dtype=int)
    for cy in y_coords:
        for cx in x_coords:
            if s_type == "1 Pixel (Point)":
                ax.plot(cx, cy, marker='+', color='red', markersize=2, markeredgewidth=0.5)
            elif s_type == "Square (N x N)":
                half = s_param / 2.0
                ax.add_patch(patches.Rectangle((cx - half, cy - half), s_param, s_param, linewidth=0.5, edgecolor='red', facecolor='none'))
            elif s_type in ["Circle (Radius R)", "Probe (Auto Calculate)"]:
                ax.add_patch(patches.Circle((cx, cy), s_param, linewidth=0.5, edgecolor='red', facecolor='none'))

# --- Main App ---
st.title("📱 Demura Heatmap Analyzer")
st.write("Upload CSV files to visualize display uniformity.")

uploaded_files = st.file_uploader("Upload Demura CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    tab_normal, tab_diff = st.tabs(["Normal Heatmaps", "Difference Heatmap (A - B)"])
    
    with tab_normal:
        MAX_COLS_PER_ROW = 3
        for i in range(0, len(uploaded_files), MAX_COLS_PER_ROW):
            chunk = uploaded_files[i:i + MAX_COLS_PER_ROW]
            cols = st.columns(MAX_COLS_PER_ROW)
            for idx, file in enumerate(chunk):
                with cols[idx]:
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, header=None)
                        numeric_df = df.select_dtypes(include=[np.number])
                        if numeric_df.empty:
                            file.seek(0)
                            df = pd.read_csv(file)
                            numeric_df = df.select_dtypes(include=[np.number])
                        if numeric_df.empty:
                            continue

                        data_values = numeric_df.values.astype(float)
                        
                        # Generate Mask
                        valid_mask = np.ones_like(data_values, dtype=bool)
                        if enable_mask:
                            if mask_mode == "Threshold (%)":
                                d_max = np.nanmax(data_values)
                                threshold_val = d_max * (mask_threshold_pct / 100.0)
                                valid_mask = data_values > threshold_val
                            else:
                                valid_mask = create_shape_mask(data_values)

                        if filter_type != "None":
                            data_values = apply_spatial_filter(data_values, filter_type, filter_param)

                        if enable_mask:
                            data_values = np.where(valid_mask, data_values, np.nan)

                        if np.all(np.isnan(data_values)):
                            st.warning(f"All data masked in {file.name}.")
                            continue

                        actual_min = float(np.nanmin(data_values))
                        actual_max = float(np.nanmax(data_values))
                        actual_ave = float(np.nanmean(data_values))
                        actual_std = float(np.nanstd(data_values))

                        st.markdown(f"**{file.name}**")
                        scale_mode = st.radio("Color Scale Mode", ["Min / Max", "Sigma (Ave ± Nσ)"], key=f"mode_{file.name}_{i}_{idx}", horizontal=True)
                        if scale_mode == "Min / Max":
                            sc1, sc2 = st.columns(2)
                            user_vmin = sc1.number_input("Display Min", value=actual_min, key=f"vmin_{file.name}_{i}_{idx}")
                            user_vmax = sc2.number_input("Display Max", value=actual_max, key=f"vmax_{file.name}_{i}_{idx}")
                        else:
                            sigma_n = st.number_input("Sigma Multiplier (N)", value=3.0, step=0.5, key=f"sigma_{file.name}_{i}_{idx}")
                            user_vmin = actual_ave - (sigma_n * actual_std)
                            user_vmax = actual_ave + (sigma_n * actual_std)

                        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
                        im = ax.imshow(data_values, cmap='viridis', aspect='auto', interpolation='bilinear', vmin=user_vmin, vmax=user_vmax)
                        
                        if enable_uniformity and show_points:
                            draw_measurement_points(ax, data_values.shape[0], data_values.shape[1], grid_x, grid_y, margin, spot_type, spot_param)
                        
                        cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.04)
                        cbar.ax.tick_params(labelsize=4)
                        ax.set_title(file.name + (f"\n[{filter_type.split(' ')[0]}]" if filter_type != "None" else ""), fontsize=5, pad=6)
                        ax.tick_params(axis='both', which='major', labelsize=4)
                        
                        stats_text = f"Min: {actual_min:.4g}  Max: {actual_max:.4g}\nAve: {actual_ave:.4g}  σ: {actual_std:.4g}"
                        if enable_uniformity:
                            uni_val = calculate_uniformity(data_values, grid_x, grid_y, margin, spot_type, spot_param)
                            stats_text += f"\nUniformity: {uni_val:.2f}%" if not np.isnan(uni_val) else "\nUniformity: N/A"
                        
                        ax.text(0.5, -0.16, stats_text, transform=ax.transAxes, ha='center', va='top', fontsize=5)
                        fig.subplots_adjust(left=0.15, right=0.75, top=0.85, bottom=0.25)
                        st.pyplot(fig, bbox_inches='tight', width="content")
                        plt.close(fig)
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")

    with tab_diff:
        st.subheader("Calculate Difference (CSV A - CSV B)")
        if len(uploaded_files) >= 2:
            file_indices = list(range(len(uploaded_files)))
            format_label = lambda i: f"{uploaded_files[i].name} (File {i+1})"
            col_a, col_b = st.columns(2)
            idx_a = col_a.selectbox("Select CSV A (Base)", file_indices, index=0, format_func=format_label)
            idx_b = col_b.selectbox("Select CSV B (Target to subtract)", file_indices, index=1, format_func=format_label)
                
            if idx_a is not None and idx_b is not None:
                file_a, file_b = uploaded_files[idx_a], uploaded_files[idx_b]
                try:
                    file_a.seek(0); df_a = pd.read_csv(file_a, header=None)
                    num_a = df_a.select_dtypes(include=[np.number])
                    if num_a.empty: file_a.seek(0); num_a = pd.read_csv(file_a).select_dtypes(include=[np.number])
                    
                    file_b.seek(0); df_b = pd.read_csv(file_b, header=None)
                    num_b = df_b.select_dtypes(include=[np.number])
                    if num_b.empty: file_b.seek(0); num_b = pd.read_csv(file_b).select_dtypes(include=[np.number])
                    
                    if num_a.empty or num_b.empty or num_a.shape != num_b.shape:
                        st.error("Data missing or shape mismatch.")
                    else:
                        val_a, val_b = num_a.values.astype(float), num_b.values.astype(float)
                        diff_values = val_a - val_b
                        
                        valid_mask_diff = np.ones_like(diff_values, dtype=bool)
                        if enable_mask:
                            if mask_mode == "Threshold (%)":
                                val_a_max = np.nanmax(val_a)
                                threshold_val = val_a_max * (mask_threshold_pct / 100.0)
                                valid_mask_diff = val_a > threshold_val
                            else:
                                valid_mask_diff = create_shape_mask(val_a)
                        
                        if filter_type != "None": diff_values = apply_spatial_filter(diff_values, filter_type, filter_param)
                        if enable_mask: diff_values = np.where(valid_mask_diff, diff_values, np.nan)
                            
                        if np.all(np.isnan(diff_values)):
                            st.warning("All data masked.")
                        else:
                            actual_min, actual_max, actual_ave, actual_std = float(np.nanmin(diff_values)), float(np.nanmax(diff_values)), float(np.nanmean(diff_values)), float(np.nanstd(diff_values))
                            st.markdown(f"**Difference: {format_label(idx_a)} - {format_label(idx_b)}**")
                            
                            scale_mode_diff = st.radio("Scale Mode", ["Min / Max", "Sigma (Ave ± Nσ)"], key="mode_diff", horizontal=True)
                            if scale_mode_diff == "Min / Max":
                                sc1, sc2 = st.columns(2)
                                user_vmin_diff = sc1.number_input("Display Min", value=actual_min, key="vmin_diff")
                                user_vmax_diff = sc2.number_input("Display Max", value=actual_max, key="vmax_diff")
                            else:
                                sigma_n_diff = st.number_input("Sigma (N)", value=3.0, step=0.5, key="sigma_diff")
                                user_vmin_diff, user_vmax_diff = actual_ave - (sigma_n_diff * actual_std), actual_ave + (sigma_n_diff * actual_std)
                            
                            _, center_col, _ = st.columns([1, 1, 1])
                            with center_col:
                                fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
                                im = ax.imshow(diff_values, cmap='viridis', aspect='auto', interpolation='bilinear', vmin=user_vmin_diff, vmax=user_vmax_diff)
                                if enable_uniformity and show_points: draw_measurement_points(ax, diff_values.shape[0], diff_values.shape[1], grid_x, grid_y, margin, spot_type, spot_param)
                                cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.04)
                                cbar.ax.tick_params(labelsize=4)
                                ax.set_title(f"Diff: {format_label(idx_a)} - {format_label(idx_b)}" + (f"\n[{filter_type.split(' ')[0]}]" if filter_type != "None" else ""), fontsize=5, pad=6)
                                ax.tick_params(axis='both', which='major', labelsize=4)
                                
                                stats_text = f"Min: {actual_min:.4g}  Max: {actual_max:.4g}\nAve: {actual_ave:.4g}  σ: {actual_std:.4g}"
                                if enable_uniformity:
                                    uni_val_diff = calculate_uniformity(diff_values, grid_x, grid_y, margin, spot_type, spot_param)
                                    stats_text += f"\nUniformity: {uni_val_diff:.2f}%" if not np.isnan(uni_val_diff) else "\nUniformity: N/A"
                                ax.text(0.5, -0.16, stats_text, transform=ax.transAxes, ha='center', va='top', fontsize=5)
                                fig.subplots_adjust(left=0.15, right=0.75, top=0.85, bottom=0.25)
                                st.pyplot(fig, bbox_inches='tight', width="content")
                                plt.close(fig)
                except Exception as e:
                    st.error(f"Error calculating difference: {e}")
        else:
            st.info("Please upload at least 2 CSV files.")
else:
    st.info("Awaiting CSV file uploads. Please drag and drop your files above.")