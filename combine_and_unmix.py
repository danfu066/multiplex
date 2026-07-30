# Copyright (c) 2026 Dan Fu@UW
"""
combine_and_unmix.py

Combines individual tile TIFF files based on numerical filename order (with customizable start and end tiles),
saves the combined 3D TIFF stack, and performs clean 2-component NNLS unmixing for Silicone Oil distribution.
"""

import os
import glob
import re
import argparse
import tifffile
import numpy as np
import matplotlib.pyplot as plt

def natural_sort_key(s):
    """Sort filenames numerically (e.g., 00001, 00002, ..., 00014)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def combine_and_unmix(input_dir, start_tile=1, end_tile=None, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    artifact_dir = r"C:\Users\DanFu\.gemini\antigravity\brain\b56b8d50-0f22-4427-912c-e4d35f5a9d20"
    os.makedirs(artifact_dir, exist_ok=True)

    # 1. Discover and filter TIFF files
    all_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")), key=natural_sort_key)
    # Ignore any already-combined output files
    all_files = [f for f in all_files if "combined" not in os.path.basename(f).lower() and "abundance" not in os.path.basename(f).lower()]

    if not all_files:
        raise FileNotFoundError(f"No tile TIFF files found in {input_dir}")

    total_available = len(all_files)
    start_idx = max(1, start_tile) - 1
    end_idx = min(total_available, end_tile) if end_tile is not None else total_available

    selected_files = all_files[start_idx:end_idx]
    print(f"=== Combining {len(selected_files)} Tiles (Tile #{start_idx+1} to #{end_idx} of {total_available}) ===")
    for f in selected_files:
        print("  -", os.path.basename(f))

    # 2. Read and stitch selected tiles along the X axis
    tile_stacks = []
    for fpath in selected_files:
        data = tifffile.imread(fpath) # (35, H, W)
        if data.ndim == 3:
            # Shift unsigned/signed integers to positive scale
            data = data.astype(np.float32)
            tile_stacks.append(data)
        else:
            raise ValueError(f"Unexpected array shape {data.shape} in {fpath}")

    # Concatenate tiles horizontally along X-axis (axis 2)
    combined_stack_35hw = np.concatenate(tile_stacks, axis=2) # (35, H, W_total)
    _, H, W_total = combined_stack_35hw.shape
    print(f"Combined 3D Stack Dimensions: 35 Spectral Bands × {H} Height × {W_total} Total Width")

    # Save combined 3D TIFF stack
    combined_tif_path = os.path.join(output_dir, f"Combined_Stack_Tile_{start_idx+1}_to_{end_idx}.tif")
    tifffile.imwrite(combined_tif_path, combined_stack_35hw.astype(np.float32))
    print(f"[SUCCESS] Saved Combined 3D Stack TIFF to: {combined_tif_path}")

    # 3. Prepare Hypercube for Unmixing (H, W, 35)
    hypercube = np.moveaxis(combined_stack_35hw, 0, -1) # (H, W, 35)
    B = hypercube.shape[-1]

    # Baseline Subtraction
    baseline = np.percentile(hypercube, 1, axis=(0, 1))
    datacube_sub = np.maximum(0.0, hypercube - baseline)

    # 4. Extract Clean Physical Endmembers (Silicone Oil Double Peak vs Background)
    # Peak indicator: double peak at Band 19 & 20 relative to surrounding baseline
    peak_signal = datacube_sub[:, :, 18] + datacube_sub[:, :, 19] - (datacube_sub[:, :, 13] + datacube_sub[:, :, 24])

    bg_mask = peak_signal < np.percentile(peak_signal, 15)
    bg_spectrum = np.median(datacube_sub[bg_mask], axis=0)

    # Smooth background spectrum to ensure ZERO artificial dip at Bands 15-25
    from scipy.ndimage import gaussian_filter1d
    bg_spectrum_clean = bg_spectrum.copy()
    bg_spectrum_clean[15:25] = gaussian_filter1d(bg_spectrum[15:25], sigma=2.0)

    # Silicone Oil ROI
    oil_mask = peak_signal > np.percentile(peak_signal, 99.0)
    oil_spectrum_raw = np.median(datacube_sub[oil_mask], axis=0)

    bg_norm = bg_spectrum_clean / np.max(bg_spectrum_clean)
    oil_spectrum_pure = np.maximum(0.0, oil_spectrum_raw - np.min(oil_spectrum_raw[0:5]) * bg_norm)

    E_bg = bg_spectrum_clean / np.linalg.norm(bg_spectrum_clean)
    E_oil = oil_spectrum_pure / np.linalg.norm(oil_spectrum_pure)
    E_matrix = np.column_stack([E_bg, E_oil]) # (35, 2)

    # 5. Fast Vectorized Non-Negative Least Squares (NNLS) Unmixing
    print("Running Fast Vectorized NNLS Unmixing...")
    X_full = datacube_sub.reshape(-1, B)
    inv_GtG = np.linalg.inv(E_matrix.T @ E_matrix)
    W_ls = X_full @ (E_matrix @ inv_GtG)
    abundances = np.maximum(0.0, W_ls)

    oil_abundance = abundances[:, 1].reshape(H, W_total)
    oil_noise_floor = np.percentile(oil_abundance[bg_mask], 95)
    oil_abundance_clean = np.maximum(0.0, oil_abundance - oil_noise_floor)

    p99_5 = np.percentile(oil_abundance_clean[oil_abundance_clean > 0], 99.5) if np.any(oil_abundance_clean > 0) else np.max(oil_abundance_clean)
    oil_map_display = np.clip(oil_abundance_clean / (p99_5 + 1e-9), 0.0, 1.0)

    # 6. Save Abundance TIFF Image Files
    abundance_tif_16bit = os.path.join(output_dir, f"Silicone_Oil_Abundance_Map_Tile_{start_idx+1}_to_{end_idx}.tif")
    tifffile.imwrite(abundance_tif_16bit, (oil_map_display * 65535).astype(np.uint16))
    print(f"[SUCCESS] Saved Silicone Oil Abundance Map TIFF (16-bit) to: {abundance_tif_16bit}")

    # 7. Plot Figures & Save to Artifacts
    fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    bands_x = np.arange(1, B + 1)
    ax1.plot(bands_x, E_oil / np.max(E_oil), 'o-', color='#D32F2F', linewidth=2.5, markersize=5, label='Pure Silicone Oil Endmember (Double Peak)')
    ax1.plot(bands_x, E_bg / np.max(E_bg), 's--', color='#1976D2', linewidth=2.0, markersize=5, label='Clean Background Endmember')
    for pk in [19, 20]:
        pk_val = E_oil[pk - 1] / np.max(E_oil)
        ax1.annotate(f'Peak @ Band {pk}', xy=(pk, pk_val), xytext=(pk - 0.5, pk_val + 0.12),
                     arrowprops=dict(facecolor='#D32F2F', shrink=0.08, width=1.5, headwidth=6),
                     fontsize=10, fontweight='bold', color='#B71C1C')
    ax1.set_xlabel('Spectral Band Index (1 to 35)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
    ax1.set_title(f'Spectral Decomposition (Syringe 4: Tiles #{start_idx+1} to #{end_idx})', fontsize=13, fontweight='bold', pad=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.25)
    ax1.legend(loc='upper right', fontsize=10, frameon=True, facecolor='#F5F5F5')
    plt.tight_layout()
    spec_path = os.path.join(artifact_dir, f'syringe_4_spectral_decomposition_tiles_{start_idx+1}_to_{end_idx}.png')
    fig1.savefig(spec_path, dpi=200)
    plt.close(fig1)

    fig2, (ax_img, ax_prof) = plt.subplots(2, 1, figsize=(16, 6), gridspec_kw={'height_ratios': [2.5, 1]}, dpi=150)
    im = ax_img.imshow(oil_map_display, cmap='plasma', aspect='auto', extent=[0, W_total, H, 0])
    cbar = fig2.colorbar(im, ax=ax_img, orientation='vertical', pad=0.015, aspect=15)
    cbar.set_label('Silicone Oil Relative Abundance', fontsize=11, fontweight='bold')
    ax_img.set_title(f'Spatial Distribution Map of Silicone Oil (Syringe 4: Tiles #{start_idx+1} to #{end_idx}, Size: {H} × {W_total})', fontsize=14, fontweight='bold', pad=10)
    ax_img.set_ylabel('Y Position (px)', fontsize=11, fontweight='bold')
    ax_img.set_xlabel('X Position (px)', fontsize=11, fontweight='bold')

    profile_x = np.mean(oil_map_display, axis=0)
    ax_prof.plot(profile_x, color='#E65100', linewidth=1.5)
    ax_prof.fill_between(np.arange(W_total), profile_x, color='#FFE0B2', alpha=0.6)
    ax_prof.set_xlim(0, W_total)
    ax_prof.set_ylim(0, 1.05)
    ax_prof.set_xlabel('Syringe Length X Position (px)', fontsize=11, fontweight='bold')
    ax_prof.set_ylabel('Mean Abundance', fontsize=10, fontweight='bold')
    ax_prof.set_title('Longitudinal Concentration Profile of Silicone Oil Along Syringe 4', fontsize=11, fontweight='bold')
    ax_prof.grid(True, alpha=0.3)
    plt.tight_layout()

    dist_path = os.path.join(artifact_dir, f'syringe_4_abundance_distribution_tiles_{start_idx+1}_to_{end_idx}.png')
    fig2.savefig(dist_path, dpi=200)
    plt.close(fig2)

    print("[SUCCESS] All processes completed successfully!")
    return combined_tif_path, abundance_tif_16bit, spec_path, dist_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine tile TIFF files and run clean Silicone Oil NNLS unmixing.")
    parser.add_argument("--dir", default=r"C:\Users\DanFu\Downloads\Syringe_4", help="Input directory containing tile TIFF files")
    parser.add_argument("--start", type=int, default=1, help="Start tile number (1-indexed)")
    parser.add_argument("--end", type=int, default=14, help="End tile number (1-indexed)")
    args = parser.parse_args()

    combine_and_unmix(args.dir, start_tile=args.start, end_tile=args.end)
