"""
=============================================================================
02a_fwl_asymmetric_decoupling.py
Pixel-wise Frisch-Waugh-Lovell (FWL) Asymmetric Residual Analysis
=============================================================================
This script processes stacked multi-temporal GeoTIFFs to decouple climatic
confounders from forest litterfall (PFL) and soil respiration (RS) dynamics.
It implements a memory-efficient, block-by-block processing architecture
suitable for global-scale high-resolution datasets.

Methodology: Multivariate Linear Residualization (Classic FWL Theorem)
=============================================================================
"""

import os
import gc
import warnings
from datetime import datetime
from typing import Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import stats
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# Suppress invalid value warnings during pixel-wise matrix division
warnings.filterwarnings('ignore')


# ==========================================
# 1. Configuration Module
# ==========================================
class Config:
    """Centralized configuration for I/O paths and algorithm parameters."""

    # Directories
    INPUT_DIR = "./data/stack"  # “data/stack/”
    OUTPUT_DIR = "./data/res"

    # Variable Definitions
    TARGET_Y = "RS"  # Response Variable
    TARGET_X = "PFL"  # Key Predictor (Substrate)
    # Exogenous climatic confounders to be decoupled
    CONTROLS = ["LST", "ET", "GPP", "TEM", "PRE"]

    # Processing Parameters
    BLOCK_SIZE = 256  # Process 256x256 pixel blocks to manage RAM
    NODATA_VAL = -9999.0  # Standard NoData value for output
    VALID_MIN = -9000.0  # Threshold to mask anomalous fill values


# ==========================================
# 2. Mathematical Core: FWL Engine
# ==========================================
class FWLEngine:
    """Encapsulates the statistical decoupling logic based on the FWL theorem."""

    @staticmethod
    def calculate_residual_correlation(pixel_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates partial correlation and sensitivity slope using OLS residualization.

        Args:
            pixel_matrix (np.ndarray): Shape (Time, Features).
                                       Col 0: Y (RS), Col 1: X (PFL), Col 2+: Z (Controls)
        Returns:
            Tuple[r_val, slope, p_val]
        """
        # Filter finite and valid values strictly across the temporal axis
        mask = np.all(np.isfinite(pixel_matrix) & (pixel_matrix > Config.VALID_MIN), axis=1)
        df_clean = pixel_matrix[mask]

        n_samples = df_clean.shape[0]
        # Require at least 2/3 of the time series to be valid (e.g., >= 14 years for a 21-yr series)
        if n_samples < 14:
            return Config.NODATA_VAL, Config.NODATA_VAL, Config.NODATA_VAL

        try:
            # Step 1: Z-score Standardization (Critical for comparing multi-scale variables)
            std_vals = np.std(df_clean, axis=0)
            if np.any(std_vals < 1e-6):
                return Config.NODATA_VAL, Config.NODATA_VAL, Config.NODATA_VAL

            z_data = (df_clean - np.mean(df_clean, axis=0)) / std_vals

            y_target = z_data[:, 0].reshape(-1, 1)  # Normalized RS
            x_target = z_data[:, 1].reshape(-1, 1)  # Normalized PFL
            z_controls = z_data[:, 2:]  # Normalized Climate Controls

            # Step 2: Asymmetric Residual Decoupling (FWL Theorem)
            reg = LinearRegression()
            res_y = y_target - reg.fit(z_controls, y_target).predict(z_controls)
            res_x = x_target - reg.fit(z_controls, x_target).predict(z_controls)

            res_y_flat = res_y.flatten()
            res_x_flat = res_x.flatten()

            # Step 3: Statistical Metrics Extraction
            r_val, p_val = stats.pearsonr(res_x_flat, res_y_flat)
            slope, _, _, _, _ = stats.linregress(res_x_flat, res_y_flat)

            return float(np.clip(r_val, -1.0, 1.0)), float(slope), float(p_val)

        except Exception:
            return Config.NODATA_VAL, Config.NODATA_VAL, Config.NODATA_VAL


# ==========================================
# 3. Spatial Processing Module
# ==========================================
class SpatialProcessor:
    """Handles geo-spatial I/O, block windowing, and matrix transformations."""

    def __init__(self, cfg: type):
        self.cfg = cfg
        self.var_order = [cfg.TARGET_Y, cfg.TARGET_X] + cfg.CONTROLS
        self.file_paths = []
        self.meta = None
        self.time_steps = 0

    def _initialize_and_validate(self):
        """Validates input files and synchronizes spatial metadata."""
        print(f"\n[*] Initializing Spatial Engine at {datetime.now().strftime('%H:%M:%S')}")

        for var in self.var_order:
            # Adapted to the new standardized naming convention
            f_path = os.path.join(self.cfg.INPUT_DIR, f"{var}_sample.tif")
            if not os.path.exists(f_path):
                raise FileNotFoundError(f"[!] Critical Error: Missing {f_path}")
            self.file_paths.append(f_path)

        # Extract metadata from the first file (RS)
        with rasterio.open(self.file_paths[0]) as src:
            self.meta = src.meta.copy()
            self.time_steps = src.count  # Dynamic band sniffing (e.g., 21 bands)

        print(
            f"[*] Metadata Synchronized: {self.meta['width']}x{self.meta['height']} pixels, {self.time_steps} temporal bands.")

    def run_global_decoupling(self):
        """Executes the block-wise FWL analysis and exports results."""
        self._initialize_and_validate()

        width = self.meta['width']
        height = self.meta['height']
        block_size = self.cfg.BLOCK_SIZE

        # Pre-allocate output arrays
        out_r = np.full((height, width), self.cfg.NODATA_VAL, dtype=np.float32)
        out_slope = np.full((height, width), self.cfg.NODATA_VAL, dtype=np.float32)
        out_p = np.full((height, width), self.cfg.NODATA_VAL, dtype=np.float32)

        print("[*] Commencing Global Block-wise Residual Processing...")
        total_blocks = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)

        with tqdm(total=total_blocks, desc="FWL Decoupling") as pbar:
            for row_start in range(0, height, block_size):
                for col_start in range(0, width, block_size):

                    window_h = min(block_size, height - row_start)
                    window_w = min(block_size, width - col_start)
                    window = Window(col_off=col_start, row_off=row_start, width=window_w, height=window_h)

                    # block_stack shape: (Num_Vars, Time, Y, X)
                    block_stack = np.full((len(self.var_order), self.time_steps, window_h, window_w),
                                          np.nan, dtype=np.float32)

                    # Load window data for all variables
                    read_error = False
                    for v_idx, f_path in enumerate(self.file_paths):
                        try:
                            with rasterio.open(f_path) as src:
                                data = src.read(window=window).astype(np.float32)
                                nd_val = src.nodata
                                if nd_val is not None:
                                    data[data == nd_val] = np.nan
                                data[data < self.cfg.VALID_MIN] = np.nan
                                block_stack[v_idx] = data
                        except Exception:
                            read_error = True
                            break

                    if read_error:
                        pbar.update(1)
                        continue

                    # Reshape for pixel-wise computation: (Y, X, Time, Num_Vars)
                    # Transpose shifts axes so that iteration over spatial dims gives (Time, Num_Vars) matrix
                    block_transposed = block_stack.transpose(2, 3, 1, 0)

                    # Iterate over spatial dimensions within the block
                    for i in range(window_h):
                        for j in range(window_w):
                            pixel_matrix = block_transposed[i, j]  # Shape: (Time, Num_Vars)

                            # Quick skip if the key variables (RS or PFL) are entirely NaN
                            if np.isnan(pixel_matrix[:, 0]).all() or np.isnan(pixel_matrix[:, 1]).all():
                                continue

                            rv, sv, pv = FWLEngine.calculate_residual_correlation(pixel_matrix)

                            out_r[row_start + i, col_start + j] = rv
                            out_slope[row_start + i, col_start + j] = sv
                            out_p[row_start + i, col_start + j] = pv

                    pbar.update(1)
                    gc.collect()  # Aggressive memory clearing per block

        self._export_results(out_r, out_slope, out_p)

    def _export_results(self, r_arr, slope_arr, p_arr):
        """Exports the computed statistical matrices to a multi-band GeoTIFF."""
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        out_file = os.path.join(self.cfg.OUTPUT_DIR, "FWL_ResAnalysis_Global.tif")

        out_meta = self.meta.copy()
        out_meta.update({
            'count': 3,
            'dtype': 'float32',
            'nodata': self.cfg.NODATA_VAL,
            'compress': 'lzw'
        })

        print(f"\n[*] Exporting analytical rasters to: {out_file}")
        with rasterio.open(out_file, 'w', **out_meta) as dst:
            dst.write(r_arr, 1)
            dst.set_band_description(1, 'Correlation_R')

            dst.write(slope_arr, 2)
            dst.set_band_description(2, 'Sensitivity_Slope')

            dst.write(p_arr, 3)
            dst.set_band_description(3, 'P_Value')

        print("[+] Processing Complete.")


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    try:
        start_time = datetime.now()

        processor = SpatialProcessor(Config)
        processor.run_global_decoupling()

        print(f"[*] Total Execution Time: {datetime.now() - start_time}")
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")



