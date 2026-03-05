"""
=============================================================================
02b_olson_legacy_effect.py
Spatially-Explicit Olson Kinetics and Variance Partitioning Analysis (VPA)
=============================================================================
This script evaluates the legacy effect of forest litterfall (PFL) on soil
respiration (RS). It implements a first-order exponential decay model (Olson
Kinetics) across multiple time-lag windows and decay constants, utilizing VPA
to isolate the marginal explanatory gain of substrate memory.

Methodology: Olson Decay Kinetics + OLS Variance Partitioning
=============================================================================
"""

import os
import gc
import warnings
from datetime import datetime
from typing import Tuple, List

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import signal
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# 抑制非关键性统计警告 (如全 NaN 切片的均值计算)
warnings.filterwarnings('ignore')


# ==========================================
# 1. Configuration Module
# ==========================================
class Config:
    """Centralized configuration for I/O and algorithm parameters."""

    # I/O Directories
    INPUT_DIR = "./data/stack"  # “data/stack/”
    OUTPUT_DIR = "./data/vpa"

    # Variable Configuration (Mapped to the Stacked TIFFs)
    TARGET_Y = "RS"
    TARGET_X = "PFL"
    # Environmental Driving Factors (EDFs) for the baseline model
    # Note: Ensure these perfectly match the variable prefixes in your stacked TIFFs
    ENV_CONTROLS = ["TEM", "PRE", "GPP"]

    # Algorithm Parameters
    MAX_W = 5  # Maximum lag window (years)
    K_VALUES = [0.1, 0.3, 0.5]  # Substrate decay constants for sensitivity analysis

    # Processing Parameters
    BLOCK_SIZE = 256  # Spatial block size for RAM optimization
    NODATA_VAL = -9999.0  # Standard NoData value


# ==========================================
# 2. Mathematical Core: Kinetics & VPA Engine
# ==========================================
class KineticsEngine:
    """Encapsulates the mathematical models for substrate memory and VPA."""

    @staticmethod
    def zscore_detrend(array_1d: np.ndarray) -> np.ndarray:
        """Applies linear detrending and Z-score standardization."""
        detrended = signal.detrend(array_1d, type='linear')
        std_dev = np.std(detrended)
        if std_dev < 1e-7:
            return None
        return (detrended - np.mean(detrended)) / std_dev

    @staticmethod
    def calculate_vpa_pixel(pixel_matrix: np.ndarray, w: int, k: float, max_w: int) -> Tuple[float, float, float]:
        """
        Executes Olson kinetics construction and VPA for a single pixel time-series.

        Args:
            pixel_matrix (np.ndarray): Shape (Time, Vars). Order: [RS, PFL, ENV1, ENV2, ENV3]
            w (int): Current lag window.
            k (float): Decay constant.
            max_w (int): Maximum lag window for strict temporal alignment.
        """
        time_steps = pixel_matrix.shape[0]

        # 1. Check data validity
        # Strictly avoid artificial interpolation; reject pixels with invalid data
        if np.isnan(pixel_matrix).any():
            return Config.NODATA_VAL, Config.NODATA_VAL, Config.NODATA_VAL

        # 2. Extract sequences
        y_raw = pixel_matrix[:, 0]
        pfl_raw = pixel_matrix[:, 1]
        env_raw = pixel_matrix[:, 2:]

        # 3. Construct Effective Substrate Index (Seff) based on Olson Kinetics
        # Seff(t) = Sum_{tau=0}^w [ PFL(t-tau) * e^(-k * tau) ]
        seff_full = np.zeros(time_steps)
        weights = np.exp(-k * np.arange(w + 1))

        for t in range(w, time_steps):
            # Extract historical PFL chunk: from t-w to t, reversed to match weights [tau=0, 1, ..., w]
            chunk = pfl_raw[t - w: t + 1][::-1]
            seff_full[t] = np.dot(chunk, weights)

        # 4. Strict Temporal Alignment (CRITICAL)
        # All regressions must run strictly from t = max_w to end to ensure comparability
        y_target = y_raw[max_w:]
        env_target = env_raw[max_w:, :]
        seff_target = seff_full[max_w:]

        # 5. Detrending and Standardization on the aligned temporal subset
        y_norm = KineticsEngine.zscore_detrend(y_target)
        seff_norm = KineticsEngine.zscore_detrend(seff_target)

        if y_norm is None or seff_norm is None:
            return Config.NODATA_VAL, Config.NODATA_VAL, Config.NODATA_VAL

        env_norm = np.zeros_like(env_target)
        for c in range(env_target.shape[1]):
            norm_c = KineticsEngine.zscore_detrend(env_target[:, c])
            if norm_c is None:
                return Config.NODATA_VAL, Config.NODATA_VAL, Config.NODATA_VAL
            env_norm[:, c] = norm_c

        # 6. Variance Partitioning Analysis (VPA) via OLS
        try:
            # Baseline Model (Environmental Controls Only)
            reg_base = LinearRegression().fit(env_norm, y_norm)
            r2_base = reg_base.score(env_norm, y_norm)

            # Full Model (Environmental Controls + Substrate Memory)
            x_full = np.column_stack((env_norm, seff_norm))
            reg_full = LinearRegression().fit(x_full, y_norm)
            r2_full = reg_full.score(x_full, y_norm)

            # Marginal Explanatory Gain (Delta R2)
            r2_gain = max(0.0, r2_full - r2_base)

            return float(r2_base), float(r2_full), float(r2_gain)
        except Exception:
            return Config.NODATA_VAL, Config.NODATA_VAL, Config.NODATA_VAL


# ==========================================
# 3. Spatial I/O and Processing Module
# ==========================================
class SpatialVPAProcessor:
    """Manages geospatial data ingestion, block processing, and multi-scenario output."""

    def __init__(self, cfg: type):
        self.cfg = cfg
        self.var_order = [cfg.TARGET_Y, cfg.TARGET_X] + cfg.ENV_CONTROLS
        self.file_paths = []
        self.meta = None
        self.time_steps = 0

    def _initialize(self):
        """Validates stacked TIFFs and retrieves global metadata."""
        print(f"\n[*] Initializing VPA Engine at {datetime.now().strftime('%H:%M:%S')}")

        for var in self.var_order:
            f_path = os.path.join(self.cfg.INPUT_DIR, f"{var}_sample.tif")
            if not os.path.exists(f_path):
                raise FileNotFoundError(f"[!] Critical Error: Missing stacked data {f_path}")
            self.file_paths.append(f_path)

        with rasterio.open(self.file_paths[0]) as src:
            self.meta = src.meta.copy()
            self.time_steps = src.count

        # Ensure sufficient time steps to support the maximum lag window
        if self.time_steps <= self.cfg.MAX_W + 5:
            raise ValueError(f"Time series too short ({self.time_steps}) for MAX_W={self.cfg.MAX_W}.")

        print(f"[*] Input Data Validated: {self.meta['width']}x{self.meta['height']} pixels, {self.time_steps} bands.")

    def run_sensitivity_analysis(self):
        """Executes the VPA processing loop over all combinations of K and W."""
        self._initialize()

        width = self.meta['width']
        height = self.meta['height']
        block_size = self.cfg.BLOCK_SIZE

        total_blocks = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)

        # Outer loop: Decay constants (Sensitivity Analysis)
        for k in self.cfg.K_VALUES:
            k_str = f"k{int(k * 10):02d}"
            out_sub_dir = os.path.join(self.cfg.OUTPUT_DIR, k_str)
            os.makedirs(out_sub_dir, exist_ok=True)

            print(f"\n=======================================================")
            print(f">>> Commencing Sensitivity Analysis for Decay Constant: {k}")
            print(f"=======================================================")

            # Inner loop: Lag windows
            for w in range(0, self.cfg.MAX_W + 1):
                out_r2_base = np.full((height, width), self.cfg.NODATA_VAL, dtype=np.float32)
                out_r2_full = np.full((height, width), self.cfg.NODATA_VAL, dtype=np.float32)
                out_r2_gain = np.full((height, width), self.cfg.NODATA_VAL, dtype=np.float32)

                with tqdm(total=total_blocks, desc=f"Processing w={w}") as pbar:
                    for row_start in range(0, height, block_size):
                        for col_start in range(0, width, block_size):

                            window_h = min(block_size, height - row_start)
                            window_w = min(block_size, width - col_start)
                            window = Window(col_off=col_start, row_off=row_start, width=window_w, height=window_h)

                            # block_stack shape: (Num_Vars, Time, Y, X)
                            block_stack = np.full((len(self.var_order), self.time_steps, window_h, window_w),
                                                  np.nan, dtype=np.float32)

                            # Read block for all variables
                            read_error = False
                            for v_idx, f_path in enumerate(self.file_paths):
                                try:
                                    with rasterio.open(f_path) as src:
                                        data = src.read(window=window).astype(np.float32)
                                        nd_val = src.nodata
                                        if nd_val is not None:
                                            data[data == nd_val] = np.nan
                                        # Mask anomalous values
                                        data[data < -9000] = np.nan
                                        block_stack[v_idx] = data
                                except Exception:
                                    read_error = True
                                    break

                            if read_error:
                                pbar.update(1)
                                continue

                            # Transpose to (Y, X, Time, Vars) for pixel-wise iteration
                            block_transposed = block_stack.transpose(2, 3, 1, 0)

                            for i in range(window_h):
                                for j in range(window_w):
                                    pixel_matrix = block_transposed[i, j]
                                    r2b, r2f, r2g = KineticsEngine.calculate_vpa_pixel(pixel_matrix, w, k,
                                                                                       self.cfg.MAX_W)

                                    out_r2_base[row_start + i, col_start + j] = r2b
                                    out_r2_full[row_start + i, col_start + j] = r2f
                                    out_r2_gain[row_start + i, col_start + j] = r2g

                            pbar.update(1)
                            gc.collect()

                # Export results for the current (K, W) combination
                self._export_window_results(out_sub_dir, w, k_str, out_r2_base, out_r2_full, out_r2_gain)

    def _export_window_results(self, out_dir: str, w: int, k_str: str, r2_base: np.ndarray, r2_full: np.ndarray,
                               r2_gain: np.ndarray):
        """Exports the 3-band VPA output raster."""
        out_file = os.path.join(out_dir, f"VPA_Contribution_w{w}_{k_str}.tif")

        out_meta = self.meta.copy()
        out_meta.update({
            'count': 3,
            'dtype': 'float32',
            'nodata': self.cfg.NODATA_VAL,
            'compress': 'lzw'
        })

        with rasterio.open(out_file, 'w', **out_meta) as dst:
            dst.write(r2_base, 1)
            dst.set_band_description(1, 'Env_Baseline_R2')

            dst.write(r2_full, 2)
            dst.set_band_description(2, 'Full_Model_R2')

            dst.write(r2_gain, 3)
            dst.set_band_description(3, 'Legacy_Gain_R2')


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    try:
        start_time = datetime.now()

        # Instantiate and run the VPA Processor
        processor = SpatialVPAProcessor(Config)
        processor.run_sensitivity_analysis()

        print(f"\n[+] Total Execution Time: {datetime.now() - start_time}")
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")

