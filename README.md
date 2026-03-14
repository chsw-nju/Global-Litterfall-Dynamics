# Global Litterfall Dynamics and Decoupling (GLDD)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the core codebase for estimating global forest litterfall (PFL) dynamics and evaluating its legacy effect on soil respiration. The workflow integrates Google Earth Engine (GEE) preprocessing, Frisch-Waugh-Lovell (FWL) asymmetric decoupling, Olson decay kinetics, and variance partitioning analysis (VPA). 

## 1. Project Overview

This project aims to quantify the biogeochemical coupling between forest litterfall production and soil respiration globally. 

The methodology is divided into two primary stages:
1.  **Remote Sensing-based Estimation:** Uses GEE and machine learning (Random Forest Regressor [RFR] and Gradient Boosting Decision Tree Regressor [GBDTR]) to estimate PFL across different forest biomes (e.g., Evergreen Needleleaf Forests [ENF], Deciduous Needleleaf Forests [DNF]). It incorporates a biome-constrained adaptive focal mean algorithm for gap-filling and strict QA/QC masking.
2.  **Statistical Decoupling and Legacy Analysis:** Employs Python-based spatial processing to isolate the substrate-driven signal from climatic confounders using the FWL theorem. It then applies Olson's first-order decay kinetics and VPA to quantify the legacy contribution of litterfall to soil respiration across multiple time-lag windows.

## 2. Directory Structure and Execution Workflow

The repository is structured to reflect the sequential analytical pipeline. 

### Stage 1: Google Earth Engine (GEE) Processing
These scripts must be run within the Google Earth Engine Code Editor.
* **`01a_GEE_RFR_Estimation.js`**: Reconstructs global PFL using an RFR model. Designed for biomes like ENF (IGBP Class 1). It uses MODIS products (MCD12Q1, MOD09A1, MOD13Q1, MOD15A2H, MOD17A3HGF) and includes 70/30 cross-validation. Note: You must update the `trainingFeature` asset path to your own uploaded dataset (e.g., `users/your_id/Global/RFR`).
* **`01b_GEE_GBDTR_Estimation.js`**: Reconstructs global PFL using a GBDTR model. Designed for biomes like DNF (IGBP Class 3). Features hyperparameter tuning (shrinkage, samplingRate, maxNodes) to prevent overfitting. Update the `trainingFeature` asset path accordingly (e.g., `users/your_id/Global/GBDTR`).
Handling Extreme Optical Contamination in Tropical Basins: As formally described in Section 2.2 of the manuscript, primary data gaps in NPP are resolved utilizing a climatological substitution protocol (e.g., the 2017–2021 interannual mean). However, in specific hyper-productive domains (such as the Central African and Amazon basins), persistent atmospheric contamination can occasionally render even the multi-year historical baseline unavailable (yielding persistent NaN arrays).To ensure the structural completeness of the feature matrix required for the Random Forest Regression (RFR) and Gradient Boosting (GBDTR) model executions, our preprocessing scripts incorporate an automated engineering fallback: remaining localized NaN pixels are reconstructed using empirical scaling coefficients derived from valid contemporaneous NDVI proxies. This strict fallback mechanism prevents algorithmic failure while preserving the broad-scale climatological baseline.

### Stage 2: Python Statistical Modeling
These scripts process the stacked multi-temporal GeoTIFFs generated from the GEE stage.
* **`02a_FWL_Asymmetric_Decoupling.py`**: Performs pixel-wise FWL residual analysis to decouple climatic variables (LST, ET, GPP, TEM, PRE) from RS and PFL. Outputs correlation (R), sensitivity slope, and p-value maps.
* **`02b_Olson_Legacy_Effect.py`**: Calculates the Effective Substrate Index (Seff) based on Olson kinetics across time-lag windows ($w=0$ to $5$) and decay constants ($k=0.1, 0.3, 0.5$). Uses VPA to determine the marginal explanatory gain (Delta R²) of cumulative substrate memory.

**Note on Python Execution:**
Before running the Python scripts, ensure you have correctly configured the `INPUT_DIR` and `OUTPUT_DIR` paths in the `Config` class of each script to match your local data structure. The input data should be stacked GeoTIFFs formatted as `VariableName_sample.tif` (e.g., `PFL_sample.tif`, `RS_sample.tif`).

## 3. Environment Dependencies

To run the Python scripts, you need the following libraries. Install them using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
