Leakage-Safe Validation for Geospatial ML

<p align="center">
<img width="300" height="200" alt="logo" src="https://github.com/user-attachments/assets/9e5d69ec-c477-452a-bcf3-a5696241b853" />
</p>

> Surveys have identified data leakage as a pervasive methodological flaw in 294 machine learning applications across 17 fields of study. Geoscience and remote sensing are among the most affected fields, often resulting in overoptimistic findings. One of the primary causes of data leakage in these contexts is the spatial dependency between datasets, which is called spatial autocorrelation (SAC).
> SAC, a dominant characteristic in environmental data, can inadvertently introduce information from outside the intended training set and compromise model validity. If SAC is not properly addressed, spatial analysis can be subject to misleading conclusions where the test set is not representative of the population about which scientific claims are made.

This repository tackles that problem directly by **spatially stratifying** your data *before* training/evaluation so validation reflects real-world generalization.

## What is Spatial Stratification?

Spatial stratification partitions geospatial samples into spatially meaningful groups before model training/evaluation. This reduces spatial leakage and produces more realistic validation by ensuring samples in the same fold are spatially coherent. In this repo, stratification implemented is a three-stage pipeline:


1. **Spatial Autocorrelation Assessment**
   Compute **Moran's I** and **Geary's C** to detect spatial dependence in the dataset.

2. **SAC Correlogram (Moran's I vs. Distance)**
   Scan Moran's I over multiple distance thresholds and detect the first stable-slope region as the cut off distance Where SAC become negligible. Use cut off distance as the **minimum centroid separation** for clustering.

3. **Advanced Spatial Clustering (SAC-informed)**
   Search over *K* with quality checks: minimum cluster size, centroid spacing (from SAC), class purity, and silhouette. Select *K*, fit **MiniBatchKMeans**, and optionally merge pure-dry clusters to improve balance. Export cluster maps and quality summaries.


**Outputs:** a `groups` vector ready for **GroupKFold** (or similar spatial CV) **plus** a rich `results` dictionary with metrics, plots, and clustering artifacts—so your models validate without SAC-driven leakage and your claims stand on firm ground.

## Quickstart

### Environment
(see `environment.yml`). If using `pip`, see `requirements.txt` and ensure native geospatial libs are installed (GEOS/PROJ/GDAL via Conda is recommended).

### Minimal Example

The spatial analysis pipeline is implemented in `src/SAC_Analysis.py` as a standalone script. Here's how to use it:

**Option 1: Run the script directly (recommended)**
```bash
python src/SAC_Analysis.py
```

**Option 2: Modify the script for your data**
Edit `src/SAC_Analysis.py` and change these lines at the top:
```python
# Line 31: Change the CSV path
CSV_PATH = Path("your_data_path.csv")

# Line 30: Change target column if needed  
TARGET = "your_target_column"

# Line 67-68: Adjust UTM zone if needed
UTM_ZONE = 39  # Change to your UTM zone
UTM_NORTH = True  # True for Northern hemisphere
```

Then run: `python src/SAC_Analysis.py`

**Required data format:**
- CSV file with columns: `"latitude"`, `"longitude"`, `"Label"` (binary target {0,1})
- Coordinates should be in UTM Zone 39N (EPSG:32639) - adjust zone as needed

**What the script produces:**

**Generated plots in `plots/`:**
- `sac_correlogram.png` - Moran's I vs distance with cutoff annotation
- `advanced_spatial_clusters.png` - Spatial clusters with flood/non-flood visualization
- `cluster_quality_summary.png` - Cluster statistics and quality metrics

**Global variables created (accessible after script execution):**
- `groups`: `pd.Series` of cluster IDs aligned to processed `df` (after cleaning/reindexing)
- `final_k`: selected number of clusters (post-merge)
- `kmeans`: fitted MiniBatchKMeans model
- `sac_analysis_results`: dictionary with SAC analysis metrics
- `spatial_autocorr_results`: dictionary with spatial autocorrelation statistics

**Note:** The script runs all analysis immediately when executed. Results are stored as global variables rather than returned from functions.



## Integrating with Model Training
Sample Spatially Informed Random Forest Model (Flood Hazard) : 
- A pre-trained Random Forest model trained with the spatially informing policy is included to showcase usage for flood hazard modeling.

- The repository includes a complete spatially-aware machine learning pipeline in `src/Spatially_Informed_RF.py` that demonstrates how to use the spatial groups for robust model training in the case of flood hazard modeling.

**Complete workflow:**
```bash
# 1. First run spatial analysis to generate groups
python src/SAC_Analysis.py

# 2. Then run the complete ML pipeline
python src/Spatially_Informed_RF.py
```

**Full pipeline includes:**
- **Nested cross-validation** with hyperparameter tuning using RandomizedSearchCV
- **Probability calibration** using IsotonicRegression to improve prediction reliability
- **Optimal threshold selection** based on F1-score maximization
- **Feature importance analysis** using permutation importance
- **Comprehensive performance metrics** including ROC-AUC, balanced accuracy, calibration error
- **Hazard mapping capabilities** for spatial visualization of flood risk

**Generated outputs:**
- `models/flood_model_final.joblib` - Trained Random Forest model
- `models/flood_model_final_calibrator.joblib` - Probability calibrator
- `models/flood_model_final_threshold.joblib` - Optimal classification threshold
- `plots/confusion_matrix.png` - Model performance visualization
- `plots/roc_curve.png` - ROC curve with optimal threshold
- `Results.txt` - Comprehensive performance report

- You can load these with `joblib.load(...)` and use alongside the `groups` produced by the spatial analysis pipeline for spatially robust evaluation. See `src/Spatially_Informed_RF.py` for details. 

## Sample Data
A sample dataset (\flood_data.csv\) is available in \data/sample/\. It contains the 100 rows from the original dataset and is intended for testing and demonstration purposes.