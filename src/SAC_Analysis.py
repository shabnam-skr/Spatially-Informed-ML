# This module contains spatial analysis pipeline with the two main stages:
# Stage 1- Spatial Autocorrelation Assessment (target and key features) 
# Stage 2- SAC Correlogram (Moran’s I vs. Distance) 
# Stage 3- Hybrid Spatial Clustering 

import sys
import logging
from pathlib import Path
from scipy.signal import savgol_filter
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import traceback
from typing import Optional, List
import geopandas as gpd
import esda
import libpysal as lps
from shapely.geometry import Point
SPATIAL_ANALYSIS_AVAILABLE = True

# Basic configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEBUG = bool(globals().get('DEBUG', True))
PLOTS_DIR = Path(globals().get('PLOTS_DIR', "plots"))
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# define variables
TARGET = str(globals().get('TARGET', "Label"))
CSV_PATH = Path(globals().get('CSV_PATH', r"C:\Users\ASUS\Desktop\Spatially informed  ML\Ahwz_flood\data\flood data.csv"))

# load data and validate
df = globals().get('df')
if df is None:
    try:
        csv_path = Path(CSV_PATH)
        logging.info(f"Loading dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} rows x {df.shape[1]} columns")

        # Validate coordinate columns and target
        required_columns = {"latitude", "longitude"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing coordinate columns: {sorted(missing_columns)}")
        if TARGET not in df.columns:
            raise ValueError(f"Missing target column '{TARGET}'. Sample columns: {list(df.columns)[:20]}")

        # Coerce to numeric and drop invalid rows
        numeric_columns = ["latitude", "longitude", TARGET]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        initial_rows = len(df)
        df = df.dropna(subset=numeric_columns)
        removed_rows = initial_rows - len(df)
        if removed_rows:
            logging.info(f"Dropped {removed_rows} rows with invalid coordinates/target")
        # ensure integer target once after cleaning
        df[TARGET] = df[TARGET].astype(int)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        if DEBUG:
            traceback.print_exc()
        sys.exit(1)

# UTM configuration (adjust if zone/hemisphere differs)
UTM_ZONE = int(globals().get('UTM_ZONE', 39))
UTM_NORTH = bool(globals().get('UTM_NORTH', True))
UTM_ZONE = max(1, min(60, UTM_ZONE))  # clamp to valid UTM zones
EPSG_UTM = f"EPSG:{32600 + UTM_ZONE if UTM_NORTH else 32700 + UTM_ZONE}"

##      Stage 1: Spatial Autocorrelation Assessment (target and key features)      ##
logging.info("\n=== Spatial Autocorrelation Assessment ===")
spatial_autocorr_results = {}
try:
    # Use original geographic coordinates for neighborhood construction
    coords = df[["longitude", "latitude"]].values

    # Build KNN spatial weights
    w_knn = lps.weights.KNN.from_array(coords, k=8)
    w_knn.transform = 'r'

    # Target variable assessment
    moran_target = esda.Moran(df[TARGET].values, w_knn, permutations=999)
    geary_target = esda.Geary(df[TARGET].values, w_knn, permutations=999)
    spatial_autocorr_results[TARGET] = {
        'moran_i': float(moran_target.I),
        'moran_p': float(moran_target.p_sim),
        'geary_c': float(geary_target.C),
        'geary_p': float(geary_target.p_sim),
    }

    # Selected key features (if present)
    key_features = ["Slope", "DistoRiver", "Rainfall", "NDVI", "Aspect", "DEM"]
    features_to_test = [f for f in key_features if f in df.columns]

    significant_features = []
    for feature in features_to_test:
        if np.var(df[feature]) > 0:
            try:
                moran_feat = esda.Moran(df[feature].values, w_knn, permutations=999)
                spatial_autocorr_results[feature] = {
                    'moran_i': float(moran_feat.I),
                    'moran_p': float(moran_feat.p_sim),
                }
                if moran_feat.p_sim < 0.05:
                    significant_features.append(feature)
            except Exception as fe:
                logging.debug(f"Autocorrelation test failed for {feature}: {fe}")

    total_tested = 1 + len(features_to_test)
    significant_count = (1 if moran_target.p_sim < 0.05 else 0) + len(significant_features)
    if significant_count > 0:
        logging.warning(f"Spatial dependency detected in {significant_count}/{total_tested} variables")
    else:
        logging.info("No significant spatial autocorrelation detected in tested variables")
except Exception as e:
    logging.warning(f"Spatial autocorrelation assessment failed: {e}")
    spatial_autocorr_results = {TARGET: {'moran_i': float('nan'), 'moran_p': float('nan')}}
logging.info("Spatial autocorrelation assessment completed.")

##    Stage 2: SAC Correlogram    ##

logging.info("\n=== SAC Correlogram Analysis  ===")

# Applies Savitzky-Golay smoothing to reduce noise in the correlogram 
SMOOTH_CURVE = True        
SG_WINDOW = 7              
SG_POLY = 2  

def Smooth_Curve(y: np.ndarray, window: int = SG_WINDOW, poly: int = SG_POLY) -> np.ndarray:
    data_points_available = len(y)
    is_window_valid = window >= 3 and window % 2 == 1  
    is_enough_data = data_points_available >= window

    if not (is_window_valid and is_enough_data):
        return y

    safe_polynomial_degree = min(poly, window - 1)

    smoothed_values = savgol_filter(
        y,
        window_length=window,
        polyorder=safe_polynomial_degree
    )

    return smoothed_values

def Get_Cluster_Stats(g):
    stats = g.groupby("cluster")[TARGET].agg(["count", "sum"])
    stats["ratio"] = stats["sum"] / stats["count"]
    return stats
     
# Uses a moving window slope method to find where Moran's I stabilizes 
WINDOW_SLOPE_SIZE = 8      
WINDOW_SLOPE_PERSIST = 2   
WINDOW_SLOPE_TOL = 0.1     

def Cutoff_Slope(distances: np.ndarray,
                           y: np.ndarray,
                           window: int = 10,
                           slope_tol: float = 0.8,
                           persist: int = 2) -> Optional[float]:
    x = np.asarray(distances, float)
    y = np.asarray(y, float)
    if len(x) < window or len(y) < window or window < 2:
        return None
    hits: List[bool] = []
    for i in range(len(x) - window + 1):
        xi, yi = x[i:i+window], y[i:i+window]
        if not (np.all(np.isfinite(xi)) and np.all(np.isfinite(yi))):
            hits.append(False)
            continue
        a = np.polyfit(xi, yi, 1)[0]
        avg_step = (xi[-1] - xi[0]) / (window - 1)
        change_per_step = abs(a * avg_step)
        hits.append(change_per_step < slope_tol)
    count = 0
    for i, h in enumerate(hits):
        count = count + 1 if h else 0
        if count >= persist:
            return float(x[i + window - 1])
    return None

#SAC analysis
sac_cutoff = None
sac_analysis_results = {}

# Initialize variables for export (will be set during execution)
advanced_clustering_success = False
final_k = None
groups = None
kmeans = None
MIN_PTS = 20  
SAC_RANGE = None

# Build GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df.longitude, df.latitude)],
    crs=EPSG_UTM
)
y_sac = gdf[TARGET].values


# calculate Moran's I
distances = np.linspace(200, 5000, 15)  
morans = []

for d in distances:
    try:
        w = lps.weights.DistanceBand.from_dataframe(gdf, threshold=d, silence_warnings=True)
        w.transform = 'R'
        mi = esda.Moran(y_sac, w)
        morans.append(mi.I)
        logging.debug(f"  d = {int(d):>4} m -> Moran's I = {mi.I:.4f}")
    except Exception as e:
        logging.warning(f"  d = {int(d):>4} m -> Error: {str(e)}")
        morans.append(np.nan)

# Choose cutoff 
chosen_method = None
morans_arr = np.asarray(morans, dtype=float)
y_used = Smooth_Curve(morans_arr, SG_WINDOW, SG_POLY) if Smooth_Curve else morans_arr
sac_cutoff = Cutoff_Slope(
    np.asarray(distances, dtype=float),
    y_used,
    window=WINDOW_SLOPE_SIZE,
    slope_tol=WINDOW_SLOPE_TOL,
    persist=WINDOW_SLOPE_PERSIST,
)

if sac_cutoff is not None:
    chosen_method = "moving_window_slope"
    logging.info(f"\n[OK] Moving-window slope below {WINDOW_SLOPE_TOL} for {WINDOW_SLOPE_PERSIST} windows at ~{int(sac_cutoff)} m")
else:
    logging.warning(f"No stable slope region found (tol={WINDOW_SLOPE_TOL}, window={WINDOW_SLOPE_SIZE}, persist={WINDOW_SLOPE_PERSIST})")

# Record results
if sac_cutoff is not None:
    sac_analysis_results['cutoff_distance'] = float(sac_cutoff)
    sac_analysis_results['selection_method'] = chosen_method
    sac_analysis_results['window_slope'] = {
        'window': int(WINDOW_SLOPE_SIZE),
        'slope_tol': float(WINDOW_SLOPE_TOL),
        'persist': int(WINDOW_SLOPE_PERSIST),
    }
else:
    sac_analysis_results['cutoff_distance'] = None
    sac_analysis_results['selection_method'] = None

sac_analysis_results['distances'] = distances.tolist()
sac_analysis_results['morans_i'] = morans
sac_analysis_results['max_moran'] = max(morans) if morans else 0
sac_analysis_results['min_moran'] = min(morans) if morans else 0

# Attach spatial autocorrelation summary
sac_analysis_results['spatial_autocorr'] = spatial_autocorr_results

# Creates a plot showing Moran's I vs distance with the selected cutoff
try:
    plt.figure(figsize=(8, 6))
    plt.plot(distances, morans, marker="o", linewidth=2, markersize=8, label="Moran's I")
    plt.axhline(0, color="gray", ls="--", alpha=0.7)
    if sac_cutoff is not None:
        label = f"Cutoff ({chosen_method}): {int(sac_cutoff)} m" if chosen_method else f"Cutoff: {int(sac_cutoff)} m"
        plt.axvline(sac_cutoff, color="red", ls="--", alpha=0.8, label=label)
    plt.xlabel("Threshold distance (m)")
    plt.ylabel("Moran's I")
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    finite_m = [m for m in morans if m == m]
    plt.ylim(bottom=min(0, min(finite_m)) if finite_m else 0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"sac_correlogram.png", dpi=300)
    plt.close()
    logging.info("Saved SAC correlogram plot with cutoff annotation")
except Exception as e:
    logging.error(f"Error creating SAC correlogram plot: {str(e)}")

if sac_cutoff is not None:
    logging.info(f"SAC correlogram analysis completed. Cutoff distance: {int(sac_cutoff)} m")
else:
    logging.info("SAC correlogram analysis completed. No cutoff selected.")


##      Stage 3 : Hybrid Spatial Clustering       ##

logging.info("\n=== Advanced Spatial Clustering with hybrid K Selection ===")

# Data Cleaning
if "latitude" in df.columns and "longitude" in df.columns and SPATIAL_ANALYSIS_AVAILABLE:
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import silhouette_score
        from scipy.spatial.distance import cdist
        from sklearn.neighbors import NearestNeighbors
        
        # Use UTM coordinates from GeoDataFrame geometry for clustering
        coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
        logging.info("Using UTM coordinates from GeoDataFrame geometry")
        
        # Validate coordinates 
        finite_mask = np.isfinite(coords).all(axis=1)
        if not finite_mask.all():
            logging.warning(f"Found {(~finite_mask).sum()} invalid coordinates (inf/NaN), removing them")
            coords = coords[finite_mask]
            gdf = gdf[finite_mask].reset_index(drop=True)
            df = df[finite_mask].reset_index(drop=True)
        
        logging.info(f"Using {len(coords)} valid coordinate pairs for clustering")
        
        ##  Primary step  ## 
        # 1. Minimum Point Thresholds
        MIN_PTS = 20  
        SAC_RANGE = round(sac_cutoff / 1000) * 1000 if sac_cutoff is not None else None 
        K_RANGE = range(10, 25)
        
        logging.info(f"Using SAC-informed minimum centroid separation: {SAC_RANGE}m")

        # Check quality for a single k
        def K_eval(k):
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256)
            labels = kmeans.fit_predict(coords)

            g = gdf.copy()
            g["cluster"] = labels
            sizes = g.groupby("cluster").size()

            if sizes.min() < MIN_PTS:
                return {"k": k, "valid": False, "reason": "cluster < MIN_PTS"}

            # 2.Optimal centroid separation distances
            D = cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)
            np.fill_diagonal(D, np.inf)
            if D.min() < SAC_RANGE:
                return {"k": k, "valid": False, "reason": "centroids < SAC_RANGE"}

            # 3.Class urity threshold
            ratio = g.groupby("cluster")[TARGET].mean()
            pure = ((ratio == 0) | (ratio == 1)).sum()
            if pure > 0.3 * k:  # >30% pure clusters -> reject
                return {"k": k, "valid": False, "reason": "too many pure clusters"}

            return {"k": k, "valid": True,
                    "min_size": int(sizes.min()), "max_size": int(sizes.max()),
                    "min_dist": float(D.min()), "pure_clusters": int(pure)}

        ##  Secondary step  ## 
        # Enhanced evaluation function with silhouette analysis
        def K_eval_silhouette(k):

            result = K_eval(k)
            
            if not result["valid"]:
                return result
            
            try:
                # Calculate silhouette score for valid k values
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256)
                labels = kmeans.fit_predict(coords)
                
                silhouette_avg = silhouette_score(coords, labels)

                result["silhouette_score"] = float(silhouette_avg)
                
                if silhouette_avg > 0.5:
                    result["silhouette_quality"] = "good"
                elif silhouette_avg > 0.25:
                    result["silhouette_quality"] = "fair"
                else:
                    result["silhouette_quality"] = "poor"
                    
            except Exception as e:
                result["silhouette_score"] = None
                result["silhouette_quality"] = "error"

                print(f"Silhouette calculation failed for k={k}: {e}")
            
            return result

        # 1. K selection with silhouette analysis
        logging.info("\nEvaluating K-range with silhouette analysis...")

        k_eval_results = []
        silhouette_scores = []

        for k in K_RANGE:
            single_k_eval = K_eval_silhouette(k)
            k_eval_results.append(single_k_eval)

            if single_k_eval["valid"] and single_k_eval.get("silhouette_score") is not None:
                silhouette_scores.append((k, single_k_eval["silhouette_score"]))
                logging.info(f"K={k}: Valid, Silhouette={single_k_eval['silhouette_score']:.3f}, Quality={single_k_eval['silhouette_quality']}")
            else:
                reason = single_k_eval.get("reason", "unknown")
                logging.info(f"K={k}: Invalid ({reason}) or silhouette calculation failed")

        # Convert to DataFrame for analysis
        k_eval_df = pd.DataFrame(k_eval_results)
        valid_k_evals = k_eval_df[k_eval_df["valid"] == True]

        logging.info("\nDetailed Cluster Quality Assessment Results:")
        logging.info("\n%s", k_eval_df[['k', 'valid', 'silhouette_score', 'silhouette_quality']].to_string())

        if not valid_k_evals.empty and silhouette_scores:
            # Select best K based on highest silhouette score
            valid_silhouette = [(k, score) for k, score in silhouette_scores 
                               if k in valid_k_evals["k"].values]
            
            if valid_silhouette:
                best_k = max(valid_silhouette, key=lambda x: x[1])[0]
                best_score = max(valid_silhouette, key=lambda x: x[1])[1]
                logging.info(f"\nBest K selected: {best_k} (silhouette score: {best_score:.3f})")
            else:
                logging.warning("No valid K with silhouette scores found!")
                best_k = None
        else:
            logging.warning("No valid K found!")
            best_k = None

        if best_k is None:
            raise ValueError("No k in K_RANGE satisfied all criteria.")
        
        logging.info(f"\n[OK] Best k chosen = {best_k}")

        # 2. Merging and rechecking clusters
        
        kmeans_best = MiniBatchKMeans(n_clusters=int(best_k), random_state=42)
        gdf["cluster"] = kmeans_best.fit_predict(coords)

        def Merg_Pure_Clustrs(g):
            stats = Get_Cluster_Stats(g)
            pure = stats.index[stats["ratio"] == 0]
            if pure.empty:
                return g
            cent = g.dissolve(by="cluster").centroid
            cent = gpd.GeoDataFrame(geometry=cent, crs=g.crs)
            cent["x"], cent["y"] = cent.geometry.x, cent.geometry.y
            flood_cent = cent.drop(index=pure)
            nn = NearestNeighbors(n_neighbors=1).fit(flood_cent[["x","y"]])
            for dry_id in pure:
                # Keep as DataFrame to maintain feature names consistency
                pt = cent.loc[[dry_id], ["x","y"]]
                nearest = flood_cent.iloc[nn.kneighbors(pt)[1][0][0]].name
                g.loc[g.cluster == dry_id, "cluster"] = nearest
            return g

        gdf_balanced = Merg_Pure_Clustrs(gdf)
        
        unique_labels = sorted(gdf_balanced["cluster"].unique())
        label_map = {old: new for new, old in enumerate(unique_labels)}
        gdf_balanced["cluster"] = gdf_balanced["cluster"].map(label_map)

        def Min_Center_Distance(g):
            cent = g.dissolve(by="cluster").centroid
            D = cdist(np.column_stack([cent.x, cent.y]),
                      np.column_stack([cent.x, cent.y]))
            np.fill_diagonal(D, np.inf)
            return D.min()

        min_dist = Min_Center_Distance(gdf_balanced)
        logging.info(f">> Min centroid distance after merge: {min_dist:.0f} m")
        if min_dist < SAC_RANGE:
            logging.warning("Some clusters now < SAC_RANGE apart (accept & report or merge again).")
        else:
            logging.info("[OK] All centroids >= SAC_RANGE.")

        
        ##   visualizing Clusters   ##

        def PlotClstrSummary(g, k):
            fig, axs = plt.subplots(1,3, figsize=(15,5))
            g.plot(ax=axs[0], column="cluster", cmap="tab20", markersize=25)
            axs[0].set_title(f"Map (k={k})"); axs[0].axis("off")

            stats = Get_Cluster_Stats(g)
            sns.histplot(stats["ratio"], bins=10, ax=axs[1])
            axs[1].set_title("Flood ratio per cluster")

            stats["count"].plot(kind="bar", ax=axs[2])
            axs[2].set_title("Cluster sizes"); axs[2].set_ylabel("points")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR/"cluster_quality_summary.png", dpi=150)
            plt.close()

        # Use post-merge cluster 
        n_clusters_balanced = int(gdf_balanced["cluster"].nunique())
        PlotClstrSummary(gdf_balanced, n_clusters_balanced)

        # Cluster Visualization and Summary

        fig, ax = plt.subplots(figsize=(9, 9))

        # plot flooded and non-flooded 
        gdf_balanced[gdf_balanced[TARGET] == 0].plot(

            ax=ax, column="cluster", cmap="tab20",
            marker="o", markersize=35, alpha=0.7, label="Non-flood")

        gdf_balanced[gdf_balanced[TARGET] == 1].plot(

            ax=ax, column="cluster", cmap="tab20",
            marker="^", markersize=45, alpha=0.9, label="Flood")

        ax.set_title("Clusters (color) with Flood / Non-flood Classes (shape)")
        ax.set_xlabel("Easting (UTM)")
        ax.set_ylabel("Northing (UTM)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR/"advanced_spatial_clusters.png", dpi=150)
        plt.close()

        logging.info(f"Final number of balanced clusters: {n_clusters_balanced}")

        summary = (gdf_balanced.groupby("cluster")[TARGET]
                      .agg(total="count", flooded="sum"))
        summary["non_flood"]  = summary["total"] - summary["flooded"]
        summary["flood_ratio"] = summary["flooded"] / summary["total"]

        logging.info(f"\nCluster Summary Statistics (after merging, k={n_clusters_balanced}):")
        logging.info("\n%s", summary.head(10))
       

        ##   Converting clusters to groups for cross-validation  ##
        groups = gdf_balanced["cluster"].values
        groups = pd.Series(groups)

        kmeans = kmeans_best
        
        logging.info(f"\nCreated {n_clusters_balanced} spatial clusters (after merging) with SAC-informed parameters")
        logging.info(f"GroupKFold(5) will give ~{n_clusters_balanced//5} clusters/fold (~{len(df)//5} pts)")
        logging.info(f"Minimum centroid separation enforced: {SAC_RANGE}m")
        logging.info(f"Minimum points per cluster enforced: {MIN_PTS}")
        
        # Store clustering results for later use
        advanced_clustering_success = True
        final_k = int(n_clusters_balanced)

    except Exception as e:
        logging.error(f"Error in advanced spatial clustering: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        raise
else:
    raise RuntimeError("latitude/longitude not found or spatial libraries unavailable; cannot proceed without fallbacks.")