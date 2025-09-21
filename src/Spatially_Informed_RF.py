# This module contains the code for following stages :
# Stage 3- Hybrid Spatially Informed Random Forest (nested_cross validation)
# Stage 4- Visualization and Model Evaluation

import os
import sys 
import logging
import traceback 
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import GroupKFold, cross_val_predict, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from scipy import stats
import argparse
import matplotlib.pyplot as plt, matplotlib.colors as mcolors


DEBUG = True
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
SPATIAL_ANALYSIS_AVAILABLE = True

# Data config
DATA_DIR = None
for i, arg in enumerate(sys.argv[1:]):
    if arg.startswith('--data-dir'):
        DATA_DIR = arg.split('=', 1)[1] if '=' in arg else sys.argv[i + 2]
        break
if DATA_DIR is None:
    DATA_DIR = os.getenv('DATA_DIR') or 'C:/Users/ASUS/Desktop/Spatially informed  ML/Ahwz_flood/data/'

os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, 'flood data.csv')
logging.info(f"Data directory: {os.path.abspath(DATA_DIR)}")

# output directories
PLOTS_DIR = Path("plots")
MODELS_DIR = Path("models")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

##       Data preprocessing      ##
try:
    df = pd.read_csv(DATA_FILE)
    logging.info(f"Dataset loaded: {len(df)} samples, {df.shape[1]} features")

    # Convert categorical features and handle missing data
    if 'LandCover' in df.columns:
        df['LandCover'] = df['LandCover'].astype('category')

    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.5].index
    if not cols_to_drop.empty:
        df.drop(columns=cols_to_drop, inplace=True)

    TARGET = "Label"
    if df[TARGET].isnull().sum() > 0:
        logging.error("Target column contains missing values - cannot proceed")
        exit(1)

    num_cols = df.select_dtypes(include='number').columns
    if TARGET in num_cols:
        num_cols = num_cols.drop(TARGET)
    cat_cols = df.select_dtypes(exclude='number').columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    # Remove duplicates and cap outliers
    df = df.drop_duplicates()
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    # One-hot encode categorical features
    if 'LandCover' in df.columns:
        df = pd.get_dummies(df, columns=['LandCover'], prefix='LC', drop_first=False)

    # Prepare target and features
    if TARGET not in df.columns:
        logging.error(f"Target column '{TARGET}' not found")
        exit(1)

    y = df[TARGET].values
    X = df.drop(columns=[TARGET])

    coords_to_drop = [col for col in ['latitude', 'longitude'] if col in X.columns]
    if coords_to_drop:
        X = X.drop(columns=coords_to_drop)

except Exception as e:
    logging.error(f"Data preprocessing failed: {e}")
    exit(1)


##       SAC Analysis Integration      ##

logging.info("\n=== SAC Analysis Integration ===")

# Import spatial autocorrelation and clustering results from SAC_Analysis.py
try:
    from SAC_Analysis import (
        sac_analysis_results,
        advanced_clustering_success,
        final_k,
        groups,
        sac_cutoff,
    )

    if not advanced_clustering_success:
        raise ValueError("Advanced spatial clustering did not complete successfully.")
    if groups is None or len(groups) == 0:
        raise ValueError("SAC_Analysis did not produce spatial groups.")
    if len(groups) != len(df):
        raise ValueError(
            f"Group count mismatch: SAC_Analysis produced {len(groups)} groups but dataset has {len(df)} rows. "
            "Ensure both modules use the same cleaned dataset."
        )

    logging.info(f"SAC-informed clustering ready: {final_k} clusters, cutoff {sac_cutoff}m")
except Exception as e:
    logging.error(f"SAC analysis integration failed: {e}")
    logging.error("Run SAC_Analysis.py first to generate spatial clusters.")
    sys.exit(1)

# Validate clustering results and show summary statistics
if groups is not None and len(groups) > 0:
    cluster_summary = groups.value_counts()
    logging.info(f"Cluster distribution: {len(cluster_summary)} groups, avg size {cluster_summary.mean():.1f}")


##       Stage 3-Hybrid Spatially Informed Random Forest      ##

df_analysis = df.copy()
df_analysis["true"] = y

# Prepare features for modeling
cat_cols = [col for col in X.columns if col.startswith('LC_')] if any(col.startswith('LC_') for col in X.columns) else []
pre = ColumnTransformer([("all", "passthrough", list(X.columns))])

# Configure Random Forest with balanced class weighting and spatial-aware hyperparameters
rf = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42
)
base_model = Pipeline([("pre", pre), ("rf", rf)])

# Hyperparameter search space for nested cross-validation
param_dist = {
    "rf__n_estimators": stats.randint(200, 800),
    "rf__max_depth": [None, 10, 20, 30, 50],
    "rf__min_samples_leaf": stats.randint(1, 8),
    "rf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
}

##    Nested cross-validation      ##
logging.info("\n=== Nested Cross-Validation ===")

# Use spatial clusters for unbiased performance estimation
outer_cv = GroupKFold(n_splits=5)
outer_scores = []
outer_probs = np.zeros_like(y, dtype=float)
outer_raw_probs = np.zeros_like(y, dtype=float)
outer_preds = np.zeros_like(y, dtype=int)
fold = 0

try:
    for train_idx, test_idx in outer_cv.split(X, y, groups=groups):
        fold += 1
        print(f"\nOuter fold {fold}/5 (evaluating on {len(test_idx)} samples)...")

        # Split data
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        groups_train_outer = groups.iloc[train_idx] if isinstance(groups, pd.Series) else groups[train_idx]

        # Create a fresh pipeline for this fold
        fold_pipeline = Pipeline([
            ("pre", pre),
            ("rf", RandomForestClassifier(
                class_weight='balanced',
                random_state=42
            ))
        ])

        # Inner CV for hyperparameter tuning
        inner_cv = GroupKFold(n_splits=4)
        try:
            inner_search = RandomizedSearchCV(
                fold_pipeline,
                param_distributions=param_dist,
                n_iter=20,
                scoring="balanced_accuracy",
                cv=inner_cv,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )

            # Fit using spatial groups
            inner_search.fit(X_train_outer, y_train_outer, groups=groups_train_outer)

            print(f"  Best parameters: {inner_search.best_params_}")
            print(f"  Inner CV score: {inner_search.best_score_:.3f}")

            # Get the best model for this fold
            best_model_fold = inner_search.best_estimator_

            # Refit the model on all training data for this fold
            best_model_fold.fit(X_train_outer, y_train_outer)

            # Get raw probabilities for test set (for analysis)
            test_probs = best_model_fold.predict_proba(X_test_outer)[:, 1]
            outer_raw_probs[test_idx] = test_probs

            # Apply probability calibration using cross-validation to avoid data leakage
            cal_cv = GroupKFold(n_splits=4)
            cal_probs_all = np.zeros_like(y_train_outer, dtype=float)
            y_cal_all = np.copy(y_train_outer)

            # Collect predictions from all folds for calibration
            for cal_train_idx, cal_test_idx in cal_cv.split(X_train_outer, y_train_outer, groups=groups_train_outer):
                X_cal_train, X_cal_test = X_train_outer.iloc[cal_train_idx], X_train_outer.iloc[cal_test_idx]
                y_cal_train = y_train_outer[cal_train_idx]

                cal_model = clone(best_model_fold)
                cal_model.fit(X_cal_train, y_cal_train)
                cal_probs_all[cal_test_idx] = cal_model.predict_proba(X_cal_test)[:, 1]

            # Fit final calibrator on collected predictions
            final_calibrator = IsotonicRegression(out_of_bounds='clip')
            final_calibrator.fit(cal_probs_all, y_cal_all)

            # Calibrate test probabilities and store
            calibrated_test_probs = final_calibrator.predict(test_probs)
            outer_probs[test_idx] = calibrated_test_probs

            # Calculate fold metrics
            fold_auc = roc_auc_score(y_test_outer, calibrated_test_probs)
            outer_scores.append(fold_auc)
            print(f"  Fold ROC-AUC: {fold_auc:.3f}")

        except Exception as e:
            print(f"  Error in fold {fold}: {str(e)}")
            if DEBUG:
                traceback.print_exc()
            print("  Using default model for this fold")

            
            default_model = Pipeline([
                ("pre", pre),
                ("rf", RandomForestClassifier(
                    n_estimators=500,
                    class_weight='balanced',
                    random_state=42
                ))
            ])
            default_model.fit(X_train_outer, y_train_outer)

            # Evaluate
            test_probs = default_model.predict_proba(X_test_outer)[:, 1]
            outer_probs[test_idx] = test_probs
            fallback_fold_auc = roc_auc_score(y_test_outer, test_probs)
            outer_scores.append(fallback_fold_auc)
            print(f"  Fallback model ROC-AUC: {fallback_fold_auc:.3f}")

    # Calculate overall metrics
    nested_auc = roc_auc_score(y, outer_probs)

    # Find optimal threshold (F1-maximizing)
    prec, rec, pr_thresholds = precision_recall_curve(y, outer_probs)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_f1_idx = np.argmax(f1s)
    optimal_threshold = pr_thresholds[best_f1_idx]
    print(f"Optimal threshold (F1): {optimal_threshold:.3f}, F1: {f1s[best_f1_idx]:.3f}")

    # Save threshold alongside other artifacts
    import joblib
    joblib.dump(optimal_threshold, str(MODELS_DIR / "flood_model_final_threshold.joblib"))

    # Create predictions with optimal threshold
    outer_preds = (outer_probs >= optimal_threshold).astype(int)
    nested_balanced_acc = balanced_accuracy_score(y, outer_preds)

    print("\n=== Nested CV Results ===")
    print(f"Mean ROC-AUC across folds: {np.mean(outer_scores):.3f} (std: {np.std(outer_scores):.3f})")
    print(f"Overall ROC-AUC: {nested_auc:.3f}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Balanced accuracy @ optimal threshold: {nested_balanced_acc:.3f}")
    print("\nClassification Report (Nested CV, optimal threshold):")
    print(classification_report(y, outer_preds, digits=3))

except Exception as e:
    print(f"\nError in nested CV: {str(e)}")
    if DEBUG:
        traceback.print_exc()
    print("Falling back to simple cross-validation")

    # Simple cross-validation as fallback
    cv = GroupKFold(n_splits=5)
    outer_probs = cross_val_predict(base_model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    outer_preds = cross_val_predict(base_model, X, y, cv=cv, groups=groups, method="predict")

    # Calculate metrics
    cv_auc = roc_auc_score(y, outer_probs)
    cv_acc = balanced_accuracy_score(y, outer_preds)

    # Find optimal threshold (F1-maximizing)
    prec, rec, pr_thresholds = precision_recall_curve(y, outer_probs)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_f1_idx = np.argmax(f1s)
    optimal_threshold = pr_thresholds[best_f1_idx]
    print(f"Optimal threshold (F1): {optimal_threshold:.3f}, F1: {f1s[best_f1_idx]:.3f}")
    import joblib
    joblib.dump(optimal_threshold, str(MODELS_DIR / "flood_model_final_threshold.joblib"))

    # Create predictions with optimal threshold
    outer_preds_opt = (outer_probs >= optimal_threshold).astype(int)

    print("\n=== Cross-Validation Results ===")
    print(f"ROC-AUC: {cv_auc:.3f}")
    print(f"Balanced accuracy (default threshold): {cv_acc:.3f}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print("\nClassification Report (default threshold):")
    print(classification_report(y, outer_preds, digits=3))
    print("\nClassification Report (optimal threshold):")
    print(classification_report(y, outer_preds_opt, digits=3))

    # Set variables for later use
    nested_auc = cv_auc
    optimal_threshold = optimal_threshold

    # Add predictions with optimal threshold
    outer_preds = outer_preds_opt

# Store results in analysis dataframe
df_analysis["prob"] = outer_probs
df_analysis["pred"] = outer_preds
df_analysis["err"] = df_analysis["pred"] != df_analysis["true"]
df_analysis["raw_prob"] = outer_raw_probs

##       Final model training      ##
logging.info("\n=== Training Final Model ===") 

try:
    final_model = RandomizedSearchCV(
        base_model, param_dist, n_iter=50, scoring="balanced_accuracy",
        cv=GroupKFold(n_splits=5), n_jobs=-1, random_state=42, verbose=1
    ).fit(X, y, groups=groups).best_estimator_

    # Save trained model and calibrator
    import joblib
    joblib.dump(final_model, str(MODELS_DIR / "flood_model_final.joblib"))

    cal_probs = np.zeros_like(y, dtype=float)
    for cal_train_idx, cal_test_idx in GroupKFold(n_splits=5).split(X, y, groups=groups):
        cal_model = clone(final_model)
        cal_model.fit(X.iloc[cal_train_idx], y[cal_train_idx])
        cal_probs[cal_test_idx] = cal_model.predict_proba(X.iloc[cal_test_idx])[:, 1]

    final_calibrator = IsotonicRegression(out_of_bounds='clip').fit(cal_probs, y)
    joblib.dump(final_calibrator, str(MODELS_DIR / "flood_model_final_calibrator.joblib"))

    # Save optimal threshold
    joblib.dump(optimal_threshold, str(MODELS_DIR / "flood_model_final_threshold.joblib"))
    logging.info(f"Optimal threshold saved: {optimal_threshold}")

    ##  Feature importance analysis  ##
    perm = permutation_importance(final_model, X, y, n_repeats=10, random_state=42, scoring="balanced_accuracy", n_jobs=-1)
    feature_names = np.array(X.columns)

    lc_indices = [i for i, name in enumerate(feature_names) if name.startswith('LC_')]
    if lc_indices:
        # Combine LandCover feature importances
        lc_importance = np.sum(perm.importances_mean[lc_indices])
        lc_std = np.sqrt(np.sum(np.square(perm.importances_std[lc_indices])))

        non_lc_indices = [i for i in range(len(feature_names)) if i not in lc_indices]
        combined_names = np.append(feature_names[non_lc_indices], "LandCover (Combined)")
        combined_means = np.append(perm.importances_mean[non_lc_indices], lc_importance)
        combined_stds = np.append(perm.importances_std[non_lc_indices], lc_std)

        top_features = np.argsort(combined_means)[::-1][:min(10, len(combined_names))]
        print("\n=== Top Feature Importances ===")
        for i in top_features:
            print(f"{combined_names[i]}: {combined_means[i]:.4f} ± {combined_stds[i]:.4f}")

except Exception as e:
    logging.error(f"Final model training failed: {e}")
    final_model = None

##      Stage 4- Visualization and Model Evaluation      ##

# Generate performance plots (confusion matrix and ROC curve)
try: 
    #confusion matrix 
    cm = confusion_matrix(y, df_analysis["pred"])
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("(b)", fontweight='bold', fontsize=18)
    plt.colorbar()

    for i, j in np.ndindex(cm.shape):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                color=color, fontsize=14)

    plt.xticks([0, 1], ["Not Flooded (0)", "Flooded (1)"], fontsize=12)
    plt.yticks([0, 1], ["Not Flooded (0)", "Flooded (1)"], fontsize=12)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"confusion_matrix.png", dpi=300)
    plt.close()

    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, outer_probs)
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {nested_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
             label=f'Optimal threshold ({optimal_threshold:.3f})')
    plt.xlim([0.0, 1.0]), plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('(a)', fontweight='bold', fontsize=22)
    plt.legend(loc="lower right", prop={'size': 14})
    plt.grid(True)
    plt.savefig(PLOTS_DIR/"roc_curve.png", dpi=300)
    plt.close()

except Exception as e:
    logging.error(f"Visualization creation failed: {e}")

##       Model calibration and performance metrics      ##
logging.info("\n=== Model Calibration Analysis ===")

y_true, y_prob = df_analysis["true"], np.clip(df_analysis["prob"], 1e-15, 1-1e-15)

# Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
def ece_mce(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bin_edges[1:-1], right=True)
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        mask = binids == i
        if mask.any():
            acc, conf = y_true[mask].mean(), y_prob[mask].mean()
            gap = abs(acc - conf)
            ece += gap * mask.mean()
            mce = max(mce, gap)
    return ece, mce

ece, mce = ece_mce(y_true.values, y_prob.values)
logging.info(f"Calibration: ECE={ece:.4f}, MCE={mce:.4f}")

# Overall predictive performance metrics
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
brier = brier_score_loss(y_true, y_prob)
logloss = log_loss(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)

logging.info(f"Performance: Brier={brier:.4f}, LogLoss={logloss:.4f}, AUC={auc:.4f}")

##   Performance Summary   ##
logging.info("\n=== Performance Summary ===")

metrics = {
    "ROC-AUC": nested_auc,
    "Accuracy": accuracy_score(y, df_analysis["pred"]),
    "Balanced Accuracy": balanced_accuracy_score(y, df_analysis["pred"]),
    "F1 Score": f1_score(y, df_analysis["pred"]),
    "Optimal Threshold": optimal_threshold
}

print("\nFinal Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")

## Save comprehensive results report
with open("Results.txt", "w") as f:
    f.write("Flood Hazard Prediction Model Results\n")
    f.write("="*50 + "\n\n")

    # Model configuration and performance
    f.write("Model Configuration:\n")
    if final_model is not None:
        f.write(f"Best hyperparameters: {final_model.get_params()}\n")
    f.write("\nPerformance Metrics:\n")
    for metric, value in metrics.items():
        f.write(f"{metric}: {value:.3f}\n")
    f.write(f"\nClassification Report:\n{classification_report(y, df_analysis['pred'], digits=3)}")

    # Feature importance summary
    if perm is not None:
        f.write("\nFeature Importance (Top 5):\n")
        feature_names = np.array(X.columns)
        top_features = np.argsort(perm.importances_mean)[::-1][:5]
        for i in top_features:
                f.write(f"{feature_names[i]}: {perm.importances_mean[i]:.4f} ± {perm.importances_std[i]:.4f}\n")

    # SAC clustering summary
    if 'sac_analysis_results' in locals() and sac_analysis_results:
        f.write(f"\nSAC Analysis: Cutoff {sac_analysis_results.get('cutoff_distance', 'N/A')}m")
        f.write(f" | Clustering: {final_k} spatial groups\n")

logging.info("Results saved to 'Results.txt'")
logging.info("SPATIALLY INFORMED FLOOD MODELING COMPLETE")


##       Hazard mapping      ##
logging.info("\n=== Hazard Mapping ===")

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

# function to get calibrated prediction
def get_calibrated_predictions(model, final_calibrator, X, gdf, optimal_threshold):

    zero_features = [col for col in X.columns if (X[col] == 0).all()]
    for feature in zero_features:
        X[feature] = 1.0

    # Get and calibrate probabilities
    probas = model.predict_proba(X)[:, 1]
    calibrated_probas = final_calibrator.predict(probas) if final_calibrator else probas
    preds = (calibrated_probas >= optimal_threshold).astype(int)

    # Add predictions to geodataframe
    gdf = gdf.copy()  # Avoid modifying original
    gdf['flood_prob'] = calibrated_probas
    gdf['flood_pred'] = preds
    gdf['Label'] = preds

    return gdf

# function to create hazard maps 
def create_hazard_maps(gdf, output_dir='hazard_maps', model_name='flood_model'):

    os.makedirs(output_dir, exist_ok=True)

    # Define color schemes
    prob_cmap = plt.cm.Reds
    binary_cmap = mcolors.ListedColormap(['lightblue', 'darkred'])

    # Create probability map
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(column='flood_prob', ax=ax, cmap=prob_cmap,
             legend=True, alpha=0.7, edgecolor='black', linewidth=0.3)

    if HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        except:
            pass

    ax.set_title(f'{model_name} - Flood Probability Map')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/flood_probability_map.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create binary prediction map
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(column='flood_pred', ax=ax, cmap=binary_cmap,
             legend=True, alpha=0.7, edgecolor='black', linewidth=0.3)

    if HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        except:
            pass

    ax.set_title(f'{model_name} - Flood Hazard Zones')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/flood_hazard_zones.png', dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Hazard maps saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Flood Hazard Prediction and Mapping')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                       help='Mode: train (default) or predict')
    parser.add_argument('--input', help='Input shapefile for prediction mode')
    parser.add_argument('--output', default='hazard_maps',
                       help='Output directory for maps (default: hazard_maps)')
    parser.add_argument('--model', default=str(Path('models') / 'flood_model_final.joblib'),
                       help='Path to trained model file')
    parser.add_argument('--data-dir', help='Data directory path')

    try:
        args = parser.parse_args()
    except SystemExit:
        args = None

    if args and args.mode == 'predict':
        if not args.input:
            print("Usage: python script.py --mode predict --input shapefile.shp")
            sys.exit(1)

        # Load trained model for prediction
        import joblib
        import geopandas as gpd

        try:
            model_path = os.path.abspath(args.model)
            logging.info(f"Loading model from: {model_path}")

            if not os.path.exists(model_path):
                logging.error(f"Model file not found: {model_path}")
                sys.exit(1)

            final_model = joblib.load(model_path)
            logging.info("Model loaded successfully")

            # Load calibrator and threshold
            try:
                final_calibrator = joblib.load(model_path.replace('.joblib', '_calibrator.joblib'))
                logging.info("Calibrator loaded successfully")
            except:
                final_calibrator = None
                logging.warning("No calibrator found, using raw probabilities")

            try:
                optimal_threshold = joblib.load(model_path.replace('.joblib', '_threshold.joblib'))
                logging.info(f"Optimal threshold loaded: {optimal_threshold}")
            except:
                optimal_threshold = 0.5
                logging.warning("No optimal threshold found, using default 0.5")

            # Load input shapefile
            if not os.path.exists(args.input):
                logging.error(f"Input shapefile not found: {args.input}")
                sys.exit(1)

            logging.info(f"Loading shapefile: {args.input}")
            gdf = gpd.read_file(args.input)

            # Prepare features for prediction 
            logging.info("Converting LandCover to one-hot encoded features...")

            # Convert LandCover to one-hot encoded features (LC_1, LC_2, etc.)
            if 'LandCover' in gdf.columns:
                unique_categories = gdf['LandCover'].unique()
                logging.info(f"Found {len(unique_categories)} unique land cover categories: {unique_categories}")

                # Create one-hot encoded columns for all 13 expected categories
                expected_lc_features = [f'LC_{i}' for i in range(1, 14)]  # LC_1 to LC_13

                for lc_feature in expected_lc_features:
                    gdf[lc_feature] = 0 

                # Set the appropriate LC feature to 1 based on actual land cover value
                for idx, row in gdf.iterrows():
                    lc_value = row['LandCover']
                    
                    if lc_value in unique_categories:
                        lc_index = list(unique_categories).index(lc_value) + 1
                        lc_col = f'LC_{lc_index}'
                        if lc_col in gdf.columns:
                            gdf.at[idx, lc_col] = 1

                gdf = gdf.drop('LandCover', axis=1)
                logging.info(f"Converted LandCover to {len(expected_lc_features)} LC features")

            # Get all features for prediction
            feature_cols = [col for col in gdf.columns
                           if col not in ['geometry', 'Label', 'flood_prob', 'flood_pred']]

            if len(feature_cols) == 0:
                logging.error("No feature columns found in shapefile")
                sys.exit(1)

            X_pred = gdf[feature_cols].fillna(gdf[feature_cols].median())

            # Generate predictions
            logging.info("Generating predictions...")
            gdf_pred = get_calibrated_predictions(final_model, final_calibrator,
                                                X_pred, gdf, optimal_threshold)

            # Create hazard maps
            logging.info("Creating hazard maps...")
            create_hazard_maps(gdf_pred, args.output, "Flood Hazard Model")

            logging.info(f"Prediction completed successfully!")
            logging.info(f"Results saved to: {os.path.abspath(args.output)}")

            # Save prediction results
            output_shp = f"{args.output}/flood_predictions.shp"
            gdf_pred.to_file(output_shp)
            logging.info(f"Prediction shapefile saved: {output_shp}")

        except Exception as e:
            logging.error(f"Error in prediction mode: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    else:
        logging.info("Training mode completed. Use --mode predict --input shapefile.shp for mapping.")