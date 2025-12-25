import pandas as pd
import numpy as np
import re
import itertools
import warnings
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef
from snowflake.snowpark.context import get_active_session
from snowflake.ml.experiment.experiment_tracking import ExperimentTracking
from snowflake.ml.registry import Registry

st.title("🐻 Bear Species Classification - CLEAN Training")
warnings.filterwarnings("ignore")

# =========================
# 1️⃣ Snowflake Session
# =========================
session = get_active_session()
st.write("✅ Connected to Snowflake!")

# =========================
# 2️⃣ Load & CLEAN Dataset
# =========================
st.subheader("📊 Step 1: Loading & Cleaning Data")

bear_df = session.table("BEAR").to_pandas()
st.write(f"Original dataset shape: {bear_df.shape}")

# Convert numeric columns
numeric_cols = ['body_mass_kg', 'shoulder_hump_height_cm', 'claw_length_cm',
                'snout_length_cm', 'forearm_circumference_cm', 'ear_length_cm']

for col in numeric_cols:
    bear_df[col] = pd.to_numeric(bear_df[col], errors='coerce')

# CLEAN CATEGORICAL COLUMNS - Remove garbage values
def clean_categorical_value(val):
    """Filter out garbage values"""
    if pd.isna(val):
        return None
    
    val_str = str(val).strip().replace('"', '')
    
    # Filter criteria
    if len(val_str) > 30:
        return None
    if any(char in val_str for char in ['«', '›', '非', 'ム', '\n', '<<<', '>>>']):
        return None
    
    return val_str

# Clean each categorical column
categorical_cols = ['COLOR', 'FACIAL_PROFILE', 'PAW_PAD_TEXTURE']

st.write("🧹 Cleaning categorical columns...")
for col in categorical_cols:
    before_count = len(bear_df)
    bear_df[col] = bear_df[col].apply(clean_categorical_value)
    bear_df = bear_df.dropna(subset=[col])
    after_count = len(bear_df)
    removed = before_count - after_count
    st.write(f"- {col}: Removed {removed} garbage rows")

st.success(f"✅ Clean dataset shape: {bear_df.shape}")

# Show unique values after cleaning
st.write("**Clean categorical values:**")
for col in categorical_cols:
    unique_vals = sorted(bear_df[col].unique().tolist())
    st.write(f"- {col}: {unique_vals}")

# =========================
# 3️⃣ Features & Target
# =========================
X = bear_df.drop(columns=['species', 'ID'])
y = bear_df['species']

# =========================
# 4️⃣ Train/Test Split
# =========================
st.subheader("🔀 Step 2: Train/Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
st.write(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# =========================
# 5️⃣ Column Cleaning
# =========================
st.subheader("🧼 Step 3: Column Name Cleaning")

def clean_column(col):
    col = col.lower()
    col = re.sub(r'[^a-z0-9_]', '_', col)
    col = re.sub(r'_+', '_', col)
    col = col.strip('_')
    return col[:30]

X_train.columns = [clean_column(c) for c in X_train.columns]
X_test.columns = [clean_column(c) for c in X_test.columns]

st.write(f"Cleaned column names: {list(X_train.columns)}")

# =========================
# 6️⃣ Feature Scaling & One-Hot Encoding
# =========================
st.subheader("⚙️ Step 4: Feature Engineering")

numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

st.write(f"Numerical features: {list(numerical_features)}")
st.write(f"Categorical features: {list(categorical_features)}")

# Scale numeric features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_features])
X_test_num = scaler.transform(X_test[numerical_features])

# One-hot encode categorical features
onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat = onehot.fit_transform(X_train[categorical_features])
X_test_cat = onehot.transform(X_test[categorical_features])

# Clean one-hot column names
cat_feature_names = []
for f, cats in zip(categorical_features, onehot.categories_):
    for c in cats:
        name = f"{f}_{c}".lower()
        name = re.sub(r'[^a-z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        name = name[:30]
        cat_feature_names.append(name)

st.write(f"One-hot encoded columns: {cat_feature_names}")

# Combine features
all_features = list(numerical_features) + cat_feature_names
X_train_scaled = pd.DataFrame(np.hstack([X_train_num, X_train_cat]),
                              columns=all_features, index=X_train.index)
X_test_scaled = pd.DataFrame(np.hstack([X_test_num, X_test_cat]),
                             columns=all_features, index=X_test.index)

st.success(f"✅ Feature engineering done! Training shape: {X_train_scaled.shape}")

# =========================
# 7️⃣ Encode Target
# =========================
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

st.write(f"Species encoding: {dict(enumerate(le.classes_))}")

# =========================
# 8️⃣ Save Test Data to Snowflake
# =========================
st.subheader("💾 Step 5: Saving Test Data")

test_df = X_test_scaled.copy()
test_df['ACTUAL_SPECIES'] = y_test_encoded
test_df = test_df.reset_index(drop=True)

snowpark_df = session.create_dataframe(test_df)
snowpark_df.write.mode("overwrite").save_as_table(
    "SNOWFLAKE_LEARNING_DB.PUBLIC.BEAR_TEST_DATA_CLEAN"
)
st.success(f"✅ Test data saved! Rows: {len(test_df)}, Columns: {len(test_df.columns)}")

# =========================
# 9️⃣ Experiment Tracking
# =========================
st.subheader("🧪 Step 6: Experiment Tracking")

exp = ExperimentTracking(session=session)
exp.set_experiment("Bear_Classification_Clean")
st.success("✅ Experiment tracking initialized")

# =========================
# 🔟 Hyperparameter Tuning
# =========================
st.subheader("🎯 Step 7: Hyperparameter Tuning")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2"]
}

results = []
param_combinations = [dict(zip(param_grid.keys(), v)) 
                     for v in itertools.product(*param_grid.values())]

progress_bar = st.progress(0)
status_text = st.empty()

for idx, p in enumerate(param_combinations):
    p['random_state'] = 42
    run_name = f"RF_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    status_text.text(f"Training model {idx+1}/{len(param_combinations)}...")
    
    with exp.start_run(run_name):
        exp.log_params(p)
        model = RandomForestClassifier(**p)
        model.fit(X_train_scaled, y_train_encoded)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='macro')
        recall = recall_score(y_test_encoded, y_pred, average='macro')
        mcc = matthews_corrcoef(y_test_encoded, y_pred)
        
        exp.log_metric("accuracy", acc)
        exp.log_metric("precision", precision)
        exp.log_metric("recall", recall)
        exp.log_metric("mcc", mcc)
        
        results.append({**p, 'accuracy': acc, 'precision': precision, 
                       'recall': recall, 'mcc': mcc})
    
    progress_bar.progress((idx + 1) / len(param_combinations))

status_text.text("Training complete!")
st.success(f"✅ Trained {len(param_combinations)} models")

# =========================
# 1️⃣1️⃣ Best Model
# =========================
st.subheader("🏆 Step 8: Best Model Selection")

results_df = pd.DataFrame(results)
st.dataframe(results_df.sort_values('accuracy', ascending=False))

best_model = results_df.loc[results_df['accuracy'].idxmax()]

st.write("**Best Model Parameters:**")
st.json({
    "n_estimators": int(best_model['n_estimators']),
    "max_depth": int(best_model['max_depth']),
    "min_samples_leaf": int(best_model['min_samples_leaf']),
    "max_features": best_model['max_features']
})

st.write("**Best Model Metrics:**")
st.metric("Accuracy", f"{best_model['accuracy']:.4f}")
col1, col2, col3 = st.columns(3)
col1.metric("Precision", f"{best_model['precision']:.4f}")
col2.metric("Recall", f"{best_model['recall']:.4f}")
col3.metric("MCC", f"{best_model['mcc']:.4f}")

# Train final model
best_params = {
    "n_estimators": int(best_model['n_estimators']),
    "max_depth": int(best_model['max_depth']),
    "min_samples_leaf": int(best_model['min_samples_leaf']),
    "max_features": best_model['max_features'],
    "random_state": 42
}

final_model = RandomForestClassifier(**best_params)
final_model.fit(X_train_scaled, y_train_encoded)

# =========================
# 1️⃣2️⃣ Model Registry
# =========================
st.subheader("📦 Step 9: Model Registration")

registry = Registry(session)
model_name = "BEAR_SPECIES_CLASSIFIER_CLEAN"
model_version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Ensure columns <=30 chars
X_train_scaled_clean = X_train_scaled.copy()
X_train_scaled_clean.columns = [c[:30] for c in X_train_scaled_clean.columns]

model_ref = registry.log_model(
    model=final_model,
    model_name=model_name,
    version_name=model_version,
    sample_input_data=X_train_scaled_clean.head(5),
    metrics={
        "accuracy": float(best_model['accuracy']),
        "precision": float(best_model['precision']),
        "recall": float(best_model['recall']),
        "mcc": float(best_model['mcc'])
    },
    options={"case_sensitive": True},
    comment="Clean training - no garbage data"
)
st.success(f"✅ Model registered: {model_name} {model_version}")

# =========================
# 1️⃣3️⃣ Deploy as Service
# =========================
st.subheader("🚀 Step 10: Model Deployment")

with st.spinner("Deploying model service..."):
    session.sql("DROP SERVICE IF EXISTS bear_rf_classifier_clean").collect()
    model_ref.create_service(
        service_name="bear_rf_classifier_clean",
        service_compute_pool="system_compute_pool_cpu",
        ingress_enabled=True
    )

st.success("✅ Model service deployed as: **bear_rf_classifier_clean**")

# =========================
# 1️⃣4️⃣ Test Prediction
# =========================
st.subheader("🧪 Step 11: Test Prediction (via SERVICE)")

# Use same columns as training
input_df = X_test_scaled.head(3).copy()

# Create temp view
temp_view = "TEMP_BEAR_PREDICTION_VIEW_CLEAN"
session.create_dataframe(input_df).create_or_replace_temp_view(temp_view)

# Call SERVICE using SQL
query = f"""
SELECT
    bear_rf_classifier_clean!PREDICT(*) AS prediction
FROM {temp_view}
"""

result = session.sql(query).collect()

# Parse results
pred_ids = [json.loads(r["PREDICTION"])["output_feature_0"] for r in result]
pred_labels = le.inverse_transform(pred_ids)

# Show results
test_results = pd.DataFrame({
    "Predicted Species": pred_labels,
    "Actual Species": le.inverse_transform(y_test_encoded[:3])
})

st.dataframe(test_results)
