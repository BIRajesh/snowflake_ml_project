import streamlit as st
import json
import pandas as pd
import numpy as np
import time
import re
import uuid
from snowflake.snowpark.context import get_active_session

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="🐻 Bear Species Prediction App",
    layout="wide"
)

st.title("🐻 Bear Species Prediction App")
st.warning("Predict bear species based on physical characteristics using a deployed Snowflake model.")

# --------------------------------------------------
# SNOWFLAKE SESSION
# --------------------------------------------------
session = get_active_session()

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def clean_column(col):
    """Clean column names exactly like training"""
    col = col.lower()
    col = re.sub(r'[^a-z0-9_]', '_', col)
    col = re.sub(r'_+', '_', col)
    col = col.strip('_')
    return col[:30]

def clean_categorical_value(val):
    """Filter out garbage values"""
    if pd.isna(val):
        return None
    val_str = str(val).strip().replace('"', '')
    if len(val_str) > 30:
        return None
    if any(char in val_str for char in ['«', '›', '非', 'ム', '\n', '<<<', '>>>']):
        return None
    return val_str

# --------------------------------------------------
# MANUAL SCALER (No sklearn needed!)
# --------------------------------------------------
class ManualScaler:
    """Simple StandardScaler implementation"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        self.mean_ = X.mean()
        self.std_ = X.std()
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.std_

# --------------------------------------------------
# LOAD SERVICES
# --------------------------------------------------
services = session.sql("SHOW SERVICES").collect()
service_list = [s['name'] for s in services if 'bear' in s['name'].lower()]

if not service_list:
    st.error("❌ No BEAR model services found in Snowflake.")
    st.info("💡 Please run the training code first to deploy a model service.")
    st.stop()

# --------------------------------------------------
# LOAD TRAINING DATA & SETUP
# --------------------------------------------------
@st.cache_data
def load_training_setup():
    """Load data and prepare scaler"""
    try:
        # Load BEAR table
        bear_df = session.sql("SELECT * FROM SNOWFLAKE_LEARNING_DB.PUBLIC.BEAR").to_pandas()
        
        # Convert numeric columns
        numeric_cols = ['body_mass_kg', 'shoulder_hump_height_cm', 'claw_length_cm',
                        'snout_length_cm', 'forearm_circumference_cm', 'ear_length_cm']
        
        for col in numeric_cols:
            bear_df[col] = pd.to_numeric(bear_df[col], errors='coerce')
        
        # Clean categorical columns
        categorical_cols = ['COLOR', 'FACIAL_PROFILE', 'PAW_PAD_TEXTURE']
        
        for col in categorical_cols:
            bear_df[col] = bear_df[col].apply(clean_categorical_value)
            bear_df = bear_df.dropna(subset=[col])
        
        # Prepare features
        X = bear_df.drop(columns=['species', 'ID'])
        
        # Clean column names
        X.columns = [clean_column(c) for c in X.columns]
        
        # Get column types
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Fit manual scaler on all data
        scaler = ManualScaler()
        scaler.fit(X[numerical_features])
        
        return X, numerical_features, categorical_features, scaler
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

with st.spinner("Loading training data..."):
    bear_data, numerical_cols, categorical_cols, scaler = load_training_setup()

st.success(f"✅ Loaded {len(bear_data)} clean samples")

# --------------------------------------------------
# GET TRAINED COLUMNS
# --------------------------------------------------
@st.cache_data
def get_trained_columns():
    """Get exact column order from training - USE CLEAN VERSION"""
    try:
        # Use BEAR_TEST_DATA_CLEAN which has exactly 22 features
        test_data = session.sql(
            "SELECT * FROM SNOWFLAKE_LEARNING_DB.PUBLIC.BEAR_TEST_DATA_CLEAN LIMIT 0"
        ).to_pandas()
        
        feature_cols = [c for c in test_data.columns if c != 'ACTUAL_SPECIES']
        
        st.info(f"📋 Using BEAR_TEST_DATA_CLEAN with {len(feature_cols)} features")
        
        return feature_cols
        
    except Exception as e:
        # Fallback to original if clean doesn't exist
        try:
            test_data = session.sql(
                "SELECT * FROM SNOWFLAKE_LEARNING_DB.PUBLIC.BEAR_TEST_DATA LIMIT 0"
            ).to_pandas()
            feature_cols = [c for c in test_data.columns if c != 'ACTUAL_SPECIES']
            st.warning(f"⚠️ Using BEAR_TEST_DATA with {len(feature_cols)} features")
            return feature_cols
        except:
            st.error(f"❌ Error: No test data table found!")
            st.stop()

trained_columns = get_trained_columns()

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

# Force BEAR_RF_CLASSIFIER_CLEAN as default if available
default_service = 'BEAR_RF_CLASSIFIER_CLEAN' if 'BEAR_RF_CLASSIFIER_CLEAN' in service_list else service_list[0]

model_option = st.sidebar.selectbox(
    "Select a model service:",
    service_list,
    index=service_list.index(default_service) if default_service in service_list else 0,
    help="Choose the deployed model service to use for predictions"
)

# Show warning if not using CLEAN service
if model_option != 'BEAR_RF_CLASSIFIER_CLEAN':
    st.sidebar.warning("⚠️ Recommended: Use BEAR_RF_CLASSIFIER_CLEAN")

st.sidebar.header("📥 Input Parameters")
st.sidebar.subheader("Physical Measurements")

input_values = {}

# Numerical features
for col in numerical_cols:
    display_name = col.replace('_', ' ').title()
    input_values[col] = st.sidebar.slider(
        display_name,
        min_value=float(bear_data[col].min()),
        max_value=float(bear_data[col].max()),
        value=float(bear_data[col].mean()),
        step=0.1
    )

st.sidebar.subheader(":material/category: Categorical Features")

# Categorical features - ONLY show clean values, not garbage
for col in categorical_cols:
    display_name = col.replace('_', ' ').title()
    
    # Get ONLY clean values (no garbage prefixed with 'i_')
    all_unique = sorted(bear_data[col].unique().tolist())
    
    # Filter out garbage values that start with specific patterns
    clean_values = []
    for val in all_unique:
        val_clean = str(val).lower()
        # Skip garbage values
        if not any(pattern in val_clean for pattern in ['cosmic', 'basin', 'duitse', 'allem', 'i_', 'idt', 'nhuman']):
            clean_values.append(val)
    
    if not clean_values:
        # If all filtered out, use first value as fallback
        clean_values = [all_unique[0]]
    
    input_values[col] = st.sidebar.selectbox(display_name, clean_values)

# --------------------------------------------------
# MODEL STATUS
# --------------------------------------------------
st.subheader(":material/signal_wifi_statusbar_not_connected: Model Status")
st.success(f"✅ Service `{model_option}` is ready")
st.info(f"📊 Model expects {len(trained_columns)} features in specific order")

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------
if st.sidebar.button("🔮 Make Prediction", type="primary"):
    st.header("🎯 Prediction Results")
    start_time = time.time()
    
    with st.spinner("Making prediction..."):
        try:
            # Step 1: Scale numerical features
            numeric_input = pd.DataFrame([[input_values[col] for col in numerical_cols]], 
                                        columns=numerical_cols)
            scaled_numeric = scaler.transform(numeric_input)
            
            # Step 2: One-hot encode categorical features
            all_cat_columns = {}
            
            for cat_col in categorical_cols:
                selected_value = input_values[cat_col]
                
                # Find all one-hot columns for this categorical feature in trained_columns
                prefix = cat_col + '_'
                cat_onehot_cols = [c for c in trained_columns if c.startswith(prefix)]
                
                # Create one-hot encoding for ALL columns (including garbage from training)
                for onehot_col in cat_onehot_cols:
                    category_name = onehot_col[len(prefix):]
                    clean_selected = clean_column(selected_value)
                    
                    # Match if category name matches cleaned selected value
                    all_cat_columns[onehot_col] = 1.0 if category_name == clean_selected else 0.0
            
            # Step 3: Combine features in exact order
            prediction_row = {}
            
            # Convert scaled_numeric DataFrame to dict for easy access
            scaled_dict = {col: scaled_numeric[col].iloc[0] for col in scaled_numeric.columns}
            
            for col in trained_columns:
                if col in numerical_cols:
                    prediction_row[col] = float(scaled_dict[col])
                elif col in all_cat_columns:
                    prediction_row[col] = float(all_cat_columns[col])
                else:
                    prediction_row[col] = 0.0
            
            # Create DataFrame
            prediction_df = pd.DataFrame([prediction_row], columns=trained_columns)
            
            # DEBUG: Show what we're sending
            st.write("### 🔍 DEBUG INFO:")
            st.write(f"**Total columns in prediction:** {len(prediction_df.columns)}")
            st.write(f"**Expected by model:** 22")
            st.write(f"**Column types:** {prediction_df.dtypes.value_counts().to_dict()}")
            st.write(f"**Sample columns:**")
            st.dataframe(prediction_df.head())
            
            # Show column names
            with st.expander("View All Column Names"):
                st.write(list(prediction_df.columns))
            
            # Create temporary table
            unique_id = str(uuid.uuid4()).replace('-', '_')[:8]
            temp_table = f"SNOWFLAKE_LEARNING_DB.PUBLIC.BEAR_PRED_{unique_id}"
            
            # Save to Snowflake
            input_df = session.create_dataframe(prediction_df)
            input_df.write.mode("overwrite").save_as_table(temp_table)
            
            # Make prediction
            result = session.sql(
                f"SELECT {model_option} ! PREDICT(*) AS predicted_species FROM {temp_table}"
            ).collect()
            
            # Clean up
            session.sql(f"DROP TABLE IF EXISTS {temp_table}").collect()
            
            # Display result
            species_mapping = {
                0: 'American Black Bear (ABB)',
                1: 'Eurasian Brown Bear (EUR)',
                2: 'Grizzly Bear (GRZ)',
                3: 'Kodiak Bear (KDK)'
            }
            
            if result:
                predicted_species = json.loads(result[0]['PREDICTED_SPECIES'])
                species_id = int(predicted_species["output_feature_0"])
                
                st.write("### 🐻 Predicted Bear Species:")
                
                # Display with color coding
                if species_id in species_mapping:
                    st.success(f"**{species_mapping[species_id]}**")
                else:
                    st.warning(f"Unknown species ID: {species_id}")
                
                end_time = time.time()
                st.caption(f"⏱️ Prediction completed in {(end_time - start_time):.2f} seconds")
                
                # Show input summary
                with st.expander("📊 View Input Summary"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Physical Measurements:**")
                        for col in numerical_cols:
                            st.write(f"• {col.replace('_', ' ').title()}: {input_values[col]:.2f}")
                    
                    with col2:
                        st.write("**Categorical Features:**")
                        for col in categorical_cols:
                            st.write(f"• {col.replace('_', ' ').title()}: {input_values[col]}")
                
                # Debug info (optional)
                with st.expander("🔍 Technical Details"):
                    st.write(f"**Model Service:** {model_option}")
                    st.write(f"**Total Features:** {len(prediction_df.columns)}")
                    st.write(f"**Scaled Numerical Values (first 3):**")
                    st.write({col: f"{prediction_row[col]:.3f}" for col in numerical_cols[:3]})
            
        except Exception as e:
            st.error(f"❌ Error making prediction")
            
            with st.expander("📋 Error Details"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())
            
            st.info("""
            **Troubleshooting Tips:**
            1. Make sure the training code has completed successfully
            2. Check that the model service is running: `SHOW SERVICES;`
            3. Verify test data table exists: `SELECT * FROM BEAR_TEST_DATA_CLEAN LIMIT 1;`
            4. Ensure BEAR table has clean data (no excessive garbage values)
            """)
            
            # Cleanup on error
            try:
                session.sql(f"DROP TABLE IF EXISTS {temp_table}").collect()
            except:
                pass

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("🐻 Bear Species Prediction App | Powered by Snowflake ML")
st.caption(f"📊 Using {len(bear_data)} training samples | {len(trained_columns)} features")
