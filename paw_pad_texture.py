import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import types as T

session = get_active_session()

# -------------------------
# Prompt for Paw Pad Texture
# -------------------------
prompt = """
Analyze the provided image of a bear. Describe only the paw pad texture of the bear. 
The response must be one of the following:
- Smooth
- Rough

Return only one word with no explanation.
"""

# -------------------------
# Step 1: Load image files
# -------------------------
staged_files = session.sql(
    "LIST @SNOWFLAKE_LEARNING_DB.PUBLIC.input_stage"
).collect()

image_files = [
    row["name"]
    for row in staged_files
    if row["name"].lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
]

st.success(f"Found {len(image_files)} images")

# -------------------------
# Step 2: Run AI analysis
# -------------------------
results_list = []

with st.spinner("Analyzing paw pad texture…"):
    for image_path in image_files:
        image_name = image_path.split("/")[-1]  # extract file name only
        id_value = image_name.rsplit(".", 1)[0]  # remove extension safely

        query = f"""
            SELECT AI_COMPLETE(
                'claude-3-5-sonnet',
                $$ {prompt} $$,
                TO_FILE('@SNOWFLAKE_LEARNING_DB.PUBLIC.input_stage', '{image_name}')
            );
        """

        result = session.sql(query).collect()
        ai_output = result[0][0]  # AI text output

        results_list.append((id_value, ai_output))

# -------------------------
# Step 3: Build DataFrames
# -------------------------
schema = T.StructType([
    T.StructField("ID", T.StringType()),
    T.StructField("PAW_PAD_TEXTURE", T.StringType())
])

df_sp = session.create_dataframe(results_list, schema=schema)
df_paw_pad = pd.DataFrame(df_sp.to_pandas())
df_paw_pad

