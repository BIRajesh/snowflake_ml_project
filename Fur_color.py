import streamlit as st
import os
import tempfile
import pandas as pd
from snowflake.snowpark.context import get_active_session
session = get_active_session()
from snowflake.snowpark import types as T

#@st.cache_data (show_spinner=True)

#def analyze_fur_color_cached(image_files,prompt):
    #for images_path in image_files:
        #images = images_path.split('/')[-1]
    



prompt = """
Analyze the provided image of a bear. Describe only the fur color of the bear
by choosing the most appropriate term from the following list. The response
should be a single value.
- Light Brown
- Medium Brown
- Blond
- Dark Brown
- Grizzled (A mix of colors with silver-tipped hairs)
- Reddish Brown
- Blackish Brown
- Black
- Brown
- Cinnamon
"""
result_list =[]
#image_files =[]
staged_file = session.sql(f"list @SNOWFLAKE_LEARNING_DB.PUBLIC.input_stage").collect()
image_files =[ row['name'] for row in staged_file 
             if row['name'].lower().endswith((".png",".jpg",".jpeg"))
             ]
image_files = image_files
#print(image_files)
st.success(f"found {len(image_files)} images")
for images_path in image_files:
    images = images_path.split('/')[-1]
    #print(images)
    query = f"""
            select AI_COMPLETE('claude-3-5-sonnet','{prompt}',
            TO_FILE('@SNOWFLAKE_LEARNING_DB.PUBLIC.input_stage','{images}')
            );
    
    """
    result = session.sql(query).collect()
    #print(result[0][0])
    result_list.append((images.replace(".png",""),result[0][0]))
#print(result_list)

df_result = pd.DataFrame(result_list,columns=["id","fur_color"])
#st.dataframe(df_result)

schema = T.StructType([
    T.StructField("id", T.StringType()),
    T.StructField("color", T.StringType())
])
# Convert the results_list to a Snowpark DataFrame


df_results = session.create_dataframe(result_list, schema=schema)
df_fur = pd.DataFrame(df_results.to_pandas())
df_fur


