import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import types as T
import pandas as pd

session = get_active_session()

prompt = """
Analyze the provided image of a bear. Describe only the facial profile of the bear. 
The response must be one of the following two values as a single word with no explanation:
- Dished (Concave profile, where the bridge of the nose dips)
- Straight (Flat profile, with no dip from the forehead to the nose)
"""
#nrows = 2
staged_file = session.sql(f"list @SNOWFLAKE_LEARNING_DB.PUBLIC.input_stage").collect()
#print(staged_file)
images_files = [ row["name"] for row in  staged_file 
               if row["name"].lower().endswith((".png",".jpg",".jpeg",".gif"))
               
               ]
st.success(f"Found{len(images_files)} images")
results_list = []
with st.spinner("Analyzing Facial profiles"):
    for image_path in images_files:
        image_name = image_path.split('/')[-1]

        id_value = image_name.rsplit('.',1)[0]

        query =f"""
        select AI_COMPLETE('claude-3-5-sonnet', '{prompt}',
        TO_FILE('@SNOWFLAKE_LEARNING_DB.PUBLIC.input_stage','{image_name}')
        );
        """
        result = session.sql(query).collect()
        results_list.append((id_value, result[0][0]))
        #print(result[0][0])

schema = T.StructType([
    
            T.StructField("ID", T.StringType()),
            T.StructField("FACIAL_PROFILE", T.StringType())
    
        ])
df_sp = session.create_dataframe(results_list, schema=schema)

df_facial_profile = pd.DataFrame(df_sp.to_pandas())

df_facial_profile

