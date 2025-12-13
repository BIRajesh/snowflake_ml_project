df['ID'] = df['ID'].str.upper()  # Ensure IDs are in uppercase
#df_fur
df_fur['ID'] = df_fur['ID'].str.upper()
#df_facial_profile
df_facial_profile['ID'] = df_facial_profile['ID'].str.upper()
df_paw_pad['ID'] = df_paw_pad['ID'].str.upper()
#df_paw_pad
# Perform sequential merges to combine all features using proper indexing
df_combined = df.merge(df_fur, on='ID', how='inner')
df_combined = df_combined.merge(df_facial_profile, on='ID', how='inner')
df_combined = df_combined.merge(df_paw_pad, on='ID', how='inner')

# Display the combined DataFrame df_fur
#df_combined
