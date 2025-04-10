import snowflake.connector
import streamlit as st

def test_snowflake_connection():
    try:
        # Connect using Streamlit secrets
        conn = snowflake.connector.connect(
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            account=st.secrets["snowflake"]["account"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            database=st.secrets["snowflake"]["database"],
            schema=st.secrets["snowflake"]["schema"]
        )
        
        # If connection is successful, return a success message
        st.success("Connection to Snowflake was successful!")
        
        # Optionally, run a test query
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_VERSION()")
        version = cursor.fetchone()
        st.write(f"Snowflake version: {version[0]}")
        
        # Clean up
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error connecting to Snowflake: {e}")

# Test connection on app startup
test_snowflake_connection()
