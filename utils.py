import sqlite3
import pandas as pd

def export_to_excel(db_path='face_logs.db', output_file='face_logs.xlsx'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM detections", conn)
    df.to_excel(output_file, index=False)
    conn.close()
