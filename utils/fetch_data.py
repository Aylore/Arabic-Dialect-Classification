import pandas as pd
import sqlite3
# import os


def get_data():

    conn = sqlite3.connect('./data/dialects_database.db')

    df_label = pd.read_sql_query("SELECT * FROM id_text", conn)

    df_target = pd.read_sql_query("SELECT * FROM id_dialect" , conn)

    df = pd.merge(df_label ,df_target , on="id")

    conn.close()

    return df



get_data()