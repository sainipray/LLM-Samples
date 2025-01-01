import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor


def connect_to_postgres(host, dbname, user, password):
    """
    Establish a connection to the PostgreSQL database.
    """
    try:
        return psycopg2.connect(host=host, database=dbname, user=user, password=password)
    except Exception as e:
        raise Exception(f"Error connecting to PostgreSQL: {e}")


def get_table_structure(conn, table_name):
    """
    Retrieve the structure of a table, including column names and data types.
    """
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public';
                """,
                (table_name,),
            )
            return cur.fetchall()
    except Exception as e:
        raise Exception(f"Error retrieving table structure: {e}")


def execute_query(conn, sql_query):
    """
    Execute an SQL query and return the results as a Pandas DataFrame.
    """
    with conn.cursor() as cur:
        cur.execute(sql_query)
        # Fetch column names
        columns = [desc[0] for desc in cur.description]
        # Fetch the query results
        rows = cur.fetchall()
        # Create a pandas DataFrame with column names
        results_df = pd.DataFrame(rows, columns=columns)
        return results_df
