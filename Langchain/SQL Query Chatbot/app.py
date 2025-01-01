import streamlit as st
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from connection import connect_to_postgres, get_table_structure, execute_query

# Sidebar for database credentials
st.sidebar.title("PostgreSQL Database Connection")
host = st.sidebar.text_input("Host", value="localhost")  # Input for database host
user = st.sidebar.text_input("User")  # Input for database username
password = st.sidebar.text_input("Password", type="password")  # Input for password

# Initialize session state variables to retain information across interactions
if "databases" not in st.session_state:
    st.session_state.databases = []
if "selected_db" not in st.session_state:
    st.session_state.selected_db = ""  # Default database
if "tables" not in st.session_state:
    st.session_state.tables = []
if "selected_table" not in st.session_state:
    st.session_state.selected_table = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Function to reset the chat history
def reset_chat():
    st.session_state.chat_history = []


# Load databases from the PostgreSQL server
if st.sidebar.button("Load Databases"):
    conn = connect_to_postgres(host, "postgres", user, password)
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT datname FROM pg_database WHERE datistemplate = false;"
                )
                st.session_state.databases = [row[0] for row in cur.fetchall()]  # Store database names
            conn.close()
        except Exception as e:
            st.error(f"Error retrieving databases: {e}")

# Dropdown to select a database from the list of loaded databases
if st.session_state.databases:
    st.session_state.selected_db = st.sidebar.selectbox(
        "Select Database", st.session_state.databases, index=0
    )

# Retrieve and display tables in the selected database
if st.session_state.selected_db:
    conn = connect_to_postgres(host, st.session_state.selected_db, user, password)
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public';
                    """
                )
                st.session_state.tables = [row[0] for row in cur.fetchall()]  # Store table names
            conn.close()
        except Exception as e:
            st.error(f"Error retrieving tables: {e}")

# Dropdown to select a table and reset chat when table changes
if st.session_state.tables:
    st.session_state.selected_table = st.sidebar.selectbox(
        "Select Table", st.session_state.tables,
        on_change=reset_chat
    )

# Chatbot functionality starts when a table is selected
if st.session_state.selected_table:
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("SQL Query Chatbot with Llama-3.2")
    with col2:
        st.button("Clear ↺", on_click=reset_chat)  # Button to clear chat history

    # Display the selected table's data
    conn = connect_to_postgres(host, st.session_state.selected_db, user, password)
    try:
        sql_query = f"SELECT * FROM {st.session_state.selected_table};"  # Query to fetch all table data
        results = execute_query(conn, sql_query)

        # Show table data in the UI
        if not results.empty:
            st.write(f"Selected Table Name: {st.session_state.selected_table}")
            st.write("Table Data:")
            st.dataframe(results)
        else:
            st.write("No data found in the table.")
    except Exception as e:
        st.error(f"Error executing query: {e}")
    finally:
        conn.close()

    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            st.chat_message(message["role"]).markdown(message["content"])

    # User input box for questions
    user_input = st.chat_input("Ask a question about the table")

    if user_input:
        # Add user input to the chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # Retrieve table structure to provide context for LLM
        conn = connect_to_postgres(host, st.session_state.selected_db, user, password)
        table_structure = get_table_structure(conn, st.session_state.selected_table)
        conn.close()

        # Prepare the context using the table's structure
        columns = ", ".join(
            [f"{col['column_name']} ({col['data_type']})" for col in table_structure]
        )
        context = (
            f"The table `{st.session_state.selected_table}` has the following columns: {columns}."
        )

        # Define PromptTemplate for generating SQL queries
        query_prompt_template = PromptTemplate(
            input_variables=["user_input", "context"],
            template=(
                "You are an SQL expert. Based on the user's input, "
                "generate only the SQL query without any explanation.\n"
                "Context: {context}\n"
                "User Input: {user_input}\n"
                "Generate a valid SQL query based on the user input."
            ),
        )

        parser = StrOutputParser()  # Output parser for extracting query

        # Initialize LangChain LLM
        llm = ChatOllama(model="llama3.2", temperature=0.7)
        query_chain = LLMChain(
            llm=llm,
            prompt=query_prompt_template,
            output_key="sql_query",
            output_parser=parser,
        )

        with st.spinner("Generating answer..."):
            # Run the chain and generate SQL query
            result = query_chain.invoke({"user_input": user_input, "context": context})
            sql_query = result["sql_query"]

            # Add generated query to chat history and display it
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"Generated SQL Query:\n```sql\n{sql_query}\n```"}
            )
            st.chat_message("assistant").markdown(f"Generated SQL Query:\n```sql\n{sql_query}\n```")

            # Execute the generated query and show results
            conn = connect_to_postgres(host, st.session_state.selected_db, user, password)
            try:
                results = execute_query(conn, sql_query)
                st.write("Query Results:")
                st.dataframe(results)
            except Exception as e:
                st.error(str(e))
            finally:
                conn.close()
