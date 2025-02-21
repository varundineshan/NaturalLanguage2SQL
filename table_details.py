import pandas as pd
import streamlit as st
from operator import itemgetter
from dotenv import load_dotenv
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from pydantic.v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List
import os

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo")


@st.cache_data
def get_table_details():
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the CSV file
        csv_path = os.path.join(current_dir, "database_table_descriptions.csv")

        # Read the CSV file into a DataFrame
        table_description = pd.read_csv(csv_path)

        # Convert DataFrame to the required string format
        table_details = ""
        for _, row in table_description.iterrows():
            table_details += f"Table Name: {row['Table']}\n"
            table_details += f"Table Description: {row['Description']}\n\n"

        if not table_details:
            raise ValueError("No table descriptions found in the CSV file")

        return table_details

    except FileNotFoundError:
        st.error(f"Could not find database_table_descriptions.csv in {csv_path}")
        raise
    except Exception as e:
        st.error(f"Error reading table descriptions: {str(e)}")
        raise


class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")


def get_tables(tables: List[Table]) -> List[str]:
    tables = [table.name for table in tables]
    return tables


try:
    table_details = get_table_details()
    table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
    The tables are:

    {table_details}

    Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

    table_chain = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm,
                                                                                       system_message=table_details_prompt) | get_tables

except Exception as e:
    st.error("Failed to initialize table details")
    raise