import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os

examples = [
    {
        "input": "List all customers in France with a credit limit over 20,000.",
        "query": "SELECT * FROM customers WHERE country = 'France' AND creditLimit > 20000;"
    },
    {
        "input": "Get the highest payment amount made by any customer.",
        "query": "SELECT MAX(amount) FROM payments;"
    },
    {
        "input": "Show product details for products in the 'Motorcycles' product line.",
        "query": "SELECT * FROM products WHERE productLine = 'Motorcycles';"
    },
    {
        "input": "Retrieve the names of employees who report to employee number 1002.",
        "query": "SELECT firstName, lastName FROM employees WHERE reportsTo = 1002;"
    },
    {
        "input": "List all products with a stock quantity less than 7000.",
        "query": "SELECT productName, quantityInStock FROM products WHERE quantityInStock < 7000;"
    },
    {
        'input': "what is price of `1968 Ford Mustang`",
        "query": "SELECT `buyPrice`, `MSRP` FROM products  WHERE `productName` = '1968 Ford Mustang' LIMIT 1;"
    }
]


@st.cache_resource
def get_example_selector():
    # Create a persist_directory if it doesn't exist
    persist_directory = "chroma_db"
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize the embeddings
    embeddings = OpenAIEmbeddings()

    # Convert examples to Documents
    documents = [
        Document(
            page_content=ex["input"],
            metadata={"input": ex["input"], "query": ex["query"]}
        )
        for ex in examples
    ]

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=persist_directory
    )

    # Create the example selector
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
        input_keys=["input"],
    )

    return example_selector