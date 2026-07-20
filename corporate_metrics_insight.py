"""
InsightForge: BI Assistant with RAG Integration
Converts Jupyter notebook to production Python script

pip install pypdf langchain-openai langchain-community langchain-experimental faiss-cpu

"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Configure API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
 
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# try:
#     response = llm.invoke("Say 'Hello from OpenAI!' in exactly 4 words.")
#     print("Connection successful!")
#     print(f"Response: {response.content}")
# except Exception as e:
#     print(f"Connection failed: {e}")

# # ============================================================================
# # SECTION 1.1: Data Prep: Loading & Exploration
# # ============================================================================

# # Update these paths to your local dataset
DATA_PATH = "./dataset/sales_data.csv"
PDF_PATH = "./dataset/PDF Folder"
INDEX_PATH = "./insightforge_faiss_index"

# # Load CSV data
df = pd.read_csv(DATA_PATH)

print("--- Data Preview ---")
print(df.head())
print("\n--- Schema Information ---")
print(df.info())

# # ============================================================================
# # SECTION 1.2: KB Creation: Generate Knowledge Base Metadata
# # ============================================================================

knowledge_base_meta = f"""
### DATASET SCHEMA
- Columns: {', '.join(df.columns.tolist())}

### DATA TYPES
{df.dtypes.to_string()}

### SAMPLE DATA
{df.head(3).to_markdown()}
"""

# # ============================================================================
# # SECTION 1.3: LLM App dev:  System Prompt Definition
# # ============================================================================

system_prompt = f"""
You are an expert Data Analyst BI Assistant.
You have access to a pandas DataFrame.
Your goal is to answer user questions by generating accurate and executable pandas code.

---
KNOWLEDGE BASE:
{knowledge_base_meta}
---

ANALYTICAL CAPABILITIES:
You are capable of performing:
1. Sales performance analysis by time period.
2. Product and regional performance analysis.
3. Customer segmentation analysis by demographics.
4. Statistical calculations (e.g., median, standard deviation).

INSTRUCTIONS:
1. Only use the columns available in the KNOWLEDGE BASE.
2. Return ONLY the python code necessary to answer the question.
3. Use the variable name 'df' for the DataFrame.
4. Do not include markdown formatting; just return the code.
5. For multi-metric requests, break the analysis into sequential, logical steps.
6. Assign results to clearly named variables and include print() statements for the final outputs.
"""

# # ============================================================================
# # SECTION 1.3.1:   Pandas Agent
# # ============================================================================


# Create Pandas Agent
pandas_agent = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type="tool-calling",
    verbose=True,
    allow_dangerous_code=True
)

# # ============================================================================
# # SECTION 1.3.2: Integration with RAG System: RAG Setup (PDF Retrieval)
# # ============================================================================

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Check if index exists, otherwise build it
if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("Building new FAISS index from PDFs...")
    # Load PDFs
    loader = PyPDFDirectoryLoader(PDF_PATH)
    documents = loader.load()

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    # Create FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"Index saved. Processed {len(docs)} chunks.")

# Initialize retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# # ============================================================================
# # SECTION 1.4.1: Chain Promps
# # ============================================================================

@tool
def pandas_data_analyst(query: str):
    """Useful for quantitative analysis, sales trends, and statistical calculations on the sales dataset."""
    return pandas_agent.invoke({"input": query})

@tool
def document_retriever(query: str):
    """Useful for retrieving qualitative context, internal memos, or business strategy documents."""
    return retriever.invoke(query)

tools = [pandas_data_analyst, document_retriever]

# # ============================================================================
# # SECTION 1.6: Initialize Agent with Memory
# # ============================================================================

system_instruction = f"""
You are InsightForge, a professional BI Assistant acting as a Traffic Controller.

TRAFFIC CONTROLLER RULES:
- If the user asks for numbers, sales metrics, or statistics, use the 'pandas_data_analyst' tool.
- If the user asks for context, explanations, or business strategy documents, use the 'document_retriever' tool.
- After using a tool, synthesize the findings into a single, cohesive business response.

{system_prompt}
"""

# # Initialize checkpointer for memory
checkpointer = MemorySaver()

# Create agent executor
agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_instruction,
    checkpointer=checkpointer
)

# Part 2: LLMOps (Model Evaluation, Monitoring,
# and User Interface Creation Using Streamlit)

# 7. External tool integration

# 7.1 Model evaluation: Apply QAEvalChain to assess the model's performance and accuracy

# 7.2 Data visualization: Create various plots and visualizations to present insights, including:
# Sales trends over time
# Product performance comparisons
# Regional analysis
# Customer demographics and segmentation

# 7.3 Streamlit UI: Develop an intuitive user interface using Streamlit, allowing users to interact with the AI assistant and access visualizations and insights

# # ============================================================================
# # SECTION 8: Agent Query Function
# # ============================================================================

def test_agent(query: str = None, thread_id: str = "session_1"):
    """
    Test the InsightForge agent with a query.

    Args:
        query: The question to ask the agent
        thread_id: Session ID for memory management (default: "session_1")

    Returns:
        Response from the agent
    """
    if query is None:
        query = "Calculate the average 'Customer_Satisfaction' score per 'Product' category. Then, search the PDF research documents to identify management strategies for improving customer satisfaction in product lines that are currently underperforming the company average."

    # Configure session with memory
    config = {"configurable": {"thread_id": thread_id}}

    # Invoke agent
    response = agent_executor.invoke(
        {"messages": [("user", query)]},
        config=config
    )

    # Print response
    print("\n" + "=" * 80)
    print("AGENT RESPONSE:")
    print("=" * 80)
    print(response["messages"][-1].content)

    return response

# # Uncomment below to test the agent
# if __name__ == "__main__":
#     test_agent()

