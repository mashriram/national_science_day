import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Dict, Any, List, Tuple, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory  # Used to load memory

# Additional Tools
from langchain_community.tools import ShellTool
from langchain_community.tools import DuckDuckGoSearchRun
from tempfile import TemporaryDirectory
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools import YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool

# Gradio
import gradio as gr

# ----------------------- 0. Create Data Directory (if it doesn't exist) -----------------------
DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# ----------------------- 1. Load and Prepare PDF Data (Load Separately) -----------------------
def load_and_prepare_pdf(pdf_path):
    documents = load_pdf(pdf_path)
    chunks = split_text(documents)
    vectordb = create_vectorstore(chunks)
    retriever = create_retriever(vectordb)
    return retriever


def load_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return []


def split_text(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# ----------------------- 2. Set up Embeddings and Vectorstore -----------------------
def create_vectorstore(
    chunks, embedding_model="all-MiniLM-L6-v2", persist_directory="chroma_db"
):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_directory
    )
    return vectordb


def create_retriever(vectordb: Chroma):  # Removed search_kwargs
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8}
    )  # Remove the Search
    return retriever


# ----------------------- 3. Initialize ChatOllama -----------------------
def initialize_ollama(model="gemma:latest"):
    llm = ChatOllama(model=model)
    return llm


# ----------------------- 4. Define All Tools -----------------------
def define_tools(root_dir, retriever):  # Pass in retriever  # URL Parameter

    # Standard Tools
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    python_repl = PythonREPLTool()
    shell_tool = ShellTool()
    search = DuckDuckGoSearchRun()
    youtube_search = YouTubeSearchTool()

    # Define a tool for the retriever
    retriever_tool = Tool(
        name="PDF Retriever",
        func=retriever,
        description="Useful for when you need to answer questions related to the PDF document.USE this whenever PDF related questions are asked.",
    )

    # File Management Toolkit
    toolkit = FileManagementToolkit(root_dir=root_dir)
    file_tools = toolkit.get_tools()

    tools = [
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="Useful for when you need to answer general questions about factual topics. Input should be a search query.",
        ),
        Tool(
            name="Python REPL",
            func=python_repl.run,
            description="Useful for when you need to perform calculations or run Python code. Input should be valid Python code.",
        ),
        Tool(
            name="Shell Tool",
            func=shell_tool.run,
            description="Useful for executing shell commands.  Use with caution.",
        ),
        Tool(
            name="DuckDuckGo Search",
            func=search.run,
            description="Useful for searching the internet. Input should be a search query.",
        ),
        Tool(
            name="YouTube Search",
            func=youtube_search.run,
            description="Useful for searching on YouTube. Input should be a search query.",
        ),
        Tool(
            name="YouTube Search",
            func=retriever_tool.run,
            description="Useful for searching on PDFS. Input should be a search query for the document.",
        ),  # PDF REtrieval
        *file_tools,  # Unpack the file tools.
    ]

    return tools


# ----------------------- 6. Create Agent (Load Tools Seperately) -----------------------


def create_agent(llm, tools, memory):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant, tasked with answering questions using the following tools.
              Use the following format:

              Question: the input question you must answer
              Thought: you should always think about what to do
              Action: the action to take, should be one of [{tool_names}]
              Action Input: the input to the action
              Observation: the result of the action
              ... (this Thought/Action/Action Input/Observation can repeat N times)
              Thought: I have the summary
              Final Answer: <generated summary>

              Always start with "retriever" Tool when you need to answer questions or generate a summary about the PDF. give the answer that you got from pdf in the Final Answer DO NOT STORE anything temporarily in files
              """,
            ),
            ("user", "{input}"),
            ("agent:{agent_scratchpad}"),
        ]
    )
    prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, memory=memory
    )

    return agent_executor


# ----------------------- 7. Process Query (Pass the agent and the tool)-----------------------
def process_query(query, agent_executor, memory):

    result = agent_executor.invoke({"input": query})
    final_message = result["output"]
    return final_message


# ----------------------- Gradio Interface -----------------------
def gradio_interface(pdf_file):

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = load_and_prepare_pdf(pdf_file)

    llm = initialize_ollama()
    working_directory = TemporaryDirectory()
    tools = define_tools(str(working_directory.name), retriever.invoke)
    agent_executor = create_agent(llm, tools, memory)

    def chat(query, history):
        if query.lower() == "quit":
            working_directory.cleanup()
            return None

        response = process_query(
            query, agent_executor, memory
        )  # Pass in the memory here

        history.append([query, response])

        # Format history for Gradio
        new_history = []
        for entry in history:
            if len(entry) == 2:
                human, ai = entry
                new_history.append({"role": "user", "content": human})
                new_history.append({"role": "assistant", "content": ai})
            else:
                print(f"Skipping malformed history entry: {entry}")

        return new_history

    with gr.Blocks(css=".gradio-container {width: 100% !important;}") as demo:
        gr.HTML(
            "<p style='text-align: center; font-size: 2em;'>National Science Day PDF Chatbot</p>"
        )
        chatbot = gr.Chatbot(type="messages", height=600, show_copy_button=True)

        text_input = gr.Textbox(label="Enter your query:")
        clear_button = gr.ClearButton([text_input, chatbot])

        text_input.submit(chat, [text_input, chatbot], [chatbot])
        clear_button.click(lambda: None, None, chatbot, queue=False)

    return demo


# ----------------------- Example Usage (Gradio) -----------------------
if __name__ == "__main__":
    pdf_file = os.path.join(DATA_DIR, "document.pdf")  # Look in the ./data directory

    # Create a dummy PDF file if one doesn't exist
    if not os.path.exists(pdf_file):
        with open(pdf_file, "w") as f:
            f.write("This is a dummy PDF file for testing.")
        print(f"Created a dummy PDF file at {pdf_file}")

    demo = gradio_interface(pdf_file)
    demo.launch()
