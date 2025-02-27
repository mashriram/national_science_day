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


# ----------------------- 1. Load and Prepare PDF Data -----------------------
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


def create_retriever(vectordb, search_kwargs={"k": 3}):
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    return retriever


# ----------------------- 3. Initialize ChatOllama -----------------------
def initialize_ollama(model="gemma:latest"):
    llm = ChatOllama(model=model)
    return llm


# ----------------------- 4. Define All Tools -----------------------
def define_tools(
    root_dir, stable_diffusion_url="http://localhost:7860"
):  # URL Parameter

    # Standard Tools
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    python_repl = PythonREPLTool()
    shell_tool = ShellTool()
    search = DuckDuckGoSearchRun()
    youtube_search = YouTubeSearchTool()

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
        *file_tools,  # Unpack the file tools.
    ]

    return tools


# ----------------------- 6. Main Function (Modified for Gradio) -----------------------
def create_agent(pdf_path, memory):
    """Create the agent components once."""

    documents = load_pdf(pdf_path)
    chunks = split_text(documents)

    # 2. Create Vectorstore and Retriever
    vectordb = create_vectorstore(chunks)
    retriever = create_retriever(vectordb)

    # 3. Initialize ChatOllama
    llm = initialize_ollama()

    # 4. Define Tools
    working_directory = TemporaryDirectory()
    tools = define_tools(str(working_directory.name))

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
              Thought: I am ready to answer
              Final Answer: the final answer to the original input question

              {chat_history}
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

    return agent_executor, working_directory


def process_query(
    query, memory, agent_executor  # Pass in agent_executor
):  # Added stable diffusion URL

    # 6. Invoke the agent
    result = agent_executor.invoke({"input": query})

    final_message = result["output"]
    return final_message


# ----------------------- Gradio Interface -----------------------
def gradio_interface(pdf_file):  # Removed stable_diffusion_url - Not Used

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )  # Load Memory
    agent_executor, working_directory = create_agent(pdf_file, memory)  # Load Agent

    def chat(query, history):
        if query.lower() == "quit":
            working_directory.cleanup()  # Clean it up
            return None  # Signal to end the session

        response = process_query(query, memory, agent_executor)  # Pass in URL

        # Format history for Gradio
        history.append([query, response])
        new_history = []
        for entry in history:
            if len(entry) == 2:
                human, ai = entry
                new_history.append({"role": "user", "content": human})
                new_history.append({"role": "assistant", "content": ai})
            else:
                print(f"Skipping malformed history entry: {entry}")

        return new_history

    with gr.Blocks(
        css=".gradio-container {height: 100% !important;}"
    ) as demo:  # Full Screen
        gr.HTML(
            "<p style='text-align: center; font-size: 2em;'>National Science Day PDF Chatbot</p>"
        )  # Title
        chatbot = gr.Chatbot(
            type="messages", max_height="90vh", show_copy_button=True
        )  # Increase Chatbot height

        text_input = gr.Textbox(label="Enter your query:")
        clear_button = gr.ClearButton([text_input, chatbot])

        text_input.submit(chat, [text_input, chatbot], [chatbot])
        clear_button.click(lambda: None, None, chatbot, queue=False)

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

    demo = gradio_interface(pdf_file)  # Pass in the Stable Diffusion URL
    demo.launch()
