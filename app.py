import streamlit as st
import os
import asyncio
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI 
from mcp_use import MCPAgent, MCPClient 
# Removed: from langchain.agents import AgentExecutor # Not needed if MCPAgent is self-contained


# Load environment variables from .env file
load_dotenv()

# --- Configuration and Environment Variables ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GITHUB_TOKEN:
    st.error("GitHub Personal Access Token not found. Please set the GITHUB_TOKEN environment variable.")
    st.stop()

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# --- MCP Client Configuration ---
mcp_config = {
    "mcpServers": {
        "github-mcp-remote": {
            "type": "http",
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {
                "Authorization": f"Bearer {GITHUB_TOKEN}"
            }
        }
    }
}

# --- LangChain Setup with MCPAgent ---

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)

try:
    mcp_client = MCPClient(mcp_config["mcpServers"])
except Exception as e:
    st.error(f"Failed to initialize MCP client: {e}")
    st.stop()

# Create the MCPAgent
agent = MCPAgent( 
    llm=llm, 
    client=mcp_client,
)


# --- Streamlit UI ---

st.set_page_config(page_title="GitHub MCP AI Assistant", page_icon="💻")

st.title("💻 GitHub MCP AI Assistant")
st.markdown(
    """
    Hello! I'm your **GitHub Model Context Protocol (MCP) AI Assistant**, powered by Google Gemini.
    I connect directly to the **Remote GitHub MCP Server** (`https://api.githubcopilot.com/mcp/`)
    to help you manage your GitHub repositories using natural language commands.
    
    Think of me as a smart interface to your GitHub actions!
    
    ---
    
    **🔐 Secure Authentication:**
    Your **GitHub Personal Access Token (PAT)** is securely used to authenticate
    all commands with the Remote GitHub MCP Server. This token is passed as a Bearer token
    in the request headers, ensuring your operations are authorized.
    
    ---
    
    **🚀 Ready to Automate Your GitHub Tasks?**
    I can dynamically discover and use various tools exposed by the GitHub MCP server.
    Here are some examples of what you can ask me to do.
    Try these commands to see the power of MCP in action:
    
    * **Create a new repository:**
        `create a public repository called 'my-awesome-project' with description 'A new project managed by the MCP assistant.'`
    * **List your repositories:**
        `list all my github repositories`
    * **Delete a repository (use with caution!):**
        `delete the repository named 'your_username/old-test-repo'`
        **(Important:** Replace 'your_username' with your actual GitHub username and 'old-test-repo' with the exact repository name. This action is irreversible.)
    
    ---
    
    **💡 Pro Tips for Best Results:**
    * Be **specific** about what you want to achieve.
    * Include all **necessary details** (e.g., repository name, public/private status, description).
    * I'll do my best to understand, but if I struggle, try rephrasing your request.
    
    ---
    
    **⚠️ Critical Warning for Delete Operations:**
    Commands like `delete` perform **irreversible actions** on your GitHub account.
    Please use them with extreme care and double-check the repository name before confirming.
    """
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Debugging Section for MCP Tools ---
with st.expander("Debug: Discovered MCP Tools"):
    st.write("The `MCPAgent` internally handles tool discovery from the MCP server.")
    st.write("Directly listing tools via `mcp_client.list_tools()` is not supported by `mcp-use` in this manner.")
    st.info("""
    Since `MCPAgent` directly manages tool execution, its internal verbose logging (if any)
    would determine what appears in your console. If you're not seeing verbose `Thought`/`Action`/`Observation`
    output, it means the `MCPAgent` itself isn't printing it for direct use.
    """)
    # Removed the lines that caused the AttributeError:
    # st.write(f"Number of tools discovered by MCPAgent: {len(agent.tools)}") 
    # if agent.tools:
    #     st.write("First 3 discovered tools (names and descriptions):")
    #     for tool in agent.tools[:3]:
    #         st.write(f"- **{tool.name}**: {tool.description}")
    # else:
    #     st.write("No tools discovered by the MCPAgent. This is a critical issue.")


# Accept user input
if prompt := st.chat_input("How can I help you manage your GitHub today? (e.g., 'list my repos', 'create a new repo named X')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and interacting with GitHub MCP..."):
            try:
                # Use asyncio.run() to execute the async run call of MCPAgent
                response = asyncio.run(agent.run(prompt))
                agent_response = response # MCPAgent.run() directly returns the output

                st.markdown(agent_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": agent_response})
            except Exception as e:
                error_message = f"An error occurred while interacting with MCP: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
