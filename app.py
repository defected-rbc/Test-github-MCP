import streamlit as st
import requests
import json
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Import MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration for Remote GitHub MCP Server ---
REMOTE_MCP_SERVER_URL = "https://api.githubcopilot.com/mcp/"

# --- Simplified MCP Server Client ---
class McpServerClient:
    def __init__(self, github_pat: str):
        self.headers = {
            "Authorization": f"Bearer {github_pat}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.base_url = REMOTE_MCP_SERVER_URL

    def _call_mcp_tool(self, tool_name: str, payload: dict) -> dict:
        url = f"{self.base_url}api/v1/tools/{tool_name}/invoke"
        st.info(f"Calling MCP tool: {tool_name} with payload: {payload}")
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
            raise
        except Exception as err:
            st.error(f"Other error occurred: {err}")
            raise

    def list_branches(self, owner: str, repo: str) -> str:
        tool_name = "repositories.list_branches"
        payload = {"owner": owner, "repo": repo}
        try:
            result = self._call_mcp_tool(tool_name, payload)
            if result and isinstance(result, dict) and "output" in result and "branches" in result["output"]:
                branches = [b["name"] for b in result["output"]["branches"]]
                return f"Successfully listed branches: {json.dumps(branches, indent=2)}"
            elif result:
                return f"MCP server response for list_branches: {json.dumps(result, indent=2)}"
            else:
                return "No branches found or unexpected response format."
        except Exception as e:
            return f"Failed to list branches: {e}"


# --- LangChain Tool Definitions ---
@tool
def github_list_branches(owner: str, repo: str) -> str:
    """
    Lists branches in a specified GitHub repository.
    Input should be a JSON string with 'owner' and 'repo' keys.
    Example: {"owner": "streamlit", "repo": "streamlit"}
    """
    github_pat = st.secrets["GITHUB_PAT"]
    client = McpServerClient(github_pat=github_pat)
    return client.list_branches(owner=owner, repo=repo)

# --- Streamlit Application UI ---

st.set_page_config(page_title="GitHub MCP LangChain Agent", layout="wide")

st.title("🔗 GitHub MCP LangChain Assistant")
st.markdown("This application connects to the **remote GitHub Official MCP Server** via LangChain to perform GitHub operations.")
st.markdown("Please ensure your GitHub Personal Access Token (PAT) is configured in Streamlit secrets (`GITHUB_PAT`) with appropriate scopes for repository access.")

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Google API key not found in Streamlit secrets. Please add it to use the LLM.")
elif "GITHUB_PAT" not in st.secrets:
    st.error("GitHub Personal Access Token (PAT) not found in Streamlit secrets. Please add it to interact with GitHub MCP.")
else:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])

    tools = [github_list_branches]

    # --- CORRECTED PROMPT DEFINITION ---
    # We now explicitly include {tools} and {tool_names} in the system message,
    # and use MessagesPlaceholder for agent_scratchpad.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful GitHub assistant. You have access to the following tools:\n\n"
                    "{tools}\n\n" # This placeholder will be filled with tool descriptions
                    "Use the following format:\n\n"
                    "Question: the input question you must answer\n"
                    "Thought: you should always think about what to do\n"
                    "Action: the action to take, should be one of [{tool_names}]\n" # This placeholder will be filled with tool names
                    "Action Input: the input to the action\n"
                    "Observation: the result of the action\n"
                    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
                    "Thought: I now know the final answer\n"
                    "Final Answer: the final answer to the original input question\n"
                ),
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Recommended for chat history/scratchpad with ChatPromptTemplate
        ]
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    st.header("Ask your GitHub question:")
    user_query = st.text_area(
        "Example: List the branches in the 'streamlit/streamlit' repository.",
        placeholder="e.g., What are the branches in the 'octocat/Spoon-Knife' repository?",
        height=100
    )

    if st.button("Get GitHub Info"):
        if user_query:
            with st.spinner("Processing your request..."):
                try:
                    response = agent_executor.invoke({"input": user_query})
                    st.success("Request processed!")
                    st.subheader("Agent's Response:")
                    st.write(response["output"])
                    st.info("Check the 'Logs' above for the agent's thought process.")

                except Exception as e:
                    st.error(f"An error occurred while processing your request: {e}")
                    st.warning("Please check your GitHub PAT scopes and ensure the repository owner/name are correct.")
        else:
            st.warning("Please enter a query to get GitHub information.")

st.markdown("---")
st.caption("Note: For production environments, consider a more robust OAuth 2.0 flow for GitHub authentication.")