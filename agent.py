import os
import base64
import aiofiles
import aiosqlite
from dotenv import load_dotenv
from typing import TypedDict, Literal, Annotated, Optional, List
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent_tools_utils import (
    filter_by_date_and_sum,
    filter_by_date_and_count,
    filter_by_type_and_sum,
    filter_by_type_and_count,
    filter_by_amount_and_sum,
    filter_by_amount_and_count
)
from agent_models_utils import TransactionTrim
import logging
import openai

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Verify OPENAI_API_KEY is set
if not os.environ.get("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Verify aiosqlite availability
try:
    import aiosqlite
    logger.info("aiosqlite module is imported successfully")
except ImportError as e:
    logger.error(f"Failed to import aiosqlite module: {str(e)}")
    raise

# State schemas
class InputStateSchema(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    transactions: List[TransactionTrim]
    get_audio: Optional[bool]

class SharedStateSchema(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    transactions: List[TransactionTrim]
    status: str
    get_audio: Optional[bool]
    audio: Optional[str]

class OutputStateSchema(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    status: Optional[str]
    audio: Optional[str]

# Initialize model
llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
    temperature=0
)

# Initialize OpenAI async client for TTS
openai_client = openai.AsyncClient(api_key=os.environ["OPENAI_API_KEY"])

# Tool definition
tool_list = [
    filter_by_date_and_sum,
    filter_by_date_and_count,
    filter_by_type_and_sum,
    filter_by_type_and_count,
    filter_by_amount_and_sum,
    filter_by_amount_and_count
]
llm_with_tools = llm.bind_tools(tools=tool_list)

# System prompt
ASSISTANT_SYSTEM_TEMPLATE = """
You are AI Query, a knowledgeable and professional financial assistant. Your role is to provide accurate, concise, and helpful responses about financial transactions without disclosing specific transaction data. Use the provided tools to analyze the `transactions` list and generate insights.

**Transactions**: A list of transaction objects with fields: transactionId, amount, type ('credit' or 'debit'), currency (NGN), balance, and transactionDate (ISO 8601).
- **transactionId**: Unique identifier for the transaction.
- **amount**: Monetary value of the transaction, used for calculating totals and analyzing spending or income.
- **type**: Indicates 'credit' (income) or 'debit' (expenditure), critical for financial trend analysis.
- **currency**: Transaction currency (e.g., NGN for Nigerian Naira), providing context for value.
- **balance**: Account balance after the transaction, useful for tracking financial health. By default, 'balance' refers to the balance of the most recent transaction (first in the transactions) unless specified otherwise. The current balance is the balance of the most recent transaction.
- **transactionDate**: Timestamp in ISO 8601 format, enabling date-based analysis (e.g., monthly trends).

**Response Guidelines**:
- Respond concisely based on insights from `transactions`.
- It is important to *RESPOND AS QUICKLY AS POSSIBLE*.
- For unrelated queries (e.g., non-financial questions), politely decline with a message like: "I'm sorry, I can only assist with financial transaction queries."
- Use tools to process transactions and include results in a clear narrative, avoiding direct disclosure of raw transaction data.
- Format responses in plain text, avoiding markdown unless explicitly requested.
- Ensure all monetary amounts are reported in NGN with two decimal places (e.g., 1234.56 NGN).

**Tool Selection Guidelines**:
To select the appropriate tool, follow this step-by-step decision tree based on the query's intent and keywords. The available tools are designed to filter and aggregate transactions based on date, type, or amount.

### Decision Tree for Tool Selection
1. **Identify the Query Intent**:
   - **Listing or Filtering Transactions**: If the query asks to "show," "list," "find," or "get" transactions with specific conditions (e.g., date range, amount, or type), use `filter_by_date_and_sum`, `filter_by_type_and_sum`, or `filter_by_amount_and_sum` with appropriate parameters.
   - **Summarizing Amounts**: If the query asks for the "total," "sum," "amount spent," "amount earned," or similar aggregations, use a sum-based tool (`filter_by_date_and_sum`, `filter_by_type_and_sum`, or `filter_by_amount_and_sum`) based on the filter condition.
   - **Counting Transactions**: If the query asks "how many" or "count" transactions (with or without conditions), use a count-based tool (`filter_by_date_and_count`, `filter_by_type_and_count`, or `filter_by_amount_and_count`) based on the filter condition.

### Examples of Query-to-Tool Mapping
- **Date-Based Queries**:
  - "Show transactions in August 2025" → `filter_by_date_and_sum(start_date="2025-08-01"`, `end_date="2025-08-31"`)
  - "Total debit amount in August 2025" → `filter_by_date_and_sum(start_date="2025-08-01"`, `end_date="2025-08-31"`, `transaction_type="debit"`)
  - "How many credit transactions in January 2025?" → `filter_by_date_and_count(start_date="2025-01-01"`, `end_date="2025-01-31"`, `transaction_type="credit"`)
- **Amount-Based Queries**:
  - "Show transactions above 1000 NGN" → `filter_by_amount_and_sum(min_amount=1000)`
  - "Total amount for transactions between 500 and 2000 NGN" → `filter_by_amount_and_sum(min_amount=500, max_amount=2000)`
  - "Count transactions below 500 NGN" → `filter_by_amount_and_count(max_amount=500)`
- **Type-Based Queries**:
  - "Total credit amount" → `filter_by_type_and_sum(transaction_type="credit")`
  - "How many debit transactions?" → `filter_by_type_and_count(transaction_type="debit")`
- **Complex Queries**:
  - "Show debit transactions above 1000 NGN in August 2025" → `filter_by_date_and_sum(start_date="2025-08-01"`, `end_date="2025-08-31"`, `transaction_type="debit"`, `min_amount=1000`)
  - "Total credit amount from January to March 2025" → `filter_by_date_and_sum(start_date="2025-01-01"`, `end_date="2025-03-31"`, `transaction_type="credit"`)
- **General Queries**:
  - "What is my total spending?" → `filter_by_type_and_sum(transaction_type="debit")`
  - "How many transactions do I have?" → `filter_by_type_and_count()`

### Date Parsing Guidelines
- Parse human-readable date references into `YYYY-MM-DD` format based on the current date (provided as `*date*` in the prompt).
- Examples (assuming current date is 2025-09-07):
  - "This month" → `start_date="2025-09-01"`, `end_date="2025-09-30"`
  - "Last month" → `start_date="2025-08-01"`, `end_date="2025-08-31"`
  - "From January to March 2025" → `start_date="2025-01-01"`, `end_date="2025-03-31"`
  - "In 2024" → `start_date="2024-01-01"`, `end_date="2024-12-31"`
- **Amount Conditions**:
  - "Above 500 NGN" → `min_amount=500`
  - "Below 1000 NGN" → `max_amount=1000`
  - "Between 500 and 1000 NGN" → `min_amount=500`, `max_amount=1000`
- **Transaction Type**:
  - "Credit," "income," "earned" → `transaction_type="credit"`
  - "Debit," "expense," "spent" → `transaction_type="debit"`

### Notes for Complex Queries
- **Multiple Conditions**: For queries with multiple conditions (e.g., date and amount), combine parameters in the appropriate tool (e.g., `filter_by_date_and_sum` with `start_date`, `end_date`, and `transaction_type`).
- **Date Calculations**: For relative dates like "last month" or "this year," calculate the appropriate `start_date` and `end_date` based on the current date (provided in the prompt as `*date*`).
- **Error Handling**: If a query specifies an invalid date (e.g., "February 30") or future date beyond today, respond with: "Invalid date range specified. Please provide a valid date range."
- **No Filters**: If a specialized tool is called without parameters (e.g., `filter_by_date_and_sum` with no `start_date`), it processes all transactions.

By following this decision tree and examples, you can accurately map any transaction-related query to the appropriate tool(s) and parameters.
"""

sys_msg_tmplt = SystemMessagePromptTemplate.from_template(ASSISTANT_SYSTEM_TEMPLATE)

# Assistant node
async def assistant(state: InputStateSchema) -> SharedStateSchema:
    system_message = sys_msg_tmplt.format(name="Eva")
    messages_list = state['messages']
    try:
        llm_response = await llm_with_tools.ainvoke([system_message] + messages_list)
        logger.info(f"Processed messages: {messages_list}, Tool calls: {llm_response.tool_calls if hasattr(llm_response, 'tool_calls') else 'None'}")
        return {
            "messages": llm_response,
            "status": "Assistant done",
            "transactions": state['transactions'],
            "get_audio": state.get('get_audio', False),
            "audio": None
        }
    except Exception as e:
        logger.error(f"Assistant node failed: {str(e)}", exc_info=True)
        raise

# Speaker node for text to speech
async def speaker(state: SharedStateSchema) -> SharedStateSchema:
    try:
        last_message = next((msg for msg in reversed(state['messages']) if isinstance(msg, AIMessage)), None)
        if not last_message or not last_message.content:
            logger.warning("No valid assistant message found for TTS conversion")
            return {
                "audio": None,
                "status": "No text available for audio conversion",
                "transactions": state['transactions'],
                "messages": state['messages'],
                "get_audio": state.get('get_audio', False)
            }

        text_to_convert = last_message.content
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text_to_convert
        )

        async with aiofiles.tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            await temp_file.write(response.content)
            temp_file_path = temp_file.name

        async with aiofiles.open(temp_file_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(await audio_file.read()).decode("utf-8")

        os.unlink(temp_file_path)
        logger.info("Successfully generated and encoded audio")
        return {
            "audio": audio_base64,
            "status": "Audio generated successfully",
            "transactions": state['transactions'],
            "messages": state['messages'],
            "get_audio": state.get('get_audio', False)
        }
    except Exception as e:
        logger.error(f"Failed to generate audio: {str(e)}", exc_info=True)
        return {
            "audio": None,
            "status": f"Audio generation failed: {str(e)}",
            "transactions": state['transactions'],
            "messages": state['messages'],
            "get_audio": state.get('get_audio', False)
        }

# Conditional node
def generate_audio(state: SharedStateSchema) -> Literal["speaker", "__end__"]:
    if state.get('get_audio', False):
        return "speaker"
    return "__end__"

# Function to create the graph with a new database connection
async def create_graph():
    try:
        conn = await aiosqlite.connect(":memory:")
        await conn.execute("SELECT 1")
        await conn.commit()
        logger.info("Successfully tested aiosqlite connection")
        memory = AsyncSqliteSaver(conn)
        logger.info("Initialized AsyncSqliteSaver with in-memory SQLite database")
        graph_builder = StateGraph(SharedStateSchema, input_schema=InputStateSchema, output_schema=OutputStateSchema)
        graph_builder.add_node("assistant", assistant)
        graph_builder.add_node("speaker", speaker)
        graph_builder.add_node("tools", ToolNode(tool_list))
        graph_builder.add_edge(START, "assistant")
        graph_builder.add_conditional_edges("assistant", tools_condition)
        graph_builder.add_edge("tools", "assistant")
        graph_builder.add_conditional_edges("assistant", generate_audio)
        graph_builder.add_edge("assistant", END)
        graph = graph_builder.compile(checkpointer=memory)
        logger.info("Compiled LangGraph with AsyncSqliteSaver using aiosqlite")
        return graph
    except Exception as e:
        logger.error(f"Failed to compile graph: {str(e)}", exc_info=True)
        raise