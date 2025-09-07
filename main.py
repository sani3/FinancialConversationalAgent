import logging
from fastapi import FastAPI, HTTPException
from typing_extensions import TypedDict, List, Optional
from datetime import date
from agent import create_graph, InputStateSchema, TransactionTrim
from langchain_core.prompts import HumanMessagePromptTemplate
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("langgraph").setLevel(logging.WARNING)  # Suppress verbose langgraph logs
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Global graph variable, to be initialized during startup
graph = None

# Request and response models
class InvokeRequest(TypedDict):
    prompt: str
    thread_id: str
    transactions: List[TransactionTrim]
    get_audio: Optional[bool]

class InvokeResponse(TypedDict):
    messages: str
    audio: Optional[str]

# Manual validation for InvokeRequest
def validate_invoke_request(request: InvokeRequest):
    if not isinstance(request.get('prompt'), str) or not request['prompt']:
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(request.get('thread_id'), str) or not request['thread_id']:
        raise ValueError("thread_id must be a non-empty string")
    if not isinstance(request.get('transactions'), list):
        raise ValueError("transactions must be a list")
    if request.get('get_audio') is not None and not isinstance(request['get_audio'], bool):
        raise ValueError("get_audio must be a boolean")
    # Validate transactions
    for t in request.get('transactions', []):
        if not isinstance(t.get('transactionId'), str):
            raise ValueError("transactionId must be a string")
        if not isinstance(t.get('amount'), (str, int, float)):
            raise ValueError("amount must be a number or string")
        if t.get('type') not in ['credit', 'debit']:
            raise ValueError("type must be 'credit' or 'debit'")
        if t.get('currency') != 'NGN':
            raise ValueError("currency must be 'NGN'")
        if not isinstance(t.get('balance'), (str, int, float)):
            raise ValueError("balance must be a number or string")
        if not isinstance(t.get('transactionDate'), str) or not re.match(r"^\d{4}-\d{2}-\d{2}T.*Z$", t['transactionDate']):
            raise ValueError("transactionDate must be in ISO 8601 format")

# Startup event to initialize graph
@app.on_event("startup")
async def startup_event():
    global graph
    try:
        graph = await create_graph()
        logger.info("Successfully initialized graph")
    except Exception as e:
        logger.error(f"Failed to initialize graph: {str(e)}", exc_info=True)
        raise

# Shutdown event to clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    global graph
    try:
        await graph.checkpointer.conn.close()
        logger.info("Closed aiosqlite connection")
    except Exception as e:
        logger.error(f"Failed to close aiosqlite connection: {str(e)}", exc_info=True)

# Index
@app.get("/")
async def index():
    """
    Returns index string
    """
    return "Server is up and running"

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Invoke graph endpoint
@app.post("/conversation", response_model=InvokeResponse)
async def invoke_graph(request: InvokeRequest):
    global graph
    if graph is None:
        logger.error("Graph is not initialized")
        raise HTTPException(status_code=500, detail="Graph is not initialized")
    try:
        # Validate request
        validate_invoke_request(request)

        prompt = request['prompt']
        thread_id = request['thread_id']
        transactions = [t.copy() for t in request['transactions']]
        transactions = sorted(transactions, key=lambda x: x['transactionDate'], reverse=True)
        get_audio = request.get('get_audio', False)

        prompt_template = HumanMessagePromptTemplate.from_template(
            """
            *prompt*: {prompt}
            *date*: {date}
            *transactions*: {transactions}
            **Tools**: You have access to the following tools to process transactions:
                - **filter_by_date_and_sum**: Sum transactions by date range. Example: "Total debit amount in January 2025" -> call with `start_date="2025-01-01"`, `end_date="2025-01-31"`, `transaction_type="debit"`.
                - **filter_by_date_and_count**: Count transactions by date range. Example: "How many credit transactions in January 2025?" -> call with `start_date="2025-01-01"`, `end_date="2025-01-31"`, `transaction_type="credit"`.
                - **filter_by_type_and_sum**: Sum transactions by type. Example: "Total credit amount" -> call with `transaction_type="credit"`.
                - **filter_by_type_and_count**: Count transactions by type. Example: "How many debit transactions?" -> call with `transaction_type="debit"`.
                - **filter_by_amount_and_sum**: Sum transactions by amount range. Example: "Total amount for transactions above 1000 NGN" -> call with `min_amount=1000`.
                - **filter_by_amount_and_count**: Count transactions by amount range. Example: "Count transactions below 500 NGN" -> call with `max_amount=500`.
            """
        )
        prompt_message = prompt_template.format(
            prompt=prompt, 
            date=date.today(), 
            transactions=transactions
        )
        input_state: InputStateSchema = {
            "messages": [prompt_message],
            "transactions": request['transactions'],
            "get_audio": get_audio
        }
        config = {"configurable": {"thread_id": thread_id}}
        
        # Use asynchronous graph.invoke
        result = await graph.ainvoke(input_state, config=config)
        
        logger.info(f"Invoked graph with prompt: {prompt}, thread_id: {thread_id}, tool_calls: {result.get('tool_calls', 'None')}")
        return {
            "messages": result["messages"][-1].content,
            "audio": result.get("audio")
        }
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to invoke graph: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")