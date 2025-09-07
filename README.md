# Financial Conversational Agent

Financial Conversational Agent is a FastAPI-based application that provides a financial assistant powered by LangGraph and OpenAI’s GPT-4o-mini model. The assistant processes user queries about financial transactions, offering insights such as transaction counts and totals based on date, type, or amount. It supports text-to-speech (TTS) responses and uses an in-memory SQLite database for state management via `AsyncSqliteSaver`. The application is designed for asynchronous performance and is optimized for deployment on Render.

## Features
- **Financial Query Processing**: Handles queries about transactions (e.g., "Total debit amount in August 2025") using LangGraph tools.
- **Text-to-Speech**: Converts text responses to audio using OpenAI’s TTS API when requested.
- **Asynchronous Architecture**: Utilizes `FastAPI`, `AsyncSqliteSaver`, `openai.AsyncClient`, and `aiofiles` for efficient async processing.
- **In-Memory SQLite**: Stores conversation state in an in-memory SQLite database for fast, stateless operation.
- **Render Deployment**: Configured for deployment on Render with `render.yaml` and `Dockerfile`.
- **Testing Notebook**: Includes a Jupyter notebook (`consume_api.ipynb`) for testing the API locally or on Render.

## Prerequisites
- **Python**: 3.10 or higher
- **Dependencies**: Listed in `requirements.txt`
- **OpenAI API Key**: Required for LLM and TTS functionality
- **Docker**: For containerized deployment (optional for local testing)
- **Render Account**: For cloud deployment
- **Jupyter Notebook**: For running `consume_api.ipynb`

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/FinancialConversationalAgent.git
   cd FinancialConversationalAgent
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**:
   - Create a `.env` file in the project root:
     ```bash
     echo "OPENAI_API_KEY=your-openai-api-key" > .env
     ```
   - Replace `your-openai-api-key` with your OpenAI API key.

## Running Locally
1. **Start the FastAPI Application**:
   ```bash
   uvicorn main:app --reload
   ```
   - The application will be available at `http://127.0.0.1:8000`.

2. **Verify Endpoints**:
   - **Index**: `GET /` – Returns "Server is up and running".
   - **Health Check**: `GET /health` – Returns `{"status": "healthy"}`.
   - **Conversation**: `POST /conversation` – Processes financial queries (see Usage section).

3. **Test with Jupyter Notebook**:
   - Open `consume_api.ipynb` in Jupyter Notebook:
     ```bash
     jupyter notebook consume_api.ipynb
     ```
   - Update the `url` variable to `http://127.0.0.1:8000/conversation`.
   - Run the cells to send sample queries (e.g., "How many debit transactions do I have?") and verify responses.

## Deploying on Render
1. **Push to GitHub**:
   - Ensure all files (`main.py`, `agent.py`, `agent_tools_utils.py`, `agent_models_utils.py`, `requirements.txt`, `render.yaml`, `Dockerfile`, `consume_api.ipynb`) are committed to your GitHub repository.

2. **Configure Render**:
   - Create a new Blueprint Instance in the Render dashboard.
   - Connect your GitHub repository and select the branch containing the project.
   - Set the `OPENAI_API_KEY` environment variable in the Render dashboard under Environment settings.

3. **Deploy**:
   - Deploy the application using the `render.yaml` configuration, which specifies a Docker service with Gunicorn and a 10-minute timeout.
   - The application will be available at `https://your-app-name.onrender.com`.

4. **Test on Render**:
   - In `consume_api.ipynb`, update the `url` variable to `https://your-app-name.onrender.com/conversation`.
   - Run the notebook cells to send queries and verify responses, including text and optional audio output.

## Usage
The `/conversation` endpoint accepts POST requests with the following JSON payload:

```json
{
  "prompt": "How many debit transactions do I have?",
  "thread_id": "unique-thread-id",
  "transactions": [
    {
      "transactionId": "tx1",
      "amount": 1000.00,
      "type": "debit",
      "currency": "NGN",
      "balance": 5000.00,
      "transactionDate": "2025-08-01T12:00:00Z"
    }
  ],
  "get_audio": false
}
```

- **prompt**: The financial query (e.g., "Total debit amount in August 2025").
- **thread_id**: A unique identifier for the conversation thread.
- **transactions**: A list of transaction objects (see schema in `agent.py`).
- **get_audio**: Boolean to request TTS audio output (default: `false`).

**Response**:
```json
{
  "messages": "You have 1 debit transaction.",
  "audio": null
}
```

**Example Queries**:
- "Total debit amount in August 2025"
- "How many credit transactions in January 2025?"
- "Show transactions above 1000 NGN"
- "Total credit amount from January to March 2025"

## Project Structure
- **`main.py`**: FastAPI application with endpoints (`/`, `/health`, `/conversation`) and startup/shutdown events for `aiosqlite` initialization.
- **`agent.py`**: Defines the LangGraph workflow, including the assistant, speaker, and tools nodes, with `AsyncSqliteSaver` for checkpointing.
- **`agent_tools_utils.py`**: Contains tools for filtering and aggregating transactions (e.g., `filter_by_date_and_sum`).
- **`agent_models_utils.py`**: Defines data models like `TransactionTrim`.
- **`requirements.txt`**: Lists dependencies, including `fastapi`, `langgraph`, `openai`, and `aiosqlite`.
- **`render.yaml`**: Render configuration for Docker deployment.
- **`Dockerfile`**: Builds the container with Gunicorn and dependencies.
- **`consume_api.ipynb`**: Jupyter notebook for testing the API locally or on Render.

## Troubleshooting
- **Error: `OPENAI_API_KEY` not set**:
  - Ensure the `OPENAI_API_KEY` is set in the `.env` file locally or in the Render dashboard.
- **Error: `threads can only be started once`**:
  - Verify that `agent.py` and `main.py` match the latest versions, which create a fresh `aiosqlite` connection per graph instance.
- **Slow Responses**:
  - Check if the transaction list is large; consider client-side filtering in `consume_api.ipynb`.
  - Increase the Gunicorn timeout in `render.yaml` and `Dockerfile` (e.g., `--timeout 6000`) if needed, after confirming with Render support.
- **Render Deployment Issues**:
  - Check Render logs for errors related to dependencies or environment variables.
  - Ensure `requirements.txt` includes all necessary packages.

## Contributing
Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## License
This project is licensed under the MIT License.
