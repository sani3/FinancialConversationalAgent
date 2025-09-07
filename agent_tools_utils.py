import json
from datetime import datetime, timezone
from langchain_core.tools import tool
from typing import TypedDict, Optional, List, Union, Literal
from agent_models_utils import TransactionTrim, TransactionType
import re

class ToolResult(TypedDict):
    success: bool
    data: Optional[Union[List[dict], float, int]]
    error: Optional[str]

class TransactionDateFilterInput(TypedDict, total=False):
    start_date: Optional[str]  # Format: YYYY-MM-DD
    end_date: Optional[str]  # Format: YYYY-MM-DD

class TransactionAmountFilterInput(TypedDict, total=False):
    min_amount: Optional[float]
    max_amount: Optional[float]

class TransactionTypeFilterInput(TypedDict, total=False):
    transaction_type: Optional[TransactionType]

@tool
def filter_by_date_and_sum(transactions: List[TransactionTrim], start_date: Optional[str] = None, end_date: Optional[str] = None) -> float:
    """
    Filter transactions by date range and sum their amounts.

    Args:
        transactions: List of TransactionTrim dictionaries.
        start_date: Optional start date in YYYY-MM-DD format.
        end_date: Optional end date in YYYY-MM-DD format. Defaults to start_date if not provided.

    Returns:
        float: Sum of amounts for filtered transactions.
    """
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    if start_date and not re.match(date_pattern, start_date):
        raise ValueError("start_date must be in YYYY-MM-DD format")
    if end_date and not re.match(date_pattern, end_date):
        raise ValueError("end_date must be in YYYY-MM-DD format")

    filtered = transactions
    if start_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc) if end_date else start
        filtered = [
            t for t in transactions
            if start <= datetime.fromisoformat(t['transactionDate'].replace('Z', '+00:00')).astimezone(timezone.utc) <= end
        ]
    return sum(float(t['amount']) for t in filtered)

@tool
def filter_by_date_and_count(transactions: List[TransactionTrim], start_date: Optional[str] = None, end_date: Optional[str] = None) -> int:
    """
    Filter transactions by date range and count them.

    Args:
        transactions: List of TransactionTrim dictionaries.
        start_date: Optional start date in YYYY-MM-DD format.
        end_date: Optional end date in YYYY-MM-DD format. Defaults to start_date if not provided.

    Returns:
        int: Number of filtered transactions.
    """
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    if start_date and not re.match(date_pattern, start_date):
        raise ValueError("start_date must be in YYYY-MM-DD format")
    if end_date and not re.match(date_pattern, end_date):
        raise ValueError("end_date must be in YYYY-MM-DD format")

    filtered = transactions
    if start_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc) if end_date else start
        filtered = [
            t for t in transactions
            if start <= datetime.fromisoformat(t['transactionDate'].replace('Z', '+00:00')).astimezone(timezone.utc) <= end
        ]
    return len(filtered)

@tool
def filter_by_type_and_sum(transactions: List[TransactionTrim], transaction_type: Optional[TransactionType] = None) -> float:
    """
    Filter transactions by type and sum their amounts.

    Args:
        transactions: List of TransactionTrim dictionaries.
        transaction_type: Optional type of transaction to filter ('credit' or 'debit').

    Returns:
        float: Sum of amounts for filtered transactions or 0.0 if no transactions match.
    """
    filtered = transactions
    if transaction_type:
        filtered = [t for t in transactions if t['type'] == transaction_type.value]
    return sum(float(t['amount']) for t in filtered) if filtered else 0.0

@tool
def filter_by_type_and_count(transactions: List[TransactionTrim], transaction_type: Optional[TransactionType] = None) -> int:
    """
    Filter transactions by type and count them.

    Args:
        transactions: List of TransactionTrim dictionaries.
        transaction_type: Optional type of transaction to filter ('credit' or 'debit').

    Returns:
        int: Number of filtered transactions.
    """
    filtered = transactions
    if transaction_type:
        filtered = [t for t in transactions if t['type'] == transaction_type.value]
    return len(filtered)

@tool
def filter_by_amount_and_sum(transactions: List[TransactionTrim], min_amount: Optional[float] = None, max_amount: Optional[float] = None) -> float:
    """
    Filter transactions by amount range and sum their amounts.

    Args:
        transactions: List of TransactionTrim dictionaries.
        min_amount: Optional minimum transaction amount.
        max_amount: Optional maximum transaction amount.

    Returns:
        float: Sum of amounts for filtered transactions.
    """
    if min_amount is not None and min_amount < 0:
        raise ValueError("min_amount must be non-negative")
    if max_amount is not None and max_amount < 0:
        raise ValueError("max_amount must be non-negative")

    filtered = transactions
    if min_amount is not None:
        filtered = [t for t in filtered if float(t['amount']) >= min_amount]
    if max_amount is not None:
        filtered = [t for t in filtered if float(t['amount']) <= max_amount]
    return sum(float(t['amount']) for t in filtered)

@tool
def filter_by_amount_and_count(transactions: List[TransactionTrim], min_amount: Optional[float] = None, max_amount: Optional[float] = None) -> int:
    """
    Filter transactions by amount range and count them.

    Args:
        transactions: List of TransactionTrim dictionaries.
        min_amount: Optional minimum transaction amount.
        max_amount: Optional maximum transaction amount.

    Returns:
        int: Number of filtered transactions.
    """
    if min_amount is not None and min_amount < 0:
        raise ValueError("min_amount must be non-negative")
    if max_amount is not None and max_amount < 0:
        raise ValueError("max_amount must be non-negative")

    filtered = transactions
    if min_amount is not None:
        filtered = [t for t in filtered if float(t['amount']) >= min_amount]
    if max_amount is not None:
        filtered = [t for t in filtered if float(t['amount']) <= max_amount]
    return len(filtered)