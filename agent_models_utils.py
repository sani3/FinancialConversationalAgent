from typing import TypedDict, Optional, List, Union, Literal
from datetime import datetime
from enum import Enum

class TransactionType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"

class TransactionTrim(TypedDict):
    transactionId: str
    amount: Union[str, int, float]
    type: Literal["credit", "debit"]
    currency: str
    balance: Union[str, int, float]
    transactionDate: str  # ISO 8601 format

    def __init__(self, **kwargs):
        # Manual validation
        if not isinstance(kwargs.get('transactionId'), str):
            raise ValueError("transactionId must be a string")
        if not isinstance(kwargs.get('amount'), (str, int, float)):
            raise ValueError("amount must be a number or string")
        if kwargs.get('type') not in ['credit', 'debit']:
            raise ValueError("type must be 'credit' or 'debit'")
        if kwargs.get('currency') != 'NGN':
            raise ValueError("currency must be 'NGN'")
        if not isinstance(kwargs.get('balance'), (str, int, float)):
            raise ValueError("balance must be a number or string")
        if not isinstance(kwargs.get('transactionDate'), str):
            raise ValueError("transactionDate must be in ISO 8601 format")

class Transaction(TypedDict, total=False):
    transactionId: str
    narration: Optional[str]
    amount: Union[str, int, float]
    type: Literal["credit", "debit"]
    category: Optional[str]
    currency: str
    balance: Union[str, int, float]
    transactionDate: str  # ISO 8601 format
    description: Optional[str]
    icon: Optional[str]  # URL as string
    id: Optional[int]
    unAllocatedAmount: Optional[int]
    bankName: Optional[str]
    accountId: Optional[str]
    createdAt: Optional[str]  # ISO 8601 format
    updatedAt: Optional[str]  # ISO 8601 format
    deletedAt: Optional[str]  # ISO 8601 format

    def __init__(self, **kwargs):
        # Manual validation for required fields
        if not isinstance(kwargs.get('transactionId'), str):
            raise ValueError("transactionId must be a string")
        if not isinstance(kwargs.get('amount'), (str, int, float)):
            raise ValueError("amount must be a number or string")
        if kwargs.get('type') not in ['credit', 'debit']:
            raise ValueError("type must be 'credit' or 'debit'")
        if kwargs.get('currency') != 'NGN':
            raise ValueError("currency must be 'NGN'")
        if not isinstance(kwargs.get('balance'), (str, int, float)):
            raise ValueError("balance must be a number or string")
        if not isinstance(kwargs.get('transactionDate'), str):
            raise ValueError("transactionDate must be in ISO 8601 format")

def convert_to_float(value: Union[str, int, float]) -> float:
    if isinstance(value, str):
        return float(value.replace(',', ''))
    return float(value)

def transaction_trim_to_dict(t: TransactionTrim) -> dict:
    d = t.copy()
    d['amount'] = convert_to_float(d['amount'])
    d['balance'] = convert_to_float(d['balance'])
    return d

def transaction_to_dict(t: Transaction) -> dict:
    d = t.copy()
    d['amount'] = convert_to_float(d['amount'])
    d['balance'] = convert_to_float(d['balance'])
    return d