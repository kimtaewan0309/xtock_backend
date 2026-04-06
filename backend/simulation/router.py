from fastapi import APIRouter
from .schemas import CreateAccountRequest, BuySellRequest, AutoTradeRequest
from .service import (
    create_account,
    get_current_stock_price,
    buy_stock,
    sell_stock,
    get_portfolio,
    get_transaction_history,
    create_auto_trade,
    get_auto_trades,
)

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


@router.post("/account")
def create_simulation_account(payload: CreateAccountRequest):
    return {
        "success": True,
        "data": create_account(payload.nickname, payload.email, payload.pin),
    }


@router.get("/stock/{symbol}")
def get_stock(symbol: str):
    return {
        "success": True,
        "data": get_current_stock_price(symbol),
    }


@router.post("/buy")
def buy(payload: BuySellRequest):
    return {
        "success": True,
        "data": buy_stock(payload.user_id, payload.symbol, payload.quantity, payload.pin),
    }


@router.post("/sell")
def sell(payload: BuySellRequest):
    return {
        "success": True,
        "data": sell_stock(payload.user_id, payload.symbol, payload.quantity, payload.pin),
    }


@router.get("/portfolio/{user_id}")
def portfolio(user_id: str):
    return {
        "success": True,
        "data": get_portfolio(user_id),
    }


@router.get("/history/{user_id}")
def history(user_id: str):
    return {
        "success": True,
        "data": get_transaction_history(user_id),
    }


@router.post("/auto-trade")
def auto_trade(payload: AutoTradeRequest):
    return {
        "success": True,
        "data": create_auto_trade(
            payload.user_id,
            payload.symbol,
            payload.type,
            payload.target_price,
            payload.quantity,
        ),
    }


@router.get("/auto-trade/{user_id}")
def auto_trade_list(user_id: str):
    return {
        "success": True,
        "data": get_auto_trades(user_id),
    }