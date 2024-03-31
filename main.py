from datamodel import Listing, OrderDepth, Trade, TradingState, Order
from Trader_Test_8 import Trader
from typing import Dict, List, Tuple
import jsonpickle

timestamp = 1000

listings = {
	"AMETHYSTS": Listing(
		symbol="AMETHYSTS", 
		product="AMETHYSTS", 
		denomination= "SEASHELLS"
	),
	"STARFRUIT": Listing(
		symbol="STARFRUIT", 
		product="STARFRUIT", 
		denomination= "SEASHELLS"
	),
}

order_depths = {
	"AMETHYSTS": OrderDepth(
		buy_orders={10000: 7, 10001: 5},
		sell_orders={10003: -3, 10002: -5, 10000: -8}
	),
	"STARFRUIT": OrderDepth(
		buy_orders={142: 3, 141: 5},
		sell_orders={144: -5, 145: -8}
	),	
}


own_trades = {
	"AMETHYSTS": [
          Trade(
			symbol="AMETHYSTS",
			price=10000,
			quantity=3,
			buyer="SUBMISSION",
			seller="",
			timestamp=900
		),
        Trade(
			symbol="AMETHYSTS",
			price=9999,
			quantity=1,
			buyer="SUBMISSION",
			seller="",
			timestamp=900
		)],
	"STARFRUIT": []
}

market_trades = {
	"AMETHYSTS": [
		Trade(
			symbol="AMETHYSTS",
			price=10001,
			quantity=4,
			buyer="",
			seller="",
			timestamp=900
		),
        Trade(
			symbol="AMETHYSTS",
			price=10002,
			quantity=4,
			buyer="",
			seller="",
			timestamp=900
		)
	],
	"STARFRUIT": []
}

position = {
	"AMETHYSTS": 0,
	"STARFRUIT": -5
}

observations = {}
traderData = ""


state = TradingState(
	traderData,
	timestamp,
    listings,
	order_depths,
	own_trades,
	market_trades,
	position,
	observations
)

class PastData: 

    def __init__(self):
        self.market_data: Dict[str, List[Tuple[float, int, int]]] = {} #price, quantity, timestamp
        self.own_trades: Dict[str, List[Tuple[float, int, int]]] = {} #price, quantity, timestamp
        self.open_positions: Dict[str, List[Tuple[float, int]]] = {} #price, quantity     
        self.prev_mid = -1
        
if __name__ == '__main__':
    trader = Trader()
    past_trades = PastData()	       
    past_trades.market_data = {"AMETHYSTS":[(10001, 3, 700), (10005, 4, 700), (10001, 3, 800), (10005, 4, 800)], "STARFRUIT": []}
    past_trades.own_trades = {"AMETHYSTS":[(10001, 3, 700), (10005, 4, 700), (10001, 3, 800), (10005, 4, 800)], "STARFRUIT": []}
    state.traderData = jsonpickle.encode(past_trades, keys=True)    

    result, conversions, traderData = trader.run(state)
    print(result)    
    
    