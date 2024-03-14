from datamodel import Listing, OrderDepth, Trade, TradingState
from Trader_Test_2 import Trader
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
		buy_orders={10: 7, 9: 5},
		sell_orders={10: -3, 11: -4, 12: -8}
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
			price=9,
			quantity=2,
			buyer="SUBMISSION",
			seller="",
			timestamp=900
		),
        Trade(
			symbol="AMETHYSTS",
			price=20,
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
			price=11,
			quantity=4,
			buyer="",
			seller="",
			timestamp=900
		),
        Trade(
			symbol="AMETHYSTS",
			price=20,
			quantity=4,
			buyer="",
			seller="",
			timestamp=900
		)
	],
	"STARFRUIT": []
}

position = {
	"AMETHYSTS": 3,
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
        self.market_data: Dict[str:List[Tuple]] = {}
        self.open_positions: Dict[str:List[Tuple]] = {}      


if __name__ == '__main__':
	trader = Trader()
	past_trades = PastData()	
	past_trades.market_data = {'AMETHYSTS': [(1,10)], 'STARFRUIT': [(1,3)]}         
	state.traderData = jsonpickle.encode(past_trades)
	trader.run(state)
    