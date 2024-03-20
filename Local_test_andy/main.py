from datamodel import Listing, OrderDepth, Trade, TradingState, Order
from Trader import Trader, PastData
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
	"AMETHYSTS": OrderDepth(),
	"STARFRUIT": OrderDepth(),	
}

od = OrderDepth()
print(f"od.buy_orders: {type(od.buy_orders)}")

order_depths["AMETHYSTS"].buy_orders=dict({13: 7, 9: 5},)
order_depths["AMETHYSTS"].sell_orders=dict({10: -3, 11: -5, 12: -8})
order_depths["STARFRUIT"].buy_orders=dict({142: 3, 141: 5},)
order_depths["STARFRUIT"].sell_orders=dict({144: -5, 145: -8})


own_trades = {
	"AMETHYSTS": [
          Trade(
			symbol="AMETHYSTS",
			price=9,
			quantity=3,
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
	"AMETHYSTS": 0,
	"STARFRUIT": -5
}

observations = {}
traderData = ""


if __name__ == '__main__':
	trader = Trader()
	traderData = PastData()
	traderData.prev_mid = -1
	traderData = jsonpickle.encode(traderData)

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

	result, conversions, traderData = trader.run(state)
	print(f"result: {result}")    
    
    