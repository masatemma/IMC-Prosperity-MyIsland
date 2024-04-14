from datamodel import *
from Trader_14 import Trader
from typing import Dict, List, Tuple
import jsonpickle
import pandas as pd

timestamp = 0

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
    "ORCHIDS": Listing(
		symbol="ORCHIDS", 
		product="ORCHIDS", 
		denomination= "SEASHELLS"
	)
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
	"ORCHIDS": OrderDepth(
		buy_orders={1050: 60, 1049: 50},
		sell_orders={1055: -30, 1057: -20}
	),	
}


# own_trades = {
# 	"AMETHYSTS": [
#           Trade(
# 			symbol="AMETHYSTS",
# 			price=10000,
# 			quantity=3,
# 			buyer="SUBMISSION",
# 			seller="",
# 			timestamp=900
# 		),
#         Trade(
# 			symbol="AMETHYSTS",
# 			price=9999,
# 			quantity=1,
# 			buyer="SUBMISSION",
# 			seller="",
# 			timestamp=900
# 		)],
# 	"STARFRUIT": [],
#     "ORCHIDS": []
# }

# market_trades = {
# 	"AMETHYSTS": [
# 		Trade(
# 			symbol="AMETHYSTS",
# 			price=10001,
# 			quantity=4,
# 			buyer="",
# 			seller="",
# 			timestamp=900
# 		),
#         Trade(
# 			symbol="AMETHYSTS",
# 			price=10002,
# 			quantity=4,
# 			buyer="",
# 			seller="",
# 			timestamp=900
# 		)
# 	],
# 	"STARFRUIT": [],
#     "ORCHIDS": []
# }

own_trades = {
	"AMETHYSTS": [],
	"STARFRUIT": [],
    "ORCHIDS": []
}

market_trades = {
	"AMETHYSTS": [],
	"STARFRUIT": [],
    "ORCHIDS": []
}

position = {
	"AMETHYSTS": 0,
	"STARFRUIT": 0,
    "ORCHIDS": 0
}


observations = Observation({},{})
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
        self.portfolio: Dict[str, Tuple[int, int]] = {"AMETHYSTS":(0,0), "STARFRUIT":(0,0),"ORCHIDS": (0,0)}
        self.prev_mid = -1
        self.mid_prices: Dict[str, List[int]] = {"AMETHYSTS":[], "STARFRUIT":[], "ORCHIDS": []}
        self.humidity_rates_of_change: List[float] = []
        self.sunlight_rates_of_change: List[float] = []
        self.prev_humidity = -1    
        self.prev_sunlight = -1 
        self.sell_orchids_at = -1   
        self.humidity_entry = False
        self.sunlight_entry = False
        self.sunlight_exit = -1
        
if __name__ == '__main__':
	trader = Trader()
	df = pd.read_csv('../IMC-Prosperity-MyIsland/round2/past_data/prices_round_2_day_-1.csv', delimiter=';')
	past_trades = PastData()	       
	# past_trades.market_data = {"AMETHYSTS":[(10001, 3, 700), (10005, 4, 700), (10001, 3, 800), (10005, 4, 800)], "STARFRUIT": [], "ORCHIDS":[]}
	# past_trades.own_trades = {"AMETHYSTS":[(10001, 3, 700), (10005, 4, 700), (10001, 3, 800), (10005, 4, 800)], "STARFRUIT": [], "ORCHIDS":[]}  
	past_trades.market_data = {"AMETHYSTS":[], "STARFRUIT": [], "ORCHIDS":[]}
	past_trades.own_trades = {"AMETHYSTS":[], "STARFRUIT": [], "ORCHIDS":[]}    
	
	
	sell_peak_timestamp = []
	buy_timestamp = []
	for i, row in df.iterrows():		
		mid_price = row["ORCHIDS"] 	
        	
		con_ob = ConversionObservation(
			bidPrice=mid_price - 1, 
			askPrice=mid_price + 1, 
			transportFees=row['TRANSPORT_FEES'], 
			exportTariff=row['EXPORT_TARIFF'], 
			importTariff=row['IMPORT_TARIFF'], 
			sunlight=row['SUNLIGHT'], 
			humidity=row['HUMIDITY']
		)
		
		conversionObservations: Dict[Product, ConversionObservation] = {"ORCHIDS": con_ob}
		state.observations = Observation({}, conversionObservations)				
		state.traderData = jsonpickle.encode(past_trades, keys=True)    
		
		result, conversions, traderData = trader.run(state)
		state.timestamp += 100
		past_trades = jsonpickle.decode(traderData, keys=True) 
  
		if "ORCHIDS" in result and len(result["ORCHIDS"]) > 0:
			sell_peak_timestamp.append(state.timestamp)	   					
			print(f"sell at: {sell_peak_timestamp[-1]}")
			for order in result["ORCHIDS"]:				
				state.position["ORCHIDS"] += order.quantity	
			print(state.position["ORCHIDS"])
   
		if conversions > 0:
			state.position["ORCHIDS"] += conversions			
			buy_timestamp.append(state.timestamp)
			print(f"buy back at: {buy_timestamp[-1]}")
			print(state.position["ORCHIDS"])

	print(f"sell timestamp: {sell_peak_timestamp}")
	print(f"buy timestamp: {buy_timestamp}")	
    
    