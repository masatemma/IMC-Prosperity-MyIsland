from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List, Tuple
import string
import jsonpickle

class PastData: 
    def __init__(self):
        self.market_data: Dict[str, List[Tuple[float, int, int]]] = {} #price, quantity, timestamp
        self.open_positions: Dict[str, List[Tuple[float, int]]] = {} #price, quantity
        self.prev_mid = -1

class Trader:    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        # print(state.traderData == '') <- this is true
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}

        if not state.traderData:
            past_data = PastData()
        else:
            past_data = jsonpickle.decode(state.traderData)
        
        # STARTFRUIT STRATEGY
        product = "STARFRUIT"
        orders: List[Order] = []
        ob_sf = state.order_depths[product]
        arbitrage_amount = 2
        market_making_room = 10
        prev_mid = past_data.prev_mid
        
        # Making sure a valid spread is calculated.
        
        if len(ob_sf.sell_orders) != 0 and len(ob_sf.buy_orders) != 0:
            best_ask, best_ask_amount = list(ob_sf.sell_orders.items())[0]
            best_bid, best_bid_amount = list(ob_sf.buy_orders.items())[0]
            mid_price = (best_ask + best_bid)//2
            spread = best_ask - best_bid
            
            # Market take for selling high-demand fruit
            if best_bid > prev_mid and prev_mid != -1:
                orders.append(Order(product, best_bid, -best_bid_amount))
            # Market making for selling
            elif spread >= 4 and state.position[product] > -market_making_room :
                # Selling at the floor of mid price
                orders.append(Order(product, mid_price, -arbitrage_amount))

            # Market take for buying cheap startfruit
            if best_ask < prev_mid and prev_mid != -1:
                orders.append(Order(product, best_ask, -best_ask_amount))
            
            # upPeak = isUpPeak(traderData, state.order_depths["STARFRUIT"])

        # Update mid_price
        past_data.prev_mid = mid_price
        traderData = jsonpickle.encode(past_data)
        result[product] = orders
                
        conversions = None
        # traderData = ""
        return result, conversions, traderData