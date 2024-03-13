from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List, Tuple
import string
import numpy as np
import jsonpickle


class PastData: 

    def __init__(self):
        self.market_data: Dict[str:List[Tuple]] = {}
        self.open_positions: Dict[str:List[Tuple]] = {}      


class Trader:
    WINDOW_SIZE = 10
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}  
    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0}  


    """
    Taking the past trades as an argument,including own_trades and market_trades for a specific product.
    Return the VWAP value
    """
    def calculate_vwap(self, trades) -> float:        
        total_vol = 0
        total_dollar = 0
        for trade in trades:
            total_dollar += trade.quantity * trade.price
            total_vol += trade.quantity
            
        return total_dollar / total_vol
    
    
    """
    Taking the past trading prices and the window size arguments.
    Returns the simple moving average value. 
    """
    def calculate_sma(self, past_trade_prices: list[float]):
            return np.mean(past_trade_prices)
    

    """
    Only method required. It takes all buy and sell orders for all symbols as an input,
    and outputs a list of orders to be sent
    """      
    def run(self, state: TradingState):
        
        traderData = ""
        # Record past market trades prices
        past_trades = PastData()
        past_trades.market_data = {'AMETHYSTS': [Tuple], 'STARFRUIT': [Tuple]}     
        past_trades.open_positions = {'AMETHYSTS': [Tuple], 'STARFRUIT': [Tuple]}
        

        # Check if there's data in traderData
        if len(state.traderData) > 0:
            past_trades = jsonpickle.decode(state.traderData)
       
        # Orders to be placed on exchange matching engine
        result = {}
    
        for product in state.listings:              
            
            if product in state.position:       
                self.positions[product] = state.position[product]
            
            print(f"product name: {product}")
            print(f"starting_position: {self.positions[product]}")
                        
            #order depth for the product            
            buy_order_depth = state.order_depths[product].buy_orders
            sell_order_depth = state.order_depths[product].sell_orders
                        
            # Initialize the list of Orders to be sent as an empty list
            orders: List[Order] = []          
            
            # Combine all trades from own trades and market trades to calculate vwap                  
            all_trades: List[Trade] = []
            fair_price = 0.0
            
            if product in state.own_trades:
                all_trades += state.own_trades[product]    
                #past_trades.open_positions[product] += [(trade.price, trade.quantity) for trade in all_trades]

            if product in state.market_trades:
                all_trades += state.market_trades[product]
                for trade in state.market_trades[product]:
                    print(f"Trade price: {trade.price} Trade quantity: {trade.quantity}")
            
            if len(all_trades) > 0:                             
                fair_price = self.calculate_vwap(all_trades)    
                
                print(f"fair price: {fair_price}")

                # Add the past market trades and own trades into traderData
                past_trades.market_data[product] += [(trade.price, trade.quantity) for trade in all_trades if trade.timestamp == state.timestamp - 100 or trade.timestamp == state.timestamp]    

                if len(past_trades.market_data[product]) > self.WINDOW_SIZE:         
                    past_trades.market_data[product] = past_trades.market_data[product][-self.WINDOW_SIZE:]
                
                for trade in past_trades.market_data[product]:
                    print(trade)                                       
                
                
            if fair_price > 0:
                for price, quantity in buy_order_depth.items():
                    #print(f"price: {price} quantity: {quantity}") 
                    # Opportunity to sell      
                    # If there's a buy order depth at a price higher than the fair price, we sell                                                                                                         
                    if price > fair_price:                    
                        order_quantity = min(self.POSITION_LIMIT[product] - abs(self.positions[product]), quantity)
                        if order_quantity > 0:
                            orders.append(Order(product, price, -order_quantity))
                            self.positions[product] += -order_quantity
                            print(f"Sell: ${price}", {quantity})
                            print(f"position: {self.positions[product]}")
                    
                for price, quantity in sell_order_depth.items():
                    # print(f"price: {price} quantity: {quantity}") 
                    # Opportunity to buy
                    # If there's a sell order depth at a price lower than the fair price, we buy
                    if price < fair_price:
                        order_quantity = min((self.POSITION_LIMIT[product] - abs(self.positions[product])), abs(quantity))
                        if order_quantity > 0:
                            orders.append(Order(product, price, order_quantity))
                            self.positions[product] += order_quantity
                            print(f"Buy: ${price}, {quantity}")
                            print(f"position: {self.positions[product]}")
            
        
            
            result[product] = orders
        
        # Serialize past trades into traderData
        traderData = jsonpickle.encode(past_trades) 
        
        # Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData


        
        
