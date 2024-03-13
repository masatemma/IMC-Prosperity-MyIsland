from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List
import string
import copy



class Trader:
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}  
    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0}  
      
        
    def calculate_vwap(self, trades) -> float:
        total_vol = 0
        total_dollar = 0
        for trade in trades:
            total_dollar += trade.quantity * trade.price
            total_vol += trade.quantity
            
        return total_dollar / total_vol
    
    
    def run(self, state: TradingState):
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """      
       
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
            all_trades: List[Trade] =[]
            fair_price = 0.0
            
            if product in state.own_trades:
                all_trades += state.own_trades[product]
                
            if product in state.market_trades:
                all_trades += state.market_trades[product]
            
            if len(all_trades) > 0:                             
                fair_price = self.calculate_vwap(all_trades)                
                print(f"fair price: {fair_price}")
                
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
                    print(f"price: {price} quantity: {quantity}") 
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

            # String value holding Trader state data required. 
                # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
                # Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData


        
        
