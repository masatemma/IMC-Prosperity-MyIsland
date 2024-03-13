from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List
import string
import copy



class Trader:
    
      
        
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
        POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}          
        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:  
            cur_position = 0 
            
            if product in state.position:       
                cur_position = state.position[product]
            
            print(f"product name: {product}")
            print(f"starting_position: {cur_position}")
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
                        order_quantity = min(POSITION_LIMIT[product] - abs(cur_position), quantity)
                        if order_quantity > 0:
                            orders.append(Order(product, price, -order_quantity))
                            cur_position += -order_quantity
                            print(f"Sell: ${price}", {quantity})
                            print(f"position: {cur_position}")
                    
                for price, quantity in sell_order_depth.items():
                    print(f"price: {price} quantity: {quantity}") 
                    # Opportunity to buy
                    # If there's a sell order depth at a price lower than the fair price, we buy
                    if price < fair_price:
                        order_quantity = min((POSITION_LIMIT[product] - abs(cur_position)), abs(quantity))
                        if order_quantity > 0:
                            orders.append(Order(product, price, order_quantity))
                            cur_position += order_quantity
                            print(f"Buy: ${price}, {quantity}")
                            print(f"position: {cur_position}")
            
        
            
            result[product] = orders

            # String value holding Trader state data required. 
                # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
                # Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData


        
        
