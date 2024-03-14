from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List, Tuple
import string
import numpy as np
import jsonpickle


class PastData: 

    def __init__(self):
        self.market_data: Dict[str, List[Tuple[float, int]]] = {}
        self.open_positions: Dict[str, List[Tuple[float, int]]] = {}      


class Trader:
    WINDOW_SIZE = 4
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}  
    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0}  
    TICK_SIZE = 1
    PD_PRICE_INDEX = 0
    PD_QUANTITY_INDEX = 1

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
    def calculate_sma(self, past_market_data: List[Tuple[float, int]], order_depth_price: float) -> float:  
            if len(past_market_data) + 1 < self.WINDOW_SIZE:
                return 0.0
                      
            past_trades_sum_price = sum(data[self.PD_PRICE_INDEX] for data in past_market_data)
        
            past_trades_sum_price += order_depth_price
            mean_value = float(past_trades_sum_price / (len(past_market_data) + 1))

            return mean_value                 
    
    """
    Taking the trading state, past market data and the product name as arguments
    Compute the open positions for that particular product
    """
    def compute_open_pos(self, state: TradingState, past_trades: PastData, product: str):
        valid_op = []
        position_count = 0
        if state.position[product] > 0:
            # Iterate the own_trades from the most recent one
            for trade in reversed(past_trades.open_positions[product]):
                position_count += trade[self.PD_QUANTITY_INDEX]
                valid_op_trade: Tuple
                if position_count < state.position[product]:
                    valid_op_trade = (trade[self.PD_PRICE_INDEX], trade[self.PD_QUANTITY_INDEX])
                elif position_count > state.position[product]:
                    pos_diff = position_count - state.position[product]
                    valid_op_trade = (trade[self.PD_PRICE_INDEX], trade[self.PD_QUANTITY_INDEX] - pos_diff)                
                elif position_count == state.position[product]:
                    valid_op_trade = (trade[self.PD_PRICE_INDEX], trade[self.PD_QUANTITY_INDEX])

                valid_op.insert(0, valid_op_trade)
            # Update the open positions                            
        elif state.position[product] < 0:
            # Iterate the own_trades from the most recent one
            for trade in reversed(past_trades.open_positions[product]):
                position_count += trade[self.PD_QUANTITY_INDEX]
                valid_op_trade: Tuple
                if position_count > state.position[product]:
                    valid_op_trade = (trade[self.PD_PRICE_INDEX], trade[self.PD_QUANTITY_INDEX])
                elif position_count < state.position[product]:
                    pos_diff = position_count - state.position[product]
                    valid_op_trade = (trade[self.PD_PRICE_INDEX], trade[self.PD_QUANTITY_INDEX] - pos_diff)
                elif position_count == state.position[product]:
                    valid_op_trade = (trade[self.PD_PRICE_INDEX], trade[self.PD_QUANTITY_INDEX]) 

                valid_op.insert(0, valid_op_trade)
       
        past_trades.open_positions[product] = valid_op

    """
    Only method required. It takes all buy and sell orders for all symbols as an input,
    and outputs a list of orders to be sent
    """      
    def run(self, state: TradingState):

        # Orders to be placed on exchange matching engine
        result = {}
        traderData = ""  

        # Record past market trades prices
        past_trades = PastData()
        past_trades.market_data = {'AMETHYSTS': [], 'STARFRUIT': []}     
        past_trades.open_positions = {'AMETHYSTS': [], 'STARFRUIT': []}        

        # Check if there's data in traderData
        if len(state.traderData) > 0:
            past_trades = jsonpickle.decode(state.traderData)                                

        # Decide trade for each product  
        for product in state.listings:              

            if product not in past_trades.market_data:
                past_trades.market_data[product] = []
            if product not in past_trades.open_positions:
                past_trades.open_positions[product] = []

            # Record positions
            if product in state.position:       
                self.positions[product] = state.position[product]
            
            print(f"product name: {product}")
            print(f"starting_position: {self.positions[product]}")
           
            
            # Combine all trades from own trades and market trades to calculate vwap                  
            all_trades: List[Trade] = []
            
            if product in state.own_trades:
                all_trades += state.own_trades[product]  

                # Add own_trades into open positions  
                for trade in all_trades:
                    if (trade.timestamp == state.timestamp - 100) or (trade.timestamp == state.timestamp):
                        if trade.buyer == "SUBMISSION":
                            past_trades.open_positions[product].append((trade.price, trade.quantity))
                        elif trade.seller == "SUBMISSION":
                             past_trades.open_positions[product].append((trade.price, -trade.quantity))            

                print(f"{product} open positions: {len(past_trades.open_positions[product])}")
                for pos in past_trades.open_positions[product]:
                    print(f"(P: {pos[self.PD_PRICE_INDEX]} Q: {pos[self.PD_QUANTITY_INDEX]})")


                # Get rid of past own_trades that gave been closed           
                self.compute_open_pos(state, past_trades, product)
                print(f"{product} updated open positions: {len(past_trades.open_positions[product])}")
                for pos in past_trades.open_positions[product]:
                    print(f"(P: {pos[self.PD_PRICE_INDEX]} Q: {pos[self.PD_QUANTITY_INDEX]})")
                
            
            if product in state.market_trades:
                all_trades += state.market_trades[product]                         

            if len(all_trades) > 0:                                                                    

                # Add the past market trades and own trades into traderData
                past_trades.market_data[product] += [(trade.price, trade.quantity) for trade in all_trades if trade.timestamp == state.timestamp - 100 or trade.timestamp == state.timestamp]    
                
                if len(past_trades.market_data[product]) > self.WINDOW_SIZE:         
                    past_trades.market_data[product] = past_trades.market_data[product][-self.WINDOW_SIZE:]
                
                print(f"{product} Past Market Data")
                for trade in past_trades.market_data[product]:
                    print(trade)    

                      
            # Initialize the list of Orders to be sent as an empty list
            orders: List[Order] = []  
            buy_order_depth: Dict[int, int]
            sell_order_depth: Dict[int, int]

            #order depth for the product     
            if len(state.order_depths[product].buy_orders) > 0:       
                buy_order_depth = state.order_depths[product].buy_orders
            if len(state.order_depths[product].sell_orders) > 0:      
                sell_order_depth = state.order_depths[product].sell_orders

            current_sma: float    
            # Go through each buy order depth to see if there's a good opportunity to match the order by selling 
            for price, quantity in buy_order_depth.items():
                current_sma = self.calculate_sma(past_trades.market_data[product], price)
                #print(f"current SMA: {current_sma}")
                if current_sma == 0:
                    break                
                if price >= current_sma + self.TICK_SIZE:                    
                    order_quantity = min(self.POSITION_LIMIT[product] - abs(self.positions[product]), quantity)
                    if order_quantity > 0:
                        orders.append(Order(product, price, -order_quantity))
                        self.positions[product] += -order_quantity
                        print(f"Sell: ${price}", {quantity})
                        print(f"position: {self.positions[product]}")
            
            # Go through each sell order depth to see if there's a good opportunity to match the order by buying 
            for price, quantity in sell_order_depth.items():
                    current_sma = self.calculate_sma(past_trades.market_data[product], price)
                    if current_sma == 0:
                        break
                    if price <= current_sma - self.TICK_SIZE:
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


        
        
