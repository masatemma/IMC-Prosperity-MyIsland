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
    WINDOW_SIZE = {'AMETHYSTS': 4, 'STARFRUIT': 6}   # best A: 4, S: 6    
    VWAP_WINDOW = 30
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}  
    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0}  
    TICK_SIZE = 1
    PD_PRICE_INDEX = 0
    PD_QUANTITY_INDEX = 1

    """
    Taking the past trades as an argument,including own_trades and market_trades for a specific product.
    Return the VWAP value
    """
    def calculate_vwap(self, past_market_data: List[Tuple[float, int]], product:str) -> float:  

        if len(past_market_data) < self.WINDOW_SIZE[product]:   
            return 0.0
           
        total_vol = 0
        total_dollar = 0
        for data in past_market_data[-self.VWAP_WINDOW: ]:
            total_dollar += data[self.PD_PRICE_INDEX] * abs(data[self.PD_QUANTITY_INDEX])
            total_vol += abs(data[self.PD_QUANTITY_INDEX])
    
            
        return total_dollar / total_vol
    
    
    """
    Taking the past trading prices and the window size arguments.
    Returns the simple moving average value. 
    """
    def calculate_sma(self, past_market_data: List[Tuple[float, int]], order_depth_price: float, product: str) -> float:  
            if len(past_market_data) + 1 < self.WINDOW_SIZE[product]:
                return 0.0
                      
            past_trades_sum_price = sum(data[self.PD_PRICE_INDEX] for data in past_market_data[-self.WINDOW_SIZE[product]:])

            print(f"Past market data window size: {past_market_data[-self.WINDOW_SIZE[product]:]}")
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
        valid_op_trade: Tuple
        if state.position[product] > 0:
            # Iterate the own_trades from the most recent one
            for trade in reversed(past_trades.open_positions[product]):
                if trade[self.PD_QUANTITY_INDEX] > 0:           
                    position_count += trade[self.PD_QUANTITY_INDEX]

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
                if trade[self.PD_QUANTITY_INDEX] < 0:           
                    position_count += trade[self.PD_QUANTITY_INDEX]
   
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
    Return buy orders based on SMA
    """
    def compute_sell_orders_sma(self, orders: List[Order], buy_order_depth: Dict[int, int], past_trades: PastData, product: str):
        
        for price, quantity in buy_order_depth.items():
            current_sma = self.calculate_sma(past_trades.market_data[product], price, product)
            #print(f"current SMA: {current_sma}")
            if current_sma == 0:
                break                
            if price >= current_sma + self.TICK_SIZE:     
                order_quantity = min(abs(self.positions[product]), quantity)

                #order_quantity = min(self.POSITION_LIMIT[product] - abs(self.positions[product]), quantity)
                if order_quantity > 0:
                    self.positions[product] += -order_quantity
                    orders.append(Order(product, price, -order_quantity))
                    
                    print(f"Sell: ${price}", {quantity})
                    print(f"position: {self.positions[product]}")

    """
    Return sell orders based on SMA
    """
    def compute_buy_orders_sma(self, orders: List[Order], sell_order_depth: Dict[int, int], past_trades: PastData, product: str):
        # Go through each sell order depth to see if there's a good opportunity to match the order by buying 
        for price, quantity in sell_order_depth.items():
                current_sma = self.calculate_sma(past_trades.market_data[product], price, product)
                if current_sma == 0:
                    break
                if price <= current_sma - self.TICK_SIZE:
                    order_quantity = min(abs(self.positions[product]), abs(quantity))
                    #order_quantity = min((self.POSITION_LIMIT[product] - abs(self.positions[product])), abs(quantity))
                    if order_quantity > 0:
                        self.positions[product] += order_quantity
                        orders.append(Order(product, price, order_quantity))                        
                        print(f"Buy: ${price}, {quantity}")
                        print(f"position: {self.positions[product]}")
            

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

        # Place orders for each product  
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


                # Get rid of past own_trades that have been closed           
                self.compute_open_pos(state, past_trades, product)
                print(f"{product} updated open positions: {len(past_trades.open_positions[product])}")
                for pos in past_trades.open_positions[product]:
                    print(f"(P: {pos[self.PD_PRICE_INDEX]} Q: {pos[self.PD_QUANTITY_INDEX]})")
                
            
            if product in state.market_trades:
                all_trades += state.market_trades[product]                         

            if len(all_trades) > 0:                                                                    

                # Add the past market trades and own trades into traderData
                past_trades.market_data[product] += [(trade.price, trade.quantity) for trade in all_trades if trade.timestamp == state.timestamp - 100 or trade.timestamp == state.timestamp]    
                
                # if len(past_trades.market_data[product]) > self.WINDOW_SIZE[product]:         
                #     past_trades.market_data[product] = past_trades.market_data[product][-self.WINDOW_SIZE[product]:]
                
                print(f"{product} Past Market Data")
                for trade in past_trades.market_data[product][-self.WINDOW_SIZE[product]:]:
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

            
            fair_price = self.calculate_vwap(past_trades.market_data[product], product)    

            
            if (product not in state.position) and (fair_price != 0.0):
                # if the order position is at zero, we place an order based on the best bid or best ask
                best_bid = max(buy_order_depth)
                best_ask = min(sell_order_depth)
                #orders.append(Order(product, int(fair_price), 10))

                if abs(fair_price - best_bid) > abs(fair_price - best_ask):                   
                    order_quantity = min(self.POSITION_LIMIT[product] - abs(self.positions[product]), buy_order_depth[best_bid])
                    orders.append(Order(product, best_bid, -order_quantity))
                    self.positions[product] += -order_quantity

                elif abs(fair_price - best_bid) < abs(fair_price - best_ask):
                    order_quantity = min((self.POSITION_LIMIT[product] - abs(self.positions[product])), sell_order_depth[best_ask])
                    orders.append(Order(product, best_ask, order_quantity))
                    self.positions[product] += order_quantity

            if product in state.position and state.position[product] > 0:
                # Go through each buy order depth to see if there's a good opportunity to match the order by selling 
                self.compute_sell_orders_sma(orders, buy_order_depth, past_trades, product)
            
            elif product in state.position and state.position[product] < 0:
                # Go through each sell order depth to see if there's a good opportunity to match the order by buying 
                self.compute_buy_orders_sma(orders, sell_order_depth, past_trades, product)
               
            
            result[product] = orders
        
        # Serialize past trades into traderData
        traderData = jsonpickle.encode(past_trades) 
        
        # Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData


        
        
