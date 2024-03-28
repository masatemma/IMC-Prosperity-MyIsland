from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List, Tuple
import string
import numpy as np
import pandas as pd
import jsonpickle
import math
from collections import Counter


class PastData: 

    def __init__(self):
        self.market_trades: Dict[str, Dict[int, List[Tuple[float, int]]]] ={} # {product:{timestamp: [(price, quantity)]}}
        self.own_trades: Dict[str, Dict[int, List[Tuple[float, int]]]] ={} # {product:{timestamp: [(price, quantity)]}}
        self.open_positions: Dict[str, List[Tuple[float, int]]] = {} #price, quantity     
        self.prev_mid = -1


class Trader:
    WINDOW_SIZE = {'AMETHYSTS': 4, 'STARFRUIT':10}   # best A: 4, S: 10    
    VWAP_WINDOW = 20
    SF_SELL = 1
    SF_BUY = 2
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}  
    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0}  
    TICK_SIZE = 1
    PD_PRICE_INDEX = 0
    PD_QUANTITY_INDEX = 1
    PD_TIMESTAMP_INDEX = 2    
    AME_THRESHOLD_MID = 10000
    AME_THRESHOLD_UP = 10004
    AME_THRESHOLD_LOW = 9996

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
    Taking the past trading data, price of an order depth, the product name and the current timestamp as arguments.
    Calculates the simple moving average value based on the number of past trades
    """
    def calculate_sma(self, past_trades: PastData, order_depth_price: float, product: str) -> float|None:  
        
        if len(past_trades.market_trades[product]) == 0:
            return None
        
        trade_count = 1
        sum_price = 0
        reverse_market_trades = sorted((past_trades.market_trades[product]).items())
        for timestamp, trades in reverse_market_trades:
            if len(trades) > 0:
                for trade in trades:                
                    sum_price += trade[self.PD_PRICE_INDEX]
                    trade_count += 1
                    if trade_count == self.WINDOW_SIZE[product]:
                        break
            if trade_count == self.WINDOW_SIZE[product]:
                break      
        
        sum_price += order_depth_price

        if trade_count >= self.WINDOW_SIZE[product]:
            mean_value = sum_price / trade_count
        else:
            return None  
            
            
        return mean_value
    
    

    """
    Taking the past trading data, price of an order depth, the product name and the current timestamp, and the desired number of timestamps as arguments.
    Calculates the simple moving average value based on the timestamps
    """
    def calculate_sma_time(self, past_trades: PastData, product: str, cur_timestamp: int, num_iter: int) -> float|None:
     
        target_timestamp = cur_timestamp - (num_iter * 100) 
   
        if len(past_trades.market_trades[product]) == 0:
            return None
        
        reverse_market_trades = list(reversed(sorted((past_trades.market_trades[product]).items())))

        if target_timestamp < 0:
            return None

        price_sums = 0
        data_point_count = 0
        for timestamp, trades in reverse_market_trades:   
            if len(trades) > 0:
                if timestamp >= target_timestamp:
                    mean_price = self.calculate_mean_price_timestamp(trades)
                    price_sums += mean_price
                    data_point_count += 1
                else:
                    break  

        if data_point_count > 0:
            return price_sums / data_point_count                

        return None
                        
    
    def calculate_mean_price_timestamp(self, timestamp_trades: List[Tuple[float, int]]):
                
        highest = max(timestamp_trades, key=lambda x: x[0])[self.PD_PRICE_INDEX]
        lowest = min(timestamp_trades, key=lambda x: x[0])[self.PD_PRICE_INDEX]

        return highest + lowest / 2

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
    Return sell orders based on SMA
    """
    def compute_sell_orders_sma(self, buy_order_depth: Dict[int, int], past_trades: PastData, product: str, cur_timestamp: int):
        orders: List[Order] = []  

        for price, quantity in buy_order_depth.items():
            current_sma = self.calculate_sma(past_trades, price, product)
            
            if current_sma == 0:
                break                
            if price >= current_sma + self.TICK_SIZE:  
                order_quantity: int
                if self.positions[product] == 0:
                    order_quantity = quantity
                else:                    
                    #order_quantity = min(self.POSITION_LIMIT[product] - abs(self.positions[product]), quantity)
                    order_quantity = min((self.POSITION_LIMIT[product] + abs(self.positions[product])), abs(quantity))
                if order_quantity > 0:
                    self.positions[product] += -order_quantity
                    orders.append(Order(product, price, -order_quantity))
        return orders
        

    """
    Return buy orders based on SMA
    """
    def compute_buy_orders_sma(self, sell_order_depth: Dict[int, int], past_trades: PastData, product: str, cur_timestamp: int):
        orders: List[Order] = []  
        # Go through each sell order depth to see if there's a good opportunity to match the order by buying 
        for price, quantity in sell_order_depth.items():
            current_sma = self.calculate_sma(past_trades, price, product)
            if current_sma == 0:
                break
            if price <= current_sma - self.TICK_SIZE:
                order_quantity: int
                if self.positions[product] == 0:
                    order_quantity = quantity
                else:                    
                    #order_quantity = min((self.POSITION_LIMIT[product] - abs(self.positions[product])), abs(quantity))
                    order_quantity = min((self.POSITION_LIMIT[product] + abs(self.positions[product])), abs(quantity))
                if order_quantity > 0:
                    self.positions[product] += order_quantity
                    orders.append(Order(product, price, order_quantity))                        
        return orders
        
    
    """
    Compute the orders for starfruit. 
    """
    def compute_starfruit_orders(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str, cur_timestamp: int):
        if (product != "STARFRUIT"):
            return None
                
        orders: List[Order] = []
        arbitrage_amount = 2
        market_making_room = 10
        prev_mid = past_trades.prev_mid

        # Making sure a valid spread is calculated.
        
        if len(sell_order_depth) != 0 and len(buy_order_depth) != 0:
            best_ask, best_ask_amount = list(sell_order_depth.items())[0]
            best_bid, best_bid_amount = list(buy_order_depth.items())[0]
            mid_price = (best_ask + best_bid)//2
            spread = best_ask - best_bid
            
            # Market take for selling high-demand fruit
            if best_bid > prev_mid and prev_mid != -1:
                orders.append(Order(product, best_bid, -best_bid_amount))
                self.positions[product] += -best_bid_amount
            # Market making for selling
            elif spread >= 4 and self.positions[product] > -market_making_room :
                # Selling at the floor of mid price
                orders.append(Order(product, mid_price, -arbitrage_amount))
                self.positions[product] += -arbitrage_amount

            # Market take for buying cheap startfruit
            if best_ask < prev_mid and prev_mid != -1:
                orders.append(Order(product, best_ask, -best_ask_amount))
                self.positions[product] += -best_ask_amount


        past_trades.prev_mid = mid_price

        return orders
    
    """
    Uses Scalping strategy to place orders
    """
    def scalping(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str, cur_timestamp: int):
        orders: List[Order] = []          

        orders += self.compute_sell_orders_sma(buy_order_depth, past_trades, product, cur_timestamp)
        orders += self.compute_buy_orders_sma(sell_order_depth, past_trades, product, cur_timestamp)

        return orders
    
    """
    Place sell orders if the bid price is above the threshold of Amethysts.
    Place buy  orders if the ask price is below the threshold of Amethysts.
    
    """
    def compute_amethysts_orders(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str) -> List[Order]:
        orders: List[Order] = []   
        if product != "AMETHYSTS":
            return 0


        temp_pos = self.positions[product]
        market_make_amount = 9 #9 best
    
        #Place a sell order if the bid price is above the threshold
        for price, quantity in buy_order_depth.items():
            if price >= self.AME_THRESHOLD_MID:
                order_quantity: int
                if self.positions[product] > 0:
                    order_quantity = min((self.POSITION_LIMIT[product] + abs(self.positions[product])), abs(quantity))
                else: 
                    order_quantity = min((self.POSITION_LIMIT[product] + self.positions[product]), abs(quantity))

                orders.append(Order(product, price, -order_quantity))
                self.positions[product] += -order_quantity  

          #Market making: sell at upper bound
        if abs(self.positions[product]) < self.POSITION_LIMIT[product]:
            if self.positions[product] > 0:            
                order_amount = min((self.POSITION_LIMIT[product] + abs(self.positions[product])), market_make_amount)                     
            else: 
                order_amount = min((self.POSITION_LIMIT[product] + self.positions[product]), market_make_amount)
            orders.append(Order(product, self.AME_THRESHOLD_UP, -order_amount))    
            self.positions[product] += -order_amount  
                   
       
        # Place a buy order if the ask price is below the threshold
        for price, quantity in sell_order_depth.items():
            if price <= self.AME_THRESHOLD_MID:
                order_quantity: int
                if temp_pos < 0:
                    order_quantity = min((self.POSITION_LIMIT[product] + abs(temp_pos)), abs(quantity))
                else: 
                    order_quantity = min((self.POSITION_LIMIT[product] - temp_pos), abs(quantity))
                
                orders.append(Order(product, price, order_quantity))
                temp_pos += order_quantity
        
      
         #Market making: buy at lower bound
        if abs(temp_pos) < self.POSITION_LIMIT[product]:
            if temp_pos < 0:
                order_amount = min((self.POSITION_LIMIT[product] + abs(temp_pos)), market_make_amount)
            else:
                order_amount = min((self.POSITION_LIMIT[product] - temp_pos), market_make_amount)          
            orders.append(Order(product, self.AME_THRESHOLD_LOW, order_amount))
            temp_pos += order_amount  


        return orders



    """
    Only method required. It takes all buy and sell orders for all symbols as an input,
    and outputs a list of orders to be sent
    """      
    def run(self, state: TradingState):

        # Orders to be placed on exchange matching engine
        result = {}
        traderData = ""  

        past_trades: PastData
        # Check if there's data in traderData
        if not state.traderData:
            past_trades = PastData()
            past_trades.market_trades = {'AMETHYSTS': {}, 'STARFRUIT': {}}   
            past_trades.own_trades = {'AMETHYSTS': {}, 'STARFRUIT': {}}    
            past_trades.open_positions = {'AMETHYSTS': [], 'STARFRUIT': []}
        else:
            past_trades = jsonpickle.decode(state.traderData)                                

        # Place orders for each product  
        for product in state.listings:              
            print(f"product name: {product}")
            
            if product not in past_trades.market_trades:
                past_trades.market_trades[product] = {}

            if product not in past_trades.open_positions:
                past_trades.open_positions[product] = []

            # Record positions
            if product in state.position:       
                self.positions[product] = state.position[product]                        
            elif product not in state.position:
                self.positions[product] = 0
                
            print(f"Starting position: {self.positions[product]}")
           
            
            # Combine all trades from own trades and market trades                
            own_trades: List[Trade] = []
            market_trades: List[Trade] = []
                        
            if product in state.own_trades:
                own_trades = state.own_trades[product]  

                # Add own_trades into open positions  
                for trade in own_trades:
                    if (trade.timestamp == state.timestamp - 100) or (trade.timestamp == state.timestamp):
                        if trade.buyer == "SUBMISSION":
                            past_trades.open_positions[product].append((trade.price, trade.quantity))
                        elif trade.seller == "SUBMISSION":
                             past_trades.open_positions[product].append((trade.price, -trade.quantity))            

                # Get rid of past own_trades that have been closed           
                self.compute_open_pos(state, past_trades, product)

                # Add past own trades into past_trades
                if int(state.timestamp) - 100 in past_trades.own_trades[product]:
                    past_trades.own_trades[product][int(state.timestamp) - 100] +=  [(trade.price, trade.quantity) for trade in own_trades if trade.timestamp == state.timestamp - 100]       
                else:
                    past_trades.own_trades[product][int(state.timestamp) - 100] =  [(trade.price, trade.quantity) for trade in own_trades if trade.timestamp == state.timestamp - 100] 

                print(f"{product} updated open positions: {len(past_trades.open_positions[product])}")
                for pos in past_trades.open_positions[product]:
                    print(f"(P: {pos[self.PD_PRICE_INDEX]} Q: {pos[self.PD_QUANTITY_INDEX]})")
                
            
            if product in state.market_trades:
                market_trades = state.market_trades[product]                                         
                # Add past market trades by bots into past_trades   
                if int(state.timestamp) - 100 in past_trades.market_trades[product]:
                    past_trades.market_trades[product][int(state.timestamp) - 100] +=  [(trade.price, trade.quantity) for trade in market_trades if trade.timestamp == state.timestamp - 100]       
                else:
                    past_trades.market_trades[product][int(state.timestamp) - 100] = [(trade.price, trade.quantity) for trade in market_trades if trade.timestamp == state.timestamp - 100]       
                


            # Initialize the list of Orders to be sent as an empty list
            orders: List[Order] = []  
            buy_order_depth: Dict[int, int]
            sell_order_depth: Dict[int, int]

            # Separate buy order depths and sell order depths
            if len(state.order_depths[product].buy_orders) > 0:       
                buy_order_depth = state.order_depths[product].buy_orders                
            if len(state.order_depths[product].sell_orders) > 0:      
                sell_order_depth = state.order_depths[product].sell_orders

            
            print(f"SMA timestamp: {self.calculate_sma_time(past_trades, product, state.timestamp, 3)}")

            # Trade differently for each product
            if product == "STARFRUIT":
                orders += self.compute_starfruit_orders(past_trades, buy_order_depth, sell_order_depth, product, state.timestamp)
                #orders += self.scalping(past_trades, buy_order_depth, sell_order_depth, product, state.timestamp)    
                
            elif product == "AMETHYSTS":                          
                orders += self.compute_amethysts_orders(past_trades, buy_order_depth, sell_order_depth, product)
                #orders += self.scalping(past_trades, buy_order_depth, sell_order_depth, product, state.timestamp)
                
            result[product] = orders
        
        # Serialize past trades into traderData
        traderData = jsonpickle.encode(past_trades) 
        
        # Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData


        
        
