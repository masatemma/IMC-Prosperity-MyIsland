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
        self.market_data: Dict[str, List[Tuple[float, int, int]]] = {} #price, quantity, timestamp
        self.own_trades: Dict[str, List[Tuple[float, int, int]]] = {} #price, quantity, timestamp
        self.open_positions: Dict[str, List[Tuple[float, int]]] = {} #price, quantity     
        self.prev_mid = -1


class Trader:    
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}  
    WINDOW_SIZE = {'AMETHYSTS': 4, 'STARFRUIT': 10}  
    WINDOW_SIZE_TIME = {'AMETHYSTS': 25, 'STARFRUIT': 25} 
    WINDOW_SIZE_VOL = {'AMETHYSTS': 5, 'STARFRUIT': 15}
    VWAP_WINDOW = 20
    PAST_DATA_MAX = 10000
    TICK_SIZE = 1
    PD_PRICE_INDEX = 0
    PD_QUANTITY_INDEX = 1
    PD_TIMESTAMP_INDEX = 2    
    AME_THRESHOLD_MID = 10000
    AME_THRESHOLD_UP = 10004
    AME_THRESHOLD_LOW = 9996

    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0}  
    cur_timestamp = 0
    

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
    def calculate_sma(self, past_trades: PastData, order_depth_price: float, product: str) -> float:  
            
        if order_depth_price != 0 and len(past_trades.market_data[product]) + 1 < self.WINDOW_SIZE[product]:
            return 0
        elif  order_depth_price == 0 and len(past_trades.market_data[product]) < self.WINDOW_SIZE[product]:
            return 0
                    
        past_trades_sum_price = sum(data[self.PD_PRICE_INDEX] for data in past_trades.market_data[product][-self.WINDOW_SIZE[product]: ])
        
        mean_value: float
        if order_depth_price != 0:
            past_trades_sum_price += order_depth_price
            mean_value = float(past_trades_sum_price / (len(past_trades.market_data[product][-self.WINDOW_SIZE[product]:]) + 1))
        else:
            mean_value = float(past_trades_sum_price / (len(past_trades.market_data[product][-self.WINDOW_SIZE[product]:])))
            
        return mean_value 
    
    

    """
    Taking the past trading data, price of an order depth, the product name and the current timestamp, and the desired number of timestamps as arguments.
    Calculates the simple moving average value based on the timestamps
    """
    def calculate_sma_time(self, past_trades: PastData, product: str) -> float:
        target_timestamp = self.cur_timestamp - (self.WINDOW_SIZE_TIME[product] * 100) 
   
        if len(past_trades.market_data[product]) == 0:
            return 0
        
        trades_by_timestamp = sorted(self.convert_to_timestamp_dict(past_trades, product).items())
        reverse_market_trades = reversed(trades_by_timestamp)       

        if target_timestamp < 0:
            return 0

        price_sums = 0
        data_point_count = 0
        for timestamp, trades in reverse_market_trades:   
            if len(trades) > 0:
                if int(timestamp) >= target_timestamp:
                    mean_price = self.calculate_mean_price_timestamp(trades)
                    price_sums += mean_price
                    data_point_count += 1
                else:
                    break  

        if data_point_count > 0:
            return price_sums / data_point_count                

        return 0                       
    

    """
    Calculate the Volume Weighted SMA for the last `window_size` trades of a product.
    """
    def calculate_volume_weighted_sma(self, past_trades: PastData, product: str):        
        trades_by_timestamp = sorted(self.convert_to_timestamp_dict(past_trades, product).items())

        
        if len(trades_by_timestamp) < self.WINDOW_SIZE_VOL[product]:
            return 0
        
        recent_trades = trades_by_timestamp[-self.WINDOW_SIZE_VOL[product]:]
        
        total_volume = 0
        total_product = 0
        for _, trades in recent_trades:
            if len(trades) > 0:                
                mean_price = self.calculate_mean_price_timestamp(trades)          
                total_vol_time = sum(quantity for _, quantity in trades)            
                total_product += mean_price * total_vol_time                                
                total_volume += total_vol_time

        if total_volume == 0 or total_product == 0:
            return 0  # Avoid division by zero

        vw_sma = total_product / total_volume

        return vw_sma
    

    """
    Return a dictionary with the key being the timestamp and the value of being 
    a list of tuple of the trades with the same timestamp
    """
    def convert_to_timestamp_dict(self, past_trades: PastData, product: str):
        trades_by_timestamp = {}

        for price, quantity, timestamp in past_trades.market_data[product]:
            # If the timestamp is not yet a key in the dictionary, add it with the current tuple as the first item in a list
            if timestamp not in trades_by_timestamp:
                trades_by_timestamp[timestamp] = [(price, quantity)]
            # If the timestamp is already a key, append the current tuple to the associated list
            else:
                trades_by_timestamp[timestamp].append((price, quantity))

        return trades_by_timestamp

    """
    Using the highest and lowest price of executed orders from a particular timestamp
    """
    def calculate_mean_price_timestamp(self, timestamp_trades: List[Tuple[float, int]]):
                
        highest = max(timestamp_trades, key=lambda x: x[0])[self.PD_PRICE_INDEX]
        lowest = min(timestamp_trades, key=lambda x: x[0])[self.PD_PRICE_INDEX]

        return (highest + lowest) / 2 

    """
    Taking the trading state, past market data and the product name as arguments
    Compute the open positions for that particular product
    """
    def compute_open_pos(self, state: TradingState, past_trades: PastData, product: str):
        valid_op = []
        position_count = 0
        valid_op_trade: Tuple
        if product not in state.position:
            return None
        
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
    def compute_sell_orders_sma(self, buy_order_depth: Dict[int, int], past_trades: PastData, product: str, tick_size: int):
        orders: List[Order] = []  
        temp_position = self.positions[product]
        sorted_buy_order_depth = sorted(buy_order_depth.items())

        current_sma = self.calculate_sma(past_trades, 0, product)        
        print(f"sell order sma: {current_sma}")
        
        # Go through each buy order depth to see if there's a good opportunity to match the order by buying
        for price, quantity in sorted_buy_order_depth:           
            if current_sma == 0:
                break                
            if price >= current_sma + tick_size: 
                order_quantity: int                          
                order_quantity = min((self.POSITION_LIMIT[product] + temp_position), quantity)                    
                if order_quantity > 0:
                    temp_position += -order_quantity
                    orders.append(Order(product, price, -order_quantity))
        return orders, temp_position
        

    """
    Return buy orders based on SMA
    """
    def compute_buy_orders_sma(self, sell_order_depth: Dict[int, int], past_trades: PastData, product: str, tick_size: int):
        orders: List[Order] = []  
        temp_position = self.positions[product]
        sorted_buy_order_depth = reversed(sorted(sell_order_depth.items()))

        current_sma = self.calculate_sma(past_trades, 0, product)        
        print(f"buy order sma: {current_sma}")
    
        # Go through each sell order depth to see if there's a good opportunity to match the order by buying 
        for price, quantity in sorted_buy_order_depth:            
            if current_sma == 0:
                break
            if price <= current_sma - tick_size:
                order_quantity: int                                  
                order_quantity = min((self.POSITION_LIMIT[product] - temp_position), abs(quantity))               
                if order_quantity > 0:
                    temp_position += order_quantity
                    orders.append(Order(product, price, order_quantity))                        
        return orders, temp_position
        
    
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
    
    def predict_price_lr(self, past_trades: PastData, n_past_timestamps : int, cur_timestamp: int, product: str):
        market_trade = past_trades.market_data[product] 

        sorted_market_trade = sorted(market_trade, key=lambda x: x[self.PD_TIMESTAMP_INDEX])
        # Create DataFrame
        df = pd.DataFrame(sorted_market_trade, columns=['Price', 'Quantity', 'Timestamp'])

        # Step 1: Preprocess - Average prices for the same timestamp
        df_avg = df.groupby('Timestamp').agg({'Price': 'mean'}).reset_index()

        # Step 2: Filter rows to only include those before the current timestamp
        df_filtered = df_avg[df_avg['Timestamp'] <= cur_timestamp]

        # Ensure we're selecting the last n unique timestamps leading up to the current timestamp
        if len(df_filtered) > n_past_timestamps:
            df_filtered = df_filtered.iloc[-n_past_timestamps:]
        else:
            return None


        # Linear Regression
        # Assuming equally spaced timestamps in terms of their order
        X = np.arange(len(df_filtered)).reshape(-1, 1)
        ones = np.ones(len(X)).reshape(-1, 1)
        X = np.hstack((X, ones))  # Add intercept term
        Y = df_filtered['Price'].values.reshape(-1, 1)

        # Calculate coefficients
        coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        slope = coefficients[0][0]
        intercept = coefficients[1][0]

        future_X_value = (cur_timestamp / 100) + 1
        predicted_price = slope * future_X_value + intercept
        print(predicted_price)

        return predicted_price
    
    """
    Get the best ask, best bid, and the spread
    """
    def get_order_book_insight(self, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int]):
        best_ask, _ = list(sell_order_depth.items())[0]
        best_bid, _ = list(buy_order_depth.items())[0]
        spread = best_ask - best_bid

        return best_ask, best_bid, spread

    """
    Market taking
    """
    
    def scalping_strategy_one(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str):
        orders: List[Order] = []   
        
        orders_sell, _ = self.compute_sell_orders_sma(buy_order_depth, past_trades, product, 1)        
        orders_buy, _ = self.compute_buy_orders_sma(sell_order_depth, past_trades, product, 1)            
        
        orders += orders_sell + orders_buy
        return orders
    
    """
    Market making
    Determine buy or sell signal based on VW SMA and current price.
    """
    def scalping_strategy_two(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str, arbitrage_amount: int):
        
        orders: List[Order] = []
        best_ask, best_bid, _ = self.get_order_book_insight(buy_order_depth, sell_order_depth)
        #sma = self.calculate_volume_weighted_sma(past_trades, product)
        #sma = self.calculate_sma(past_trades, 0, product)
        sma = self.calculate_sma_time(past_trades, product)

        current_price = (best_bid + best_ask) / 2

        price = 0        
        if sma != 0:
            if current_price > sma:  # Price is above SMA, consider selling
                price = best_ask - self.TICK_SIZE
                order_quantity = min(self.POSITION_LIMIT[product] + self.positions[product], arbitrage_amount)
                orders.append(Order(product, price, -order_quantity)) 
            
            elif current_price < sma:  # Price is below SMA, consider buying
                price = best_bid + self.TICK_SIZE
                order_quantity = min(self.POSITION_LIMIT[product] - self.positions[product], arbitrage_amount)
                orders.append(Order(product, price, order_quantity)) 
 
        return orders

    """ 
    Combined strategy one and two
    """
    def scalping_combination(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str):
        orders: List[Order] = []   
        
        tick_size = 2
        orders_sell, temp_pos_sell = self.compute_sell_orders_sma(buy_order_depth, past_trades, product, tick_size)        
        orders_buy, temp_pos_buy = self.compute_buy_orders_sma(sell_order_depth, past_trades, product, tick_size)
        orders += orders_sell + orders_buy
        
        # Combination with strategy two
        best_ask, best_bid, _ = self.get_order_book_insight(buy_order_depth, sell_order_depth)    
        sma = self.calculate_sma_time(past_trades, product)
        
        if sma == 0:
            return orders
        
        current_price = (best_bid + best_ask) / 2        
        if current_price > sma:
            self.positions[product] = temp_pos_sell
        elif current_price < sma:
            self.positions[product] = temp_pos_buy
                                  
        arbitrage_amount = 12 #best 12
        
        orders += self.scalping_strategy_two(past_trades, buy_order_depth, sell_order_depth, product, arbitrage_amount)
        
        return orders
    
    """
    Uses Scalping strategy to place orders
    """
    def execute_scalping(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str):
        orders: List[Order] = []          
   
        #orders += self.scalping_strategy_one(past_trades, buy_order_depth, sell_order_depth, product)        
        #orders += self.scalping_strategy_two(past_trades, buy_order_depth, sell_order_depth, product)
        orders += self.scalping_combination(past_trades, buy_order_depth, sell_order_depth, product)
    
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
                order_quantity = min((self.POSITION_LIMIT[product] + self.positions[product]), abs(quantity))                            
                orders.append(Order(product, price, -order_quantity))
                self.positions[product] += -order_quantity  

          #Market making: sell at upper bound
        if abs(self.positions[product]) < self.POSITION_LIMIT[product]:
            order_amount = min((self.POSITION_LIMIT[product] + self.positions[product]), market_make_amount)                                        
            orders.append(Order(product, self.AME_THRESHOLD_UP, -order_amount))    
            self.positions[product] += -order_amount  
                   
       
        # Place a buy order if the ask price is below the threshold
        for price, quantity in sell_order_depth.items():
            if price <= self.AME_THRESHOLD_MID:
                order_quantity = min((self.POSITION_LIMIT[product] - temp_pos), abs(quantity))                                            
                orders.append(Order(product, price, order_quantity))
                temp_pos += order_quantity        
      
         #Market making: buy at lower bound
        if abs(temp_pos) < self.POSITION_LIMIT[product]:
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
        self.cur_timestamp = state.timestamp
        past_trades: PastData
        # Check if there's data in traderData
        if not state.traderData:
            past_trades = PastData()
            past_trades.market_data = {'AMETHYSTS': [], 'STARFRUIT': []}    
            past_trades.own_trades = {'AMETHYSTS': [], 'STARFRUIT': []} 
            past_trades.open_positions = {'AMETHYSTS': [], 'STARFRUIT': []}
        else:
            past_trades = jsonpickle.decode(state.traderData)                                

        # Place orders for each product  
        for product in state.listings:              
            print(f"product name: {product}")
            
            if product not in past_trades.market_data:
                past_trades.market_data[product] = []

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
                own_trades += state.own_trades[product]  
                past_trades.own_trades[product] += [(trade.price, trade.quantity, trade.timestamp) for trade in own_trades if trade.timestamp == state.timestamp - 100 or trade.timestamp == state.timestamp]                                                                                  
                # Add own_trades into open positions  
                for trade in own_trades:
                    if (trade.timestamp == state.timestamp - 100) or (trade.timestamp == state.timestamp):
                        if trade.buyer == "SUBMISSION":
                            past_trades.open_positions[product].append((trade.price, trade.quantity))
                        elif trade.seller == "SUBMISSION":
                             past_trades.open_positions[product].append((trade.price, -trade.quantity))            

                # Get rid of past own_trades that have been closed           
                self.compute_open_pos(state, past_trades, product)

                print(f"{product} updated open positions: {len(past_trades.open_positions[product])}")
                for pos in past_trades.open_positions[product]:
                    print(f"(P: {pos[self.PD_PRICE_INDEX]} Q: {pos[self.PD_QUANTITY_INDEX]})")
                

            # Store the past trades into the past data object
            if product in state.market_trades:
                market_trades += state.market_trades[product]                         
                past_trades.market_data[product] += [(trade.price, trade.quantity, trade.timestamp) for trade in market_trades if trade.timestamp == state.timestamp - 100 or trade.timestamp == state.timestamp]                                                                  
       

            # Delete extra past data
            if state.timestamp >= self.PAST_DATA_MAX:

                for trade in past_trades.market_data[product][:]:
                    if trade[self.PD_TIMESTAMP_INDEX] == state.timestamp - self.PAST_DATA_MAX:
                        past_trades.market_data[product].remove(trade)
                    else:
                        break                            

                for trade in past_trades.own_trades[product][:]:
                    if trade[self.PD_TIMESTAMP_INDEX] == state.timestamp - self.PAST_DATA_MAX:
                        past_trades.own_trades[product].remove(trade)
                    else:
                        break    
                        
            # Initialize the list of Orders to be sent as an empty list
            orders: List[Order] = []  
            buy_order_depth: Dict[int, int]
            sell_order_depth: Dict[int, int]

            # Separate buy order depths and sell order depths
            if len(state.order_depths[product].buy_orders) > 0:       
                buy_order_depth = state.order_depths[product].buy_orders                
            if len(state.order_depths[product].sell_orders) > 0:      
                sell_order_depth = state.order_depths[product].sell_orders

            
            # Trade differently for each product
            if product == "STARFRUIT":                
                orders += self.execute_scalping(past_trades, buy_order_depth, sell_order_depth, product)
                
            elif product == "AMETHYSTS":                          
                orders += self.compute_amethysts_orders(past_trades, buy_order_depth, sell_order_depth, product)                               
                
            result[product] = orders
        
        # Serialize past trades into traderData
        traderData = jsonpickle.encode(past_trades) 
        
        # Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData


        
        
