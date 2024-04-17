from typing import Dict, List, Tuple, Any
import string
import numpy as np
import pandas as pd
import jsonpickle
import math
from collections import Counter
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            # state.traderData,
            "",
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing['symbol'], listing['product'], listing['denomination']])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

class PastData: 
    def __init__(self):
        self.market_data: Dict[str, List[Tuple[float, int, int]]] = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS': [], 'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': []}     #price, quantity, timestamp
        self.own_trades: Dict[str, List[Tuple[float, int, int]]] = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS': [], 'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': []}      #price, quantity, timestamp
        self.open_positions: Dict[str, List[Tuple[float, int]]] = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS': [], 'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': []}      #price, quantity     
        self.portfolio: Dict[str, Tuple[int, int]] = {'AMETHYSTS': (0,0), 'STARFRUIT': (0,0), 'ORCHIDS': (0,0), 'CHOCOLATE': (0,0), 'STRAWBERRIES': (0,0), 'ROSES': (0,0), 'GIFT_BASKET': (0,0)}
        self.prev_mid = -1
        self.mid_prices: Dict[str, List[float]] = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS': [], 'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': []} 
        self.humidity_rates_of_change: List[float] = []
        self.sunlight_rates_of_change: List[float] = []
        self.prev_humidity = -1    
        self.prev_sunlight = -1 
        self.sell_orchids_at = -1   
        self.humidity_entry = False
        self.sunlight_entry = -1
        self.sunlight_exit = -1
        self.currently_long_spread = False
        self.currently_short_spread = False

logger = Logger()

class Trader:
    SIGMA_MULTIPLIER_MAKE = {'AMETHYSTS': 1, 'STARFRUIT': 1.5}
    SIGMA_MULTIPLIER_TAKE = {'AMETHYSTS': 1, 'STARFRUIT': 1}
    WINDOW_SIZE_LR = {'AMETHYSTS': 25, 'STARFRUIT': 35}  

    PORTFOLIO_TRADE_AMOUNT = {'AMETHYSTS': 2, 'STARFRUIT': 2}  # Not used for Starfruit
    # PORTFOLIO_TRADE_AMOUNT = {'AMETHYSTS': 10, 'STARFRUIT': 2}  # Not used for Starfruit
    PORTFOLIO_TRADE_MARGIN = {'AMETHYSTS': 0, 'STARFRUIT': 1}
    PORTFOLIO_TRADE_THRESHOLD = {'AMETHYSTS': 10, 'STARFRUIT': 15}
      
    PRODUCT_LIST = ['AMETHYSTS', 'STARFRUIT', 'ORCHIDS'] 
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60}
    WINDOW_SIZE = {'AMETHYSTS': 4, 'STARFRUIT': 10, 'ORCHIDS': 10, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 30, 'GIFT_BASKET': 5} 
    WINDOW_SIZE_TIME = {'AMETHYSTS': 25, 'STARFRUIT': 25, 'ORCHIDS': 20,'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 15, 'GIFT_BASKET': 5}
    WINDOW_SIZE_VOL = {'AMETHYSTS': 5, 'STARFRUIT': 15, 'ORCHIDS': 20, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 15, 'GIFT_BASKET': 5}
    WINDOW_SIZE_MIDPRICE = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 15, 'GIFT_BASKET': 5}
    WINDOW_SIZE_EXP = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 7, 'GIFT_BASKET': 5}
    VWAP_WINDOW = 20
    PAST_DATA_MAX = 5000    
    TICK_SIZE = 1
    PD_PRICE_INDEX = 0
    PD_QUANTITY_INDEX = 1
    PD_TIMESTAMP_INDEX = 2    
    AME_THRESHOLD_MID = 10000
    AME_THRESHOLD_UP = 10004
    AME_THRESHOLD_LOW = 9996
    HUMIDITY_OPT_LOW = 60
    HUMIDITY_OPT_HIGH = 80 
    SUNLIGHT_AVG = 2500

    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0}    
    cur_timestamp = 0
    cur_state: TradingState
    
    
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
    Calculate SMA based on the mid price of the past timestamps
    """
    
    def calculate_ema(self, past_trades: PastData, product: str):    
        if len(past_trades.mid_prices[product]) < self.WINDOW_SIZE_EXP[product]:
            return 0
        
        alpha = 2 /(self.WINDOW_SIZE_EXP[product] + 1)
        
        prices = past_trades.mid_prices[product][-self.WINDOW_SIZE_EXP[product]:]
        ema = [prices[0]]  # Start the EMA with the first price
        
        for price in prices[1:]:
            ema.append(alpha * price + (1 - alpha) * ema[-1])
            
        return ema[-1]
    
    def calculate_midprice_sma(self, past_trades: PastData, product: str):
        if len(past_trades.mid_prices[product]) < self.WINDOW_SIZE_MIDPRICE[product]:
            return 0
        
        sum_past_midprices = sum(price for price in past_trades.mid_prices[product][-self.WINDOW_SIZE_MIDPRICE[product]: ])
        return sum_past_midprices / self.WINDOW_SIZE_MIDPRICE[product]
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

    def compute_buy_sell_orderdepths(self, state: TradingState, product):
        buy_order_depth: Dict[int, int]
        sell_order_depth: Dict[int, int]
        # Separate buy order depths and sell order depths
        if len(state.order_depths[product].buy_orders) > 0:       
            buy_order_depth = state.order_depths[product].buy_orders                
        if len(state.order_depths[product].sell_orders) > 0:      
            sell_order_depth = state.order_depths[product].sell_orders
        
        return buy_order_depth, sell_order_depth 
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

        current_sma = self.calculate_ema(past_trades, product)                
        logger.print(f"SMA: {current_sma}")
        # Go through each buy order depth to see if there's a good opportunity to match the order by buying
        for price, quantity in sorted_buy_order_depth:
            if current_sma == 0:
                break                
            if price < current_sma - tick_size: 
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

        current_sma = self.calculate_ema(past_trades, product)                
        logger.print(f"SMA: {current_sma}")
        # Go through each sell order depth to see if there's a good opportunity to match the order by buying 
        for price, quantity in sorted_buy_order_depth:            
            if current_sma == 0:
                break
            if price > current_sma + tick_size:
                order_quantity: int                                  
                order_quantity = min((self.POSITION_LIMIT[product] - temp_position), abs(quantity))               
                if order_quantity > 0:
                    temp_position += order_quantity
                    orders.append(Order(product, price, order_quantity))                        
        return orders, temp_position
        
    """Use the past `self.WINDOW_SIZE_LR` number of mid prices to infer the current mid price, and make trade accordingly. """
    def compute_trade_threshold(self, past_trades: PastData, product: str) -> Dict[str, float]:
        X = [self.cur_timestamp + i * 100 for i in range(-self.WINDOW_SIZE_LR[product], 0)]
        Y = [past_trades.mid_prices[product][i] for i in range(-self.WINDOW_SIZE_LR[product]-1, -1)]

        X = np.column_stack((np.ones_like(Y), X))
        coeffs = np.linalg.inv(X.T @ X) @ X.T @ Y
    
        # Calculate residuals
        residuals = Y - X @ coeffs
        
        # Degrees of freedom
        n = X.shape[0]
        p = X.shape[1]
        dof = n - p
    
        # Estimate sample standard deviation
        stdev = np.sqrt(np.sum(residuals ** 2) / dof)

        # Infer the mid price at the current timestamp, and use the inferred value as a fair price.
        fair_price = np.array([1, self.cur_timestamp]) @ coeffs
        sell_threshold_take = math.ceil(fair_price + stdev * self.SIGMA_MULTIPLIER_TAKE[product])
        buy_threshold_take = math.floor(fair_price - stdev * self.SIGMA_MULTIPLIER_TAKE[product])
        sell_threshold_make = math.ceil(fair_price + stdev * self.SIGMA_MULTIPLIER_MAKE[product])
        buy_threshold_make = math.floor(fair_price - stdev * self.SIGMA_MULTIPLIER_MAKE[product])

        return {
            'sell_threshold_make': sell_threshold_make, 
            'buy_threshold_make': buy_threshold_make,
            'sell_threshold_take': sell_threshold_take, 
            'buy_threshold_take': buy_threshold_take,
            "fair_price": fair_price, 
            "stdev": stdev} 
    
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
        if current_price > sma:  # Price is above SMA, consider selling
            price = best_ask - self.TICK_SIZE
            order_quantity = min(self.POSITION_LIMIT[product] + self.positions[product], arbitrage_amount)
            orders.append(Order(product, price, -order_quantity)) 
        
        elif current_price < sma:  # Price is below SMA, consider buying
            price = best_bid + self.TICK_SIZE
            order_quantity = min(self.POSITION_LIMIT[product] - self.positions[product], arbitrage_amount)
            orders.append(Order(product, price, order_quantity)) 
 
        return orders

    def scalping_strategy_three(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str, arbitrage_amount: int):
        orders: List[Order] = []
        best_ask, best_bid, spread = self.get_order_book_insight(buy_order_depth, sell_order_depth)
        sma = self.calculate_midprice_sma(past_trades, product)

        mid_price = (best_bid + best_ask) / 2

        if self.positions[product] == 0 and sma != 0:   
            price = 0
            tick_size = 1
            temp_position = self.positions[product]
            if mid_price < sma + tick_size:  # Price is above SMA, consider selling
                for price, quantity in list(buy_order_depth.items()):                        
                    order_quantity: int                          
                    order_quantity = min((self.POSITION_LIMIT[product] + temp_position), quantity)                    
                    if order_quantity > 0:
                        temp_position += -order_quantity
                        orders.append(Order(product, price, -order_quantity))
            
            elif mid_price > sma - tick_size:  # Price is below SMA, consider buying
                for price, quantity in list(sell_order_depth.items()):            
                    order_quantity: int                                  
                    order_quantity = min((self.POSITION_LIMIT[product] - temp_position), abs(quantity))               
                    if order_quantity > 0:
                        temp_position += order_quantity
                        orders.append(Order(product, price, order_quantity)) 

        elif self.positions[product] < 0 and len(past_trades.open_positions[product]) > 0:            
            sorted_open_pos = reversed(sorted(past_trades.open_positions[product], key=lambda x: x[0])) #highest price first 
            profit = arbitrage_amount * 2
            stop_loss = arbitrage_amount
            for price, quantity in sorted_open_pos:
                if price - profit >= best_ask:
                    order_quantity = -quantity            
                    orders.append(Order(product, best_ask, order_quantity))
                elif price + stop_loss <= best_ask:
                    order_quantity = -quantity          
                    orders.append(Order(product, best_ask, order_quantity))
                
        elif self.positions[product] > 0 and len(past_trades.open_positions[product]) > 0:
            #Selling to close long position
            sorted_open_pos = sorted(past_trades.open_positions[product], key=lambda x: x[0]) #lowest price first
            profit = arbitrage_amount * 2
            stop_loss = arbitrage_amount
            for price, quantity in sorted_open_pos:
                if price + profit <= best_bid:  
                    order_quantity = -quantity                    
                    orders.append(Order(product, best_bid, order_quantity))
                elif price - stop_loss >= best_bid: 
                    order_quantity = -quantity                                       
                    orders.append(Order(product, best_bid, order_quantity))

        return orders
    """ 
    Combined strategy one and two
    """
    def scalping_combination(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str):
        orders: List[Order] = []   
        
        tick_size = 2
        orders_sell, temp_pos_sell = self.compute_sell_orders_sma(buy_order_depth, past_trades, product, tick_size)        
        orders_buy, temp_pos_buy = self.compute_buy_orders_sma(sell_order_depth, past_trades, product, tick_size)
                
        # Combination with strategy two
        best_ask, best_bid, _ = self.get_order_book_insight(buy_order_depth, sell_order_depth)    
        sma = self.calculate_sma_time(past_trades, product)
        current_price = (best_bid + best_ask) / 2
        
        if current_price > sma:
            self.positions[product] = temp_pos_sell
        elif current_price < sma:
            self.positions[product] = temp_pos_buy
            
        orders += orders_sell + orders_buy      
        
        arbitrage_amount = 12 #best 12
        
        orders += self.scalping_strategy_two(past_trades, buy_order_depth, sell_order_depth, product, arbitrage_amount)
        
        return orders
    
    """
    Uses Scalping strategy to place orders
    """
    def execute_scalping(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str):
        orders: List[Order] = []          
   
        orders += self.scalping_strategy_one(past_trades, buy_order_depth, sell_order_depth, product)        
        #orders += self.scalping_strategy_two(past_trades, buy_order_depth, sell_order_depth, product)
        #orders += self.scalping_combination(past_trades, buy_order_depth, sell_order_depth, product)
    
        return orders
    
    """
    Place sell orders if the bid price is above the threshold of Amethysts.
    Place buy  orders if the ask price is below the threshold of Amethysts.
    
    """
    def compute_amethysts_orders(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str) -> List[Order]:
        orders: List[Order] = []   
        if product != "AMETHYSTS":
            return 0


        temp_pos_sell = self.positions[product]
        temp_pos_buy = self.positions[product]

        market_make_amount = 40 # 11
        #Place a sell order if the bid price is above the threshold
        for price, quantity in buy_order_depth.items():
            if price >= self.AME_THRESHOLD_MID:                              
                order_quantity = min((self.POSITION_LIMIT[product] + temp_pos_sell), abs(quantity))                     
                orders.append(Order(product, price, -order_quantity))
                temp_pos_sell += -order_quantity                           

        #Market making: sell at upper bound        
        order_amount = min((self.POSITION_LIMIT[product] + temp_pos_sell), market_make_amount)                                        
        orders.append(Order(product, self.AME_THRESHOLD_UP, -order_amount))    
        temp_pos_sell += -order_amount 

        # Place a buy order if the ask price is below the threshold
        for price, quantity in sell_order_depth.items():
            if price <= self.AME_THRESHOLD_MID:
                order_quantity = min((self.POSITION_LIMIT[product] - temp_pos_buy), abs(quantity))                                         
                orders.append(Order(product, price, order_quantity))
                temp_pos_buy += order_quantity                   

        #Market making: buy at lower bound          
        order_amount = min((self.POSITION_LIMIT[product] - temp_pos_buy), market_make_amount)                    
        orders.append(Order(product, self.AME_THRESHOLD_LOW, order_amount))
        temp_pos_buy += order_amount  


        return orders

    def update_portfolio(self, position_state: Tuple[int, int], own_trades: List[Trade], mid_price: int) -> Tuple[int, int]:
        """
            class Trade:
                def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId = None, seller: UserId = None, timestamp: int = 0) -> None:
                    self.symbol = symbol
                    self.price: int = price
                    self.quantity: int = quantity
                    self.buyer = buyer
                    self.seller = seller
                    self.timestamp = timestamp
        """
        if len(own_trades) == 0:
            logger.print("No own trade.")
            return position_state
        else:
            prev_position = position_state[0]
            prev_value_avg = position_state[1]

            trade_quantity_sum = 0
            buy_quantity = 0
            sell_quantity = 0 # sell_quantity <= 0
            buy_value = 0
            sell_value = 0
            
            trade_count = 0
            
            for trade in own_trades:
                
                if int(trade.timestamp) != int(self.cur_timestamp)-100:
                    continue
                trade_count += 1

                assert trade.buyer == "SUBMISSION" or trade.seller == "SUBMISSION"

                quantity = -trade.quantity if trade.seller == "SUBMISSION" else trade.quantity
                trade_quantity_sum += quantity
                if quantity > 0:
                    buy_quantity += quantity
                    buy_value += trade.price * trade.quantity
                else:
                    sell_quantity += quantity
                    sell_value += trade.price * trade.quantity
            assert trade_quantity_sum == buy_quantity + sell_quantity
            # buy_avg = buy_value/buy_quantity
            # sell_avg = sell_value/sell_quantity

            if trade_count > 0:
                # Movement pattern 1 - Move from position 0. The larger position move determines the advance direction.
                # pos means position, NOT positive
                if prev_position == 0:
                    
                    if abs(buy_quantity) != abs(sell_quantity):
                        advance_move = buy_quantity if abs(buy_quantity) > abs(sell_quantity) else sell_quantity
                        retreat_move = sell_quantity if abs(buy_quantity) > abs(sell_quantity) else buy_quantity
                        advance_value = buy_value if abs(buy_quantity) > abs(sell_quantity) else sell_value
                        return (advance_move+retreat_move, advance_value/abs(advance_move))
                    else: 
                        return (0,0)
                else:
                    result_position = prev_position + trade_quantity_sum

                    # Movement pattern 2 - Move from any position except zero, and end up on the same side with the previous state.
                    # Use a retreat first approach. (Make more sense when retreat to zero position.)
                    if prev_position * result_position > 0:
                        advance_move = buy_quantity if result_position > 0 else sell_quantity
                        retreat_move = sell_quantity if result_position > 0 else buy_quantity
                        advance_value = buy_value if result_position > 0 else sell_value
                        
                        # Retreat first
                        intermediate_position = prev_position + retreat_move
                        assert intermediate_position + advance_move == result_position

                        # If retreat into another side or zero.
                        if intermediate_position * prev_position <= 0:
                            new_avg_value = advance_value/abs(advance_move)
                            return (intermediate_position + advance_move, new_avg_value)
                        # If staying on the same side after retreating.
                        else:
                            new_avg_value = (prev_value_avg * abs(intermediate_position) + advance_value)/abs(intermediate_position + advance_move)
                            return (intermediate_position + advance_move, new_avg_value)

                    # Movement pattern 3 - Move from any position except zero, and end up on 0.
                    elif result_position == 0:
                        return (0,0)
                    
                    # Movement pattern 4 - Move from any position except zero, and end up on the other side.
                    elif prev_position * result_position < 0:
                        advance_value = buy_value if prev_position < 0 else sell_value
                        advance_move = buy_quantity if prev_position < 0 else sell_quantity
                        value_avg = advance_value / abs(advance_move)
                        return (result_position, value_avg)
                        
                
                    # Unexpected movment pattern.
                    else:
                        logger.log("Assign random number for unexpected behavior. #4563218462")
                        logger.log("Warning - portfolio reset to mid price!")
                        return (prev_position + trade_quantity_sum, mid_price)
            else:
                return position_state
            
    def update_mid_prices(self, product: str, past_trades: PastData, state: TradingState) -> None:    
        buy_order_depth = state.order_depths[product].buy_orders                
        sell_order_depth = state.order_depths[product].sell_orders

        if len(sell_order_depth) != 0 and len(buy_order_depth) != 0:
            best_ask, _ = list(sell_order_depth.items())[0]
            best_bid, _ = list(buy_order_depth.items())[0]
            mid_price = (best_ask + best_bid)//2
            past_trades.mid_prices[product].append(mid_price)
        else:            
            past_trades.mid_prices[product].append(past_trades.mid_prices[product][-1])
        
        if len(past_trades.mid_prices[product]) > self.PAST_DATA_MAX/100:
            past_trades.mid_prices[product].pop(0)
            assert len(past_trades.mid_prices[product]) <= self.PAST_DATA_MAX/100

    """
    The function makes trade for a product when no trade was proposed by other methods.
    The trades proposed by this function is based on the value of each stock position.
    Check update_portfolio to understand how values are calculated.
    """
    def trade_portfolio(self, result: Dict[str, List[Order]], portfolio: Dict[str, Tuple[int, int]], target_products: List[str]) -> None:
        product_no_trade = set()
        for product in target_products:
            if product not in result: # No trade proposed
                product_no_trade.add(product)
            else: # Trade proposed, but with 0 quantity
                trades_proposed = result[product]
                quantity_all_zero = True
                for order in trades_proposed:
                    if order.quantity != 0: 
                        quantity_all_zero = False
                        break
                if quantity_all_zero:
                    product_no_trade.add(product)
            
        for product in product_no_trade:
            position = self.positions[product]
            product_position = portfolio[product][0]
            assert position == product_position, f"position: {position}; product_position: {product_position}" # Sanity check

            # Trade portfolio if positions are over predefined threshold.
            product_avg_value = portfolio[product][1]
            margin = self.PORTFOLIO_TRADE_MARGIN[product]
            if position > self.PORTFOLIO_TRADE_THRESHOLD[product]:
                sell_quantity = max(-self.PORTFOLIO_TRADE_AMOUNT[product], -self.POSITION_LIMIT[product]-position) 
                result[product].append(Order(product, math.ceil(product_avg_value)+margin, sell_quantity))
                logger.print(f"Portfolio sell for {product} at {math.ceil(product_avg_value)+margin}")
            elif position < -self.PORTFOLIO_TRADE_THRESHOLD[product]: 
                buy_quantity = min(self.PORTFOLIO_TRADE_AMOUNT[product], self.POSITION_LIMIT[product]-position)
                result[product].append(Order(product, math.floor(product_avg_value)-margin, buy_quantity))
                logger.print(f"Portfolio buy for {product} at {math.floor(product_avg_value)-margin}")


                
        
        # PORTFOLIO_TRADE_AMOUNT = {'AMETHYSTS': 2, 'STARFRUIT': 2}  
        # PORTFOLIO_TRADE_THRESHOLD = {'AMETHYSTS': 10, 'STARFRUIT': 10}  
        # Order(product, price, -order_quantity)
        # Order(product, self.AME_THRESHOLD_UP, -order_amount)
        
    def make_threshold_trade(self, trade_threshold, past_trades, buy_order_depth, sell_order_depth, product, trade_portfolio=True):
        orders: List[Order] = []  
        temp_position = self.positions[product]
        sorted_buy_order_depth = sorted(buy_order_depth.items())
        sorted_sell_order_depth = sorted(sell_order_depth.items())

        sell_threshold_take = trade_threshold['sell_threshold_take'] # Sell (take) above this
        buy_threshold_take = trade_threshold['buy_threshold_take'] # Buy (take) below this    
        sell_threshold_make = trade_threshold['sell_threshold_make'] # Sell (make) above this
        buy_threshold_make = trade_threshold['buy_threshold_make'] # Buy (take) below this

        # Variables below are for portfolio trading
        margin = self.PORTFOLIO_TRADE_MARGIN[product]
        product_position, product_avg_value = past_trades.portfolio[product]
        portfolio_trade_threshold = self.PORTFOLIO_TRADE_THRESHOLD[product]
        assert product_position == temp_position
        
        # Go through each buy order depth to see if there's a good opportunity to match orders

        # Market taking
        """Sell"""
        sell_temp_position = temp_position
        for price, quantity in sorted_buy_order_depth:            
            if price >= sell_threshold_take: 
                order_quantity: int                          
                order_quantity = min((self.POSITION_LIMIT[product] + sell_temp_position), quantity)                    
                if order_quantity > 0:
                    sell_temp_position += -order_quantity
                    orders.append(Order(product, price, -order_quantity))
        
        # The boolean below describe the pre-conditions for portfolio trade (sell).
        if trade_portfolio and sell_temp_position == temp_position and temp_position >= portfolio_trade_threshold:
            for price, quantity in sorted_buy_order_depth:
                if price >= math.ceil(product_avg_value + margin):
                    order_quantity: int                          
                    order_quantity = min(temp_position, quantity)
                    if order_quantity > 0:
                        sell_temp_position += -order_quantity
                        orders.append(Order(product, price, -order_quantity))                    

        
        """Buy"""
        buy_temp_position = temp_position
        for price, quantity in sorted_sell_order_depth:            
            if price <= buy_threshold_take:
                order_quantity: int                                  
                order_quantity = min((self.POSITION_LIMIT[product] - buy_temp_position), abs(quantity))               
                if order_quantity > 0:
                    buy_temp_position += order_quantity
                    orders.append(Order(product, price, order_quantity))
        
        # The boolean below describe the pre-conditions for portfolio trade (buy).
        if trade_portfolio and buy_temp_position == temp_position and temp_position <= -portfolio_trade_threshold:
            for price, quantity in sorted_sell_order_depth:
                if price <= math.floor(product_avg_value - margin):
                    order_quantity: int                          
                    order_quantity = min(abs(temp_position), abs(quantity))
                    if order_quantity > 0:
                        buy_temp_position += order_quantity
                        orders.append(Order(product, price, order_quantity))
        
        # Market making

        """
        The block below is tested for STARFRUIT. 
        The resulting values are too extreme to be taken by bots
        """
        # mid_price_current = past_trades.mid_prices[product][-1]
        # make_sell_quantity = self.POSITION_LIMIT[product] + temp_position
        # if make_sell_quantity != 0 and mid_price_current >= sell_threshold:
        #     orders.append(Order(product, best_ask-1, -make_sell_quantity))

        # make_buy_quantity = self.POSITION_LIMIT[product] - temp_position
        # if make_buy_quantity != 0 and mid_price_current <= buy_threshold:
        #     orders.append(Order(product, best_bid+1, make_buy_quantity))

        """
        The market making code below works.
        """
        make_sell_quantity = self.POSITION_LIMIT[product] + sell_temp_position
        if make_sell_quantity != 0:
            orders.append(Order(product, sell_threshold_make, -make_sell_quantity))

        make_buy_quantity = self.POSITION_LIMIT[product] - buy_temp_position
        if make_buy_quantity != 0:
            orders.append(Order(product, buy_threshold_make, make_buy_quantity))

        return orders, temp_position   
        
    """
    Update the rates of change list whenever there's new data
    """
    def update_rates_of_change(self,past_trades: PastData, con_ob: ConversionObservation):     
        cur_humidity = con_ob.humidity
        cur_sunlight = con_ob.sunlight   
        if past_trades.prev_humidity == -1:
            past_trades.prev_humidity = cur_humidity
            past_trades.prev_sunlight = cur_sunlight            
            return None
        
        if past_trades.prev_humidity > 0 or past_trades.prev_sunlight > 0:
            rate_of_change_humidity = cur_humidity - past_trades.prev_humidity
            rate_of_change_sunlight = cur_sunlight - past_trades.prev_sunlight
            
            past_trades.humidity_rates_of_change.append(rate_of_change_humidity)
            past_trades.sunlight_rates_of_change.append(rate_of_change_sunlight)
            
            past_trades.prev_humidity = cur_humidity
            past_trades.prev_sunlight = cur_sunlight                        
            
            if len(past_trades.humidity_rates_of_change) > 10:
                past_trades.humidity_rates_of_change.pop(0)
            if len(past_trades.sunlight_rates_of_change) > 100:
                past_trades.sunlight_rates_of_change.pop(0)                                        
        
    """
    Place conversion request based on current position and humidity peak
    """
    def conversion_request(self, state: TradingState, con_ob: ConversionObservation, past_trades: PastData, product: str):
        if len(past_trades.humidity_rates_of_change) < 3:
            return 0
        
        conversions = 0
        cur_humidity = con_ob.humidity
        cur_sunlight = con_ob.sunlight
        avg_sunlight_rate_of_change = sum(past_trades.sunlight_rates_of_change) / len(past_trades.sunlight_rates_of_change)
        avg_humidity_rate_of_change = sum(past_trades.humidity_rates_of_change) / len(past_trades.humidity_rates_of_change)
        
        # Close the position for humidity entry
        if past_trades.humidity_entry and self.positions[product] < 0:            
                      
            if (self.HUMIDITY_OPT_LOW <= cur_humidity and cur_humidity < self.HUMIDITY_OPT_HIGH + 5) and (past_trades.humidity_rates_of_change[-3] < 0 and past_trades.humidity_rates_of_change[-2] >= 0 and past_trades.humidity_rates_of_change[-1] > 0):
                conversions = abs(self.positions[product])
                past_trades.sell_orchids_at = -1
                past_trades.humidity_entry = False
                
            elif (self.HUMIDITY_OPT_LOW - 5 < cur_humidity and cur_humidity <= self.HUMIDITY_OPT_HIGH) and (past_trades.humidity_rates_of_change[-3] > 0 and past_trades.humidity_rates_of_change[-2] <= 0 and past_trades.humidity_rates_of_change[-1] < 0):
                conversions = abs(self.positions[product])
                past_trades.sell_orchids_at = -1
                past_trades.humidity_entry = False
                
            elif cur_sunlight < self.SUNLIGHT_AVG and avg_sunlight_rate_of_change < 0:
                conversions = abs(self.positions[product])
                past_trades.sell_orchids_at = -1
                past_trades.humidity_entry = False                                     
            
        # Close the position for sunlight entry
        if past_trades.sunlight_entry in [1, 2] and self.positions[product] < 0:         
            if state.timestamp == past_trades.sunlight_exit:
                conversions = abs(self.positions[product])
                past_trades.sell_orchids_at = -1
                past_trades.sunlight_entry = -1
                
            elif cur_humidity > self.HUMIDITY_OPT_HIGH and avg_humidity_rate_of_change > 0:
                conversions = abs(self.positions[product])
                past_trades.sell_orchids_at = -1
                past_trades.sunlight_entry = -1
                
            elif cur_humidity < self.HUMIDITY_OPT_LOW and avg_humidity_rate_of_change < 0:
                conversions = abs(self.positions[product])
                past_trades.sell_orchids_at = -1
                past_trades.sunlight_entry = -1
                             
        return conversions        
        
    """
    Place orders based on humidity
    """
    def compute_orchids_orders(self, state: TradingState, con_ob: ConversionObservation, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str):        
        orders: List[Order] = []
        temp_pos = self.positions[product]
        cur_humidity = con_ob.humidity
        cur_sunlight = con_ob.sunlight
             
        if len(past_trades.humidity_rates_of_change) < 3:
            return orders        
        
        """
        Sell entry for humidity
        """
        if not past_trades.humidity_entry:                                                 
            # Sell if detect upward peak in humidity when humidity is above 85
            if temp_pos == 0 and (cur_humidity >= self.HUMIDITY_OPT_HIGH + 5) and (past_trades.humidity_rates_of_change[-3] > 0 and past_trades.humidity_rates_of_change[-2] <= 0 and 
                                past_trades.humidity_rates_of_change[-1] < 0): 
                offset = 50000
                past_trades.sell_orchids_at = state.timestamp + offset
                past_trades.humidity_entry = True
                past_trades.sunlight_entry = -1 
            
            # Sell if detect downward peak in humidity is below 65
            elif temp_pos == 0 and (cur_humidity <= self.HUMIDITY_OPT_LOW - 5) and (past_trades.humidity_rates_of_change[-3] < 0 and past_trades.humidity_rates_of_change[-2] >= 0 and 
                                past_trades.humidity_rates_of_change[-1] > 0): 
                offset = 50000
                past_trades.sell_orchids_at = state.timestamp + offset
                past_trades.humidity_entry = True
                past_trades.sunlight_entry = -1
            
            elif state.timestamp < 100000 and cur_humidity >= self.HUMIDITY_OPT_HIGH + 14:
                past_trades.sell_orchids_at = state.timestamp
                past_trades.humidity_entry = True
                past_trades.sunlight_entry = -1
                                    
                
        """
        Sell entry for sunlight
        """  
        average_rate_of_change = sum(past_trades.sunlight_rates_of_change) / len(past_trades.sunlight_rates_of_change)
        if past_trades.sunlight_entry == -1:                          
            if temp_pos == 0 and int(cur_sunlight) in [2500,2499,2498,2497] and average_rate_of_change < 0:
                entry_offset = 80000
                exit_offset = 40000            
                past_trades.sell_orchids_at = state.timestamp + entry_offset
                past_trades.sunlight_exit = state.timestamp + entry_offset + exit_offset 
                past_trades.sunlight_entry = 1
                past_trades.humidity_entry = False
                
            elif temp_pos == 0 and int(cur_sunlight) in [2500,2501,2502,2503] and average_rate_of_change > 0:
                entry_offset = 35000
                exit_offset = 30000            
                past_trades.sell_orchids_at = state.timestamp + entry_offset
                past_trades.sunlight_exit = state.timestamp + entry_offset + exit_offset 
                past_trades.sunlight_entry = 2
                past_trades.humidity_entry = False                        
        
        # cancel false signal
        if past_trades.sunlight_entry == 1 and past_trades.sell_orchids_at > state.timestamp and average_rate_of_change > 0 and cur_sunlight > self.SUNLIGHT_AVG:
            past_trades.sell_orchids_at = -1
            past_trades.sunlight_entry = -1
        elif past_trades.sunlight_entry == 2 and past_trades.sell_orchids_at > state.timestamp and average_rate_of_change < 0 and cur_sunlight < self.SUNLIGHT_AVG:
            past_trades.sell_orchids_at = -1
            past_trades.sunlight_entry = -1
        
        #Execute the sell
        if past_trades.sell_orchids_at == state.timestamp and self.positions[product] == 0:                                
            #Market take from the orderbook 
            for price, quantity in buy_order_depth.items():
                order_quantity = min(self.POSITION_LIMIT[product] + temp_pos, quantity)
                orders.append(Order(product, price, -order_quantity))
                temp_pos -= order_quantity
                
            if abs(temp_pos) < self.POSITION_LIMIT[product]:
                order_quantity = self.POSITION_LIMIT[product] + temp_pos
                orders.append(Order(product, price, -order_quantity))  
                            
        return orders

    def market_take_orders(self, order_depths: Dict[int, int], product: str, buy: bool, close=False):
        orders: List[Order] = []
        temp_position = self.positions[product]
        # Go through each sell order depth to see if there's a good opportunity to match the order by buying 
        if not close:
            for price, quantity in order_depths.items():                                                                                             
                order_quantity = min(self.POSITION_LIMIT[product], abs(quantity))              
                if order_quantity > 0:                    
                    if buy:
                        temp_position += order_quantity
                        orders.append(Order(product, price, order_quantity)) 
                    else:
                        temp_position += -order_quantity
                        orders.append(Order(product, price, -order_quantity)) 
                break
        elif close:
            for price, quantity in order_depths.items():                                                                             
                order_quantity = min(abs(temp_position), abs(quantity))               
                if order_quantity > 0:                    
                    if buy:
                        temp_position += order_quantity
                        orders.append(Order(product, price, order_quantity)) 
                    else:
                        temp_position += -order_quantity
                        orders.append(Order(product, price, -order_quantity)) 
                break 
                        
                                      
        return orders
                
    def pairs_trading(self, past_trades: PastData, product_1: str, product_2: str, long_ma_length: int, short_ma_length: int, entry_threshold: float, exit_threshold: float) -> List[Order]: 
                     
        product_1_orders: List[Order] = []  
        product_2_orders: List[Order] = []    
        
        if len(past_trades.mid_prices[product_1]) < long_ma_length:
            return [], []
     
        product_1_prices = np.array(past_trades.mid_prices[product_1][-long_ma_length:])
        product_2_prices = np.array(past_trades.mid_prices[product_2][-long_ma_length:])
        short_product_1_prices = np.array(past_trades.mid_prices[product_1][-short_ma_length:])
        short_product_2_prices = np.array(past_trades.mid_prices[product_2][-short_ma_length:])
        
        #Get long MA
        long_ma = np.mean(product_1_prices - product_2_prices)
        logger.print(f"Long_MA: {long_ma}")
        #Get the std of the long window
        long_std = np.std(product_1_prices - product_2_prices)  
        logger.print(f"long_std: {long_std}")      
        #Get short MA
        short_ma = np.mean(short_product_1_prices - short_product_2_prices)
        logger.print(f"Short_MA: {short_ma}")
        # Compute z-score
        if long_std > 0:
            zscore = (short_ma - long_ma)/long_std
        
        logger.print(f"zscore: {zscore}")

        product_1_buy_order_depth, product_1_sell_order_depth = self.compute_buy_sell_orderdepths(self.cur_state,product_1)
        product_2_buy_order_depth, product_2_sell_order_depth = self.compute_buy_sell_orderdepths(self.cur_state,product_2)        
        if zscore > entry_threshold and not past_trades.currently_short_spread:
            product_1_orders += self.market_take_orders(product_1_buy_order_depth, product_1, False) # short top
            product_2_orders += self.market_take_orders(product_2_sell_order_depth, product_2, True) # long bottom
            past_trades.currently_short_spread = True
            past_trades.currently_long_spread = False
        
        elif zscore < -entry_threshold and not past_trades.currently_long_spread:
            product_1_orders += self.market_take_orders(product_1_sell_order_depth, product_1, True) # long top
            product_2_orders += self.market_take_orders(product_2_buy_order_depth, product_2, False) # short bottom
            past_trades.currently_short_spread = False
            past_trades.currently_long_spread = True
        
        elif abs(zscore) < exit_threshold:
            if self.positions[product_1] > 0:
                product_1_orders += self.market_take_orders(product_1_buy_order_depth, product_1, False, close=True) 
            elif self.positions[product_1] < 0:
                product_1_orders += self.market_take_orders(product_1_sell_order_depth, product_1, True, close=True)
                
            if self.positions[product_2] > 0:
                product_2_orders += self.market_take_orders(product_2_buy_order_depth, product_2, False, close=True) 
            elif self.positions[product_2] < 0:
                product_2_orders += self.market_take_orders(product_2_sell_order_depth, product_2, True, close=True)                
                                                                
            past_trades.currently_short_spread = False
            past_trades.currently_long_spread = False
            
        return product_1_orders, product_2_orders                
        
                
    """
    Only method required. It takes all buy and sell orders for all symbols as an input,
    and outputs a list of orders to be sent
    """      
    def run(self, state: TradingState): 
        # Orders to be placed on exchange matching engine
        result = {}
        traderData = ""  
        self.cur_timestamp = state.timestamp
        self.cur_state = state
        # Initialize the list of Orders to be sent as an empty list         
        buy_order_depth: Dict[int, int]
        sell_order_depth: Dict[int, int]
        
        # Check if there's data in traderData
        if not state.traderData:
            past_trades = PastData()           
        else:
            past_trades = jsonpickle.decode(state.traderData)                     

        # Process info for each product  
        for product in state.listings:
            self.update_mid_prices(product, past_trades, state)              
            logger.print(f"product name: {product}")
            
            if product not in past_trades.market_data:
                past_trades.market_data[product] = []

            if product not in past_trades.open_positions:
                past_trades.open_positions[product] = []

            # Record positions
            if product in state.position:       
                self.positions[product] = state.position[product]                        
            elif product not in state.position:
                self.positions[product] = 0                                    
            
            # Combine all trades from own trades and market trades                
            own_trades: List[Trade] = []
            market_trades: List[Trade] = []
            
            # Update trader data
            if product in state.own_trades:
                # Update portfolio
                #logger.print(f"{product} Portfolio before update: {past_trades.portfolio[product]}")
                past_trades.portfolio[product] = self.update_portfolio(past_trades.portfolio[product], state.own_trades[product], past_trades.mid_prices[product])
                #logger.print(f"{product} Portfolio after update: {past_trades.portfolio[product]}")

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
                
            # Update trader data
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
            
        """
        Main trading logics
        """      
        
        
        #Trade Gift Basket and Roses
        # roses_orders, gb_orders = self.pairs_trading(past_trades,"ROSES", "GIFT_BASKET", 40, 7, 1.5, 0.4)        
        # result["ROSES"] = roses_orders
        # result["GIFT_BASKET"] = gb_orders
        
        # Trade Chocolate and Strawberries
        choco_orders, strawberries_orders = self.pairs_trading(past_trades,"CHOCOLATE", "STRAWBERRIES", 40, 10, 1.5, 0.2)        
        result["CHOCOLATE"] = choco_orders
        result["STRAWBERRIES"] = strawberries_orders
        
        # Trade Roses
        # buy_order_depth, sell_order_depth = self.compute_buy_sell_orderdepths(state, "ROSES")
        # result["ROSES"] = self.scalping_strategy_one(past_trades, buy_order_depth, sell_order_depth, "ROSES")
        
        # Trade Orchids  
        conversions = 0                                                           
        # buy_order_depth, sell_order_depth = self.compute_buy_sell_orderdepths(state, "ORCHIDS")
        # con_ob = state.observations.conversionObservations["ORCHIDS"]                                
        # self.update_rates_of_change(past_trades, con_ob)
        # result["ORCHIDS"] = self.compute_orchids_orders(state, con_ob, past_trades, buy_order_depth, sell_order_depth, "ORCHIDS")                
        # conversions = self.conversion_request(state, con_ob, past_trades, "ORCHIDS")                                                 
        
        # Trade STARFRUIT                   
        """trade_threshold - {'sell_threshold': sell_threshold, 'buy_threshold': buy_threshold}, """
        # buy_order_depth, sell_order_depth = self.compute_buy_sell_orderdepths(state, "STARFRUIT")
        # if self.cur_timestamp / 100 > self.WINDOW_SIZE_LR["STARFRUIT"]-1: 
        #     trade_threshold = self.compute_trade_threshold(past_trades, "STARFRUIT")
        #     portfolio_trade = True
        #     threshold_orders, _ = self.make_threshold_trade(trade_threshold, past_trades, buy_order_depth, sell_order_depth, "STARFRUIT", trade_portfolio=portfolio_trade) 
        #     result["STARFRUIT"] = threshold_orders             
            
        # Trade AMETHYSTS            
        # buy_order_depth, sell_order_depth = self.compute_buy_sell_orderdepths(state, "AMETHYSTS")                    
        # result["AMETHYSTS"] = self.compute_amethysts_orders(past_trades, buy_order_depth, sell_order_depth, "AMETHYSTS")                             
                                                
        target_products = ["AMETHYSTS"]
        #self.trade_portfolio(result, past_trades.portfolio, target_products)

        # Serialize past trades into traderData
        traderData = jsonpickle.encode(past_trades) 
                       
        logger.flush(state, result, conversions, "")
        return result, conversions, traderData



    



    