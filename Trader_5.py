from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List, Tuple
import string
import numpy as np
import jsonpickle
import math


class PastData: 

    def __init__(self):
        self.market_data: Dict[str, List[Tuple[float, int, int]]] = {} #price, quantity, timestamp
        self.open_positions: Dict[str, List[Tuple[float, int]]] = {} #price, quantity     


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
    AMETHYSTS_THRESHOLD = 10000

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
    def calculate_sma(self, past_market_data: List[Tuple[float, int, int]], order_depth_price: float, product: str, cur_timestamp: int) -> float:  
            
        if order_depth_price != 0 and len(past_market_data) + 1 < self.WINDOW_SIZE[product]:
            return 0
        elif  order_depth_price == 0 and len(past_market_data) < self.WINDOW_SIZE[product]:
            return 0
                    
        past_trades_sum_price = sum(data[self.PD_PRICE_INDEX] for data in past_market_data[-self.WINDOW_SIZE[product]: ])

        #print(f"Past market data window size: {past_market_data[-self.WINDOW_SIZE[product]:]}")
        
        mean_value: float
        if order_depth_price != 0:
            past_trades_sum_price += order_depth_price
            mean_value = float(past_trades_sum_price / (len(past_market_data[-self.WINDOW_SIZE[product]:]) + 1))
        else:
            mean_value = float(past_trades_sum_price / (len(past_market_data[-self.WINDOW_SIZE[product]:])))
            
        return mean_value 
    

    """
    Taking the past trading data, price of an order depth, the product name and the current timestamp as arguments.
    Calculates the simple moving average value based on the timestamps
    """
    def calculate_sma_time(self, past_market_data: List[Tuple[float, int, int]], order_depth_price: float, product: str, cur_timestamp: int) -> float:
        target_timestamp = cur_timestamp - (self.WINDOW_SIZE[product] * 100) 
        past_trades_sum_price = 0
        past_trades_count = 0

        if target_timestamp >= 0:
            for trade in reversed(past_market_data):
                if (trade[self.PD_TIMESTAMP_INDEX] >= target_timestamp):
                    past_trades_sum_price += trade[self.PD_PRICE_INDEX]
                    past_trades_count += 1
                else:
                    break
        
        if past_trades_count == 0:
            return 0

        return past_trades_sum_price / past_trades_count
                        
    
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
        print("Compute sell order")
        orders: List[Order] = []  

        for price, quantity in buy_order_depth.items():
            current_sma = self.calculate_sma(past_trades.market_data[product], price, product, cur_timestamp)
            print(f"current SMA: {current_sma}")
            print(f"Price:{price}") 
            
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
                    
                    print(f"Sell: ${price}, {quantity}")
                    print(f"position: {self.positions[product]}")
        return orders
        

    """
    Return buy orders based on SMA
    """
    def compute_buy_orders_sma(self, sell_order_depth: Dict[int, int], past_trades: PastData, product: str, cur_timestamp: int):
        print("Compute buy order")
        orders: List[Order] = []  
        # Go through each sell order depth to see if there's a good opportunity to match the order by buying 
        for price, quantity in sell_order_depth.items():
            current_sma = self.calculate_sma(past_trades.market_data[product], price, product, cur_timestamp)
            print(f"current SMA: {current_sma}")
            print(f"Price:{price}") 
            if current_sma == 0:
                print("break")
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
                    print(f"Buy: ${price}, {quantity}")
                    print(f"position: {self.positions[product]}")
        return orders
        
    
    """
    Compute the orders for starfruit. 
    Place a sell order a bit above the Bid price from bots and 
    place a buy order a well below the simple moving average
    """
    def compute_starfruit_orders(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str, cur_timestamp: int):
        if (product != "STARFRUIT"):
            return None
        
        orders: List[Order] = []   
                      
        if self.positions[product] >= 0:
            # Selling at a price a bit above the best bid price in the order depth            
            best_bid, best_bid_amount = list(buy_order_depth.items())[0]            
            sma =  self.calculate_sma(past_trades.market_data[product], 0, product, cur_timestamp) 
            if sma != 0  and best_bid >= math.floor(sma):                            
                order_amount = self.positions[product] + self.POSITION_LIMIT[product]
                orders.append(Order(product, best_bid, - order_amount))
                self.positions[product] += -order_amount
            
        elif self.positions[product] < 0:
            # Buying at a price way below the SMA

            # best_ask, best_ask_amount = list(sell_order_depth.items())[0]  
            # sma = self.calculate_sma(past_trades.market_data[product], best_ask, product, cur_timestamp) 
            # if sma != 0:          
            #     order_amount = abs(self.positions[product]) + self.POSITION_LIMIT[product]
            #     orders.append(Order(product, math.floor(sma) - self.SF_BUY, order_amount))
            #     self.positions[product] += order_amount

            for price, quantity in sell_order_depth.items():
                for pos in past_trades.open_positions[product]:                    
                    sold_price = pos[self.PD_PRICE_INDEX]
                    print(f"open position : {pos}")
                    print(f"current order depth price: {price}")
                    if sold_price >= price + self.SF_BUY:
                        print(f"open position price: {sold_price}")
                        print(f"current order depth price: {price}")

                        order_amount = min(abs(self.positions[product]), abs(pos[self.PD_QUANTITY_INDEX]))
                        if order_amount > 0:
                            orders.append(Order(product, price, order_amount))
                            self.positions[product] += order_amount

                
        print(orders)
        
        return orders
    
    
    """
    Uses Scraping strategy to place orders
    """
    def scraping(self, past_trades: PastData, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str, cur_timestamp: int):
        orders: List[Order] = []          
        fair_price = self.calculate_vwap(past_trades.market_data[product], product)    
         # Scraping Strategy

        orders += self.compute_sell_orders_sma(buy_order_depth, past_trades, product, cur_timestamp)
        orders += self.compute_buy_orders_sma(sell_order_depth, past_trades, product, cur_timestamp)

        return orders
    
    """
    Place sell orders if the bid price is above the threshold of Amethysts.
    Place buy  orders if the ask price is below the threshold of Amethysts.
    
    """
    def compute_amethysts_orders(self, buy_order_depth: Dict[int, int], sell_order_depth: Dict[int, int], product: str) -> List[Order]:
        orders: List[Order] = []   
        if product != "AMETHYSTS":
            return 0

        # Place a sell order if the bid price is above the threshold
        # for price, quantity in buy_order_depth.items():
        #     if price >= self.AME_THRESHOLD:
        #         order_quantity: int
        #         if self.positions[product] > 0:
        #             order_quantity = min((self.POSITION_LIMIT[product] + abs(self.positions[product])), abs(quantity))
        #         else: 
        #             order_quantity = min((self.POSITION_LIMIT[product] + self.positions[product]), abs(quantity))

        #         orders.append(Order(product, price, -order_quantity))
        #         self.positions[product] += -order_quantity

        # # Place a buy order if the ask price is below the threshold
        # for price, quantity in sell_order_depth.items():
        #     if price <= self.AME_THRESHOLD:
        #         order_quantity: int
        #         if self.positions[product] < 0:
        #             order_quantity = min((self.POSITION_LIMIT[product] + abs(self.positions[product])), abs(quantity))
        #         else: 
        #             order_quantity = min((self.POSITION_LIMIT[product] - self.positions[product]), abs(quantity))
                
        #         orders.append(Order(product, price, order_quantity))
        #         self.positions[product] += order_quantity

        best_bid_price, best_bid_quantity = list(buy_order_depth.items())[0]
        if best_bid_price >= self.AMETHYSTS_THRESHOLD:
            order_quantity: int
            if self.positions[product] > 0:
                order_quantity = self.POSITION_LIMIT[product] + abs(self.positions[product])
            else: 
                order_quantity = self.POSITION_LIMIT[product] + self.positions[product]                
            orders.append(Order(product, best_bid_price, -order_quantity))
            self.positions[product] += -order_quantity

        best_ask_price, best_ask_quantity = list(sell_order_depth.items())[0]
        if best_ask_price <= self.AMETHYSTS_THRESHOLD:
            order_quantity: int
            if self.positions[product] < 0:
                order_quantity = self.POSITION_LIMIT[product] + abs(self.positions[product])
            else: 
                order_quantity = self.POSITION_LIMIT[product] - self.positions[product]                
            orders.append(Order(product, best_ask_price, order_quantity))
            self.positions[product] += order_quantity

    
        return orders



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
            print(f"product name: {product}")
            
            if product not in past_trades.market_data:
                past_trades.market_data[product] = []

            if product not in past_trades.open_positions:
                past_trades.open_positions[product] = []

            # Record positions
            if product in state.position:       
                self.positions[product] = state.position[product]                        
                print(f"Actual starting_position: {state.position[product]}")
            elif product not in state.position:
                self.positions[product] = 0
                
            print(f"Starting position: {self.positions[product]}")
           
            
            # Combine all trades from own trades and market trades                
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
                past_trades.market_data[product] += [(trade.price, trade.quantity, trade.timestamp) for trade in all_trades if trade.timestamp == state.timestamp - 100 or trade.timestamp == state.timestamp]                                                    
                print(f"{product} Past Market Data Size: {past_trades.market_data[product]}")                
                      
            # Initialize the list of Orders to be sent as an empty list
            orders: List[Order] = []  
            buy_order_depth: Dict[int, int]
            sell_order_depth: Dict[int, int]

            # Separate buy order depths and sell order depths
            if len(state.order_depths[product].buy_orders) > 0:       
                buy_order_depth = state.order_depths[product].buy_orders                
            if len(state.order_depths[product].sell_orders) > 0:      
                sell_order_depth = state.order_depths[product].sell_orders
            
            # Calculate the fair price of the product using VWAP
            fair_price = self.calculate_vwap(past_trades.market_data[product], product)    
            

            # Trade differently for each product
            if product == "STARFRUIT":
                print("STARFRUIT Strategy")
                #orders += self.compute_starfruit_orders(past_trades, buy_order_depth, sell_order_depth, product, state.timestamp)
                orders += self.scraping(past_trades, buy_order_depth, sell_order_depth, product, state.timestamp)    
                
            elif product == "AMETHYSTS":
                # Scraping Strategy          
                orders += self.scraping(past_trades, buy_order_depth, sell_order_depth, product, state.timestamp)    
                #orders += self.compute_amethysts_orders(buy_order_depth, sell_order_depth, product)

            result[product] = orders
        
        # Serialize past trades into traderData
        traderData = jsonpickle.encode(past_trades) 
        
        # Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData


        
        
