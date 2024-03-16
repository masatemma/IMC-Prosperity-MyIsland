import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import Any, Dict, List, Tuple
import jsonpickle

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
            state.traderData,
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
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

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

logger = Logger()

class PastData: 

    def __init__(self):
        self.market_data: Dict[str, List[Tuple[float, int]]] = {}
        self.open_positions: Dict[str, List[Tuple[float, int]]] = {}      

class Trader:
    WINDOW_SIZE = {'AMETHYSTS': 4, 'STARFRUIT': 6}   # best A: 4, S: 6    
    VWAP_WINDOW = 20
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
                      
            past_trades_sum_price = sum(data[self.PD_PRICE_INDEX] for data in past_market_data[-self.WINDOW_SIZE[product]: ])

            #print(f"Past market data window size: {past_market_data[-self.WINDOW_SIZE[product]:]}")
            past_trades_sum_price += order_depth_price
            mean_value = float(past_trades_sum_price / (len(past_market_data[-self.WINDOW_SIZE[product]:]) + 1))

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
    Return sell orders based on SMA
    """
    def compute_sell_orders_sma(self, buy_order_depth: Dict[int, int], past_trades: PastData, product: str):
        orders: List[Order] = []  
        for price, quantity in buy_order_depth.items():
            current_sma = self.calculate_sma(past_trades.market_data[product], price, product)
            logger.print(f"current SMA: {current_sma}")
            logger.print(f"Price:{price}") 
            
            if current_sma == 0:
                break                
            if price >= current_sma + self.TICK_SIZE:  
                order_quantity: int
                if self.positions[product] == 0:
                    order_quantity = quantity
                else:                    
                    order_quantity = min(self.POSITION_LIMIT[product] - abs(self.positions[product]), quantity)
                if order_quantity > 0:
                    self.positions[product] += -order_quantity
                    orders.append(Order(product, price, -order_quantity))
                    
                    logger.print(f"Sell: ${price}, {quantity}")
                    logger.print(f"position: {self.positions[product]}")
        return orders
    
    """
    Return buy orders based on SMA
    """
    def compute_buy_orders_sma(self, sell_order_depth: Dict[int, int], past_trades: PastData, product: str):
        orders: List[Order] = []  
        # Go through each sell order depth to see if there's a good opportunity to match the order by buying 
        for price, quantity in sell_order_depth.items():
            current_sma = self.calculate_sma(past_trades.market_data[product], price, product)
            print(f"current SMA: {current_sma}")
            if current_sma == 0:
                print("break")
                break
            if price <= current_sma - self.TICK_SIZE:
                order_quantity: int
                if self.positions[product] == 0:
                    order_quantity = quantity
                else:                    
                    order_quantity = min((self.POSITION_LIMIT[product] - abs(self.positions[product])), abs(quantity))
                if order_quantity > 0:
                    self.positions[product] += order_quantity
                    orders.append(Order(product, price, order_quantity))                        
                    print(f"Buy: ${price}, {quantity}")
                    print(f"position: {self.positions[product]}")
        return orders
            
    """
    Only method required. It takes all buy and sell orders for all symbols as an input,
    and outputs a list of orders to be sent
    """
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 1
        traderData = ""   

        # TODO: Add logic
        # Record past market trades prices
        past_trades = PastData()
        past_trades.market_data = {'AMETHYSTS': [], 'STARFRUIT': []}     
        past_trades.open_positions = {'AMETHYSTS': [], 'STARFRUIT': []}        

        # Check if there's data in traderData
        if len(state.traderData) > 0:
            past_trades = jsonpickle.decode(state.traderData)                                

        # Place orders for each product  
        for product in state.listings:              
            logger.print(f"product name: {product}")
            
            if product not in past_trades.market_data:
                past_trades.market_data[product] = []
            if product not in past_trades.open_positions:
                past_trades.open_positions[product] = []

            # Record positions
            if product in state.position:       
                self.positions[product] = state.position[product]                        
                logger.print(f"Actual starting_position: {state.position[product]}")
                
            logger.print(f"Starting position: {self.positions[product]}")
           
            
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

                logger.print(f"{product} open positions: {len(past_trades.open_positions[product])}")
                for pos in past_trades.open_positions[product]:
                    logger.print(f"(P: {pos[self.PD_PRICE_INDEX]} Q: {pos[self.PD_QUANTITY_INDEX]})")


                # Get rid of past own_trades that have been closed           
                self.compute_open_pos(state, past_trades, product)
                logger.print(f"{product} updated open positions: {len(past_trades.open_positions[product])}")
                for pos in past_trades.open_positions[product]:
                    logger.print(f"(P: {pos[self.PD_PRICE_INDEX]} Q: {pos[self.PD_QUANTITY_INDEX]})")
                
            
            if product in state.market_trades:
                all_trades += state.market_trades[product]                         

            if len(all_trades) > 0:                                                                    

                # Add the past market trades and own trades into traderData
                past_trades.market_data[product] += [(trade.price, trade.quantity) for trade in all_trades if trade.timestamp == state.timestamp - 100 or trade.timestamp == state.timestamp]    
                
                # if len(past_trades.market_data[product]) > self.WINDOW_SIZE[product]:         
                #     past_trades.market_data[product] = past_trades.market_data[product][-self.WINDOW_SIZE[product]:]
                
                logger.print(f"{product} Past Market Data Size: {len(past_trades.market_data[product])}")
                

                      
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

                if abs(fair_price - best_bid) > abs(fair_price - best_ask):                   
                    order_quantity = min(self.POSITION_LIMIT[product] - abs(self.positions[product]), buy_order_depth[best_bid])
                    orders.append(Order(product, best_bid, -order_quantity))
                    self.positions[product] += -order_quantity

                elif abs(fair_price - best_bid) < abs(fair_price - best_ask):
                    order_quantity = min((self.POSITION_LIMIT[product] - abs(self.positions[product])), sell_order_depth[best_ask])
                    orders.append(Order(product, best_ask, order_quantity))
                    self.positions[product] += order_quantity

            elif product in state.position:
                if state.position[product] == 0 and (fair_price != 0.0):
                    orders += self.compute_sell_orders_sma(buy_order_depth, past_trades, product)
                    orders += self.compute_buy_orders_sma(sell_order_depth, past_trades, product)                    
                    
                elif state.position[product] > 0:
                    # Go through each buy order depth to see if there's a good opportunity to match the order by selling 
                    orders += self.compute_sell_orders_sma(buy_order_depth, past_trades, product)
                
                elif state.position[product] < 0:
                    # Go through each sell order depth to see if there's a good opportunity to match the order by buying 
                    orders += self.compute_buy_orders_sma(sell_order_depth, past_trades, product)
               
            
            result[product] = orders
        
        # Serialize past trades into traderData
        #traderData = jsonpickle.encode(past_trades) 
        traderData = "SAMPLE" 
        # Sample conversion request. Check more details below. 
        conversions = 1
        
        logger.flush(state, result, conversions, traderData)
        traderData = jsonpickle.encode(past_trades)
        
        return result, conversions, traderData