import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

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
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        # TODO: Add logic
        for product in state.listings:              
            
            if product in state.position:       
                self.positions[product] = state.position[product]
            
            # logger.print(f"product name: {product}")
            # logger.print(f"starting_position: {self.positions[product]}")
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
                logger.print(f"fair price: {fair_price}")
                
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
                            logger.print(f"Sell: ${price}", {quantity})
                            logger.print(f"position: {self.positions[product]}")
                    
                for price, quantity in sell_order_depth.items():
                    logger.print(f"price: {price} quantity: {quantity}") 
                    # Opportunity to buy
                    # If there's a sell order depth at a price lower than the fair price, we buy
                    if price < fair_price:
                        order_quantity = min((self.POSITION_LIMIT[product] - abs(self.positions[product])), abs(quantity))
                        if order_quantity > 0:
                            orders.append(Order(product, price, order_quantity))
                            self.positions[product] += order_quantity
                            logger.print(f"Buy: ${price}, {quantity}")
                            logger.print(f"position: {self.positions[product]}")
            
        
            
            result[product] = orders
        
        conversions = 1
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data