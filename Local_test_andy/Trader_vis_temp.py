from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List, Tuple, Any
import string
import jsonpickle
import json

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
        self.market_data: Dict[str, List[Tuple[float, int, int]]] = {} #price, quantity, timestamp
        self.open_positions: Dict[str, List[Tuple[float, int]]] = {} #price, quantity
        self.prev_mid = -1
        self.prev_bid = -1
        self.prev_ask = -1

        # self.mid_prices: Dict[str, List[Tuple[float, ]]]
        # self.volume_weighted_prices: List[]

logger = Logger()
class Trader:
    spread_peak = 2  
    sf_limit = 20
    def extract_positions(self, position_dict: Dict[str, int]) -> Dict[str, int]:
        positions = {'AMETHYSTS': 0, 'STARFRUIT': 0}
        for product in positions:
            positions[product] = position_dict[product] if product in position_dict else 0
        return positions

    def run(self, state: TradingState):                
        # Orders to be placed on exchange matching engine
        result = {}
        positions = self.extract_positions(state.position)

        if not state.traderData:
            past_data = PastData()
        else:
            past_data = jsonpickle.decode(state.traderData)
        
        # STARTFRUIT STRATEGY
        product = 'STARFRUIT'
        orders: List[Order] = []
        ob_sf = state.order_depths[product]
        prev_mid = past_data.prev_mid
        prev_ask = past_data.prev_ask
        prev_bid = past_data.prev_bid
        position_sf = positions['STARFRUIT']
        
        # Making sure a valid spread is calculated.
        
        if len(ob_sf.sell_orders) != 0 and len(ob_sf.buy_orders) != 0:
            best_ask, best_ask_amount = list(ob_sf.sell_orders.items())[0]
            best_bid, best_bid_amount = list(ob_sf.buy_orders.items())[0]
            mid_price = (best_ask + best_bid)//2
            spread = best_ask - best_bid
            
            if spread <= self.spread_peak and prev_mid != -1: # The first is a peak signal. The second is to skip first round.
                bid_diff = best_bid - prev_bid
                ask_diff = prev_ask - best_ask
                if bid_diff > ask_diff: # This is an up peak. Should sell.
                    take_amount = best_bid_amount if position_sf - best_bid_amount >= -self.sf_limit else abs(-self.sf_limit-position_sf)
                    make_amount = abs(-self.sf_limit - (position_sf - take_amount))
                    orders.append(Order(product, best_bid, -(take_amount+make_amount)))
                    logger.print("Sell for bid peak!")
                else: # This is a down peak should buy.
                    take_amount = best_ask_amount if position_sf + best_ask_amount <= self.sf_limit else self.sf_limit - position_sf
                    make_amount = self.sf_limit - (position_sf + take_amount)
                    orders.append(Order(product, best_ask, take_amount+make_amount))
                    logger.print("Buy for ask peak!")

        # Update mid_price
        past_data.prev_mid = mid_price
        past_data.prev_bid = best_bid
        past_data.prev_ask = best_ask
        traderData = jsonpickle.encode(past_data)
        result[product] = orders
                
        conversions = None
        # traderData = ""
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData