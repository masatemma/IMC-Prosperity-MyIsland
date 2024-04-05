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
        # Masa attributes
        self.market_data: Dict[str, List[Tuple[float, int, int]]] = {} #price, quantity, timestamp
        self.open_positions: Dict[str, List[Tuple[float, int]]] = {} #price, quantity

        # Andy attributes
        self.prev_mid = -1 # This attribute is for the trivial Starfruit strategy 
        self.current_bucket: Dict[str, List[Tuple[int, int]]]
        self.past_buckets: Dict[str, List[float]]
        self.rsi: Dict[str, float]
        self.ma: Dict[str, float]
        

logger = Logger()
class Trader: 
    """ TradingState class and Trade class for reference
    class TradingState:
    def __init__(self,
                 traderData: str,
                 timestamp: Time,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId = None, seller: UserId = None, timestamp: int = 0) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp
    """

    BUCKET_SIZE = 20
    RSI_WINDOW = 10
    MA_WINDOW = 5
    RSI_UPPER = 60
    RSI_LOWER = 40

    def compute_rsi(self, past_prices: List[float]):
        past_prices_in_win = past_prices[-self.RSI_WINDOW]        
        price_diff = [past_prices_in_win[i] - past_prices_in_win[i-1] for i in range(1, len(past_prices_in_win))]
        pos_diff = [diff for diff in price_diff if diff > 0]
        neg_diff = [diff for diff in price_diff if diff < 0]
        pos_avg = sum(pos_diff)/len(pos_diff)
        neg_avg = -sum(neg_diff)/len(neg_diff)
        rs = pos_avg/neg_avg
        rsi = 100 - (100 / (1+rs))
            
        return rsi

    def compute_ma(self, past_prices: List[float]):
        return sum(past_prices[-self.MA_WINDOW])/self.MA_WINDOW

    def update_trader_data(self, past_data: PastData, state: TradingState) -> Dict[str, Dict[str, bool]]:
        #TODO - Need to change this to a list to include other products.
        target_product = 'STARFRUIT'
        
        # Initialise output to signal whether rsi and ma are newly updated.
        output = {target_product: {"rsi_ma": False}}

        # Update current bucket. TODO: Expand it to all products in the future.
        trades = state.market_trades[target_product]
        price_sum = sum([trade.price for trade in trades])
        quantity_sum = sum([trade.quantity for trade in trades])
        past_data.current_bucket[target_product].append((price_sum, quantity_sum))

        # Simplify namespace for the steps after.
        current_bucket = past_data.current_bucket[target_product]

        # Reset bucket when bucket size is reached. And calculate RSI and MA if enough data have been collected.
        if len(current_bucket) == self.BUCKET_SIZE:
            # Compute the aggregation of 20 iterations.
            bucket_avg = sum([bucket_item[0] for bucket_item in current_bucket])/sum([bucket_item[1] for bucket_item in current_bucket])

            #Update `past_data.past_buckets`
            if len(past_data.past_buckets) == max([self.RSI_WINDOW, self.MA_WINDOW]):
                past_data.past_buckets.pop(0)
            past_data.past_buckets.append(bucket_avg)

            #Compute RSI and MA
            is_rsi_update = False
            is_ma_update = False
            past_buckets = past_data.past_buckets[target_product]
            if len(past_buckets) >= self.RSI_WINDOW:
                past_data.rsi[target_product] = self.compute_rsi(past_buckets)
                is_rsi_update = True

            if len(past_buckets) >= self.MA_WINDOW:
                past_data.ma[target_product] = self.compute_ma(past_buckets)
                is_ma_update = True
            
            if is_rsi_update and is_ma_update:
                output[target_product]['rsi_ma'] = True

            # Reset bucket
            past_data.current_bucket[target_product] = []
        
        return output

    def extract_positions(self, position_dict: Dict[str, int]) -> Dict[str, int]:
        positions = {'AMETHYSTS': 0, 'STARFRUIT': 0}
        for product in positions:
            positions[product] = position_dict[product] if product in position_dict else 0
        return positions

    def take_all(self, action: str, state: TradingState, product: str):
        # TODO: Work on this to make sure that order volume does not surpass position limits. WIP
        """
        class OrderDepth:
        def __init__(self):
            self.buy_orders: Dict[int, int] = {}
            self.sell_orders: Dict[int, int] = {}
        """
        position = 0 # This is an assumption
        output = list()
        orderDepth = state.order_depths[product]
        if action == "sell":
            sell_orders = orderDepth.sell_orders
            for price, quantity in sell_orders.items():
                position += quantity
                output.append(Order(product, mid_price, -arbitrage_amount))
                
        elif action == "buy":
            buy_orders = orderDepth.buy_orders
        else:
            print("Invalid action in take_all")


    def run(self, state: TradingState):
        """This is the stage of updating/initialising and loading traderData"""
        if not state.traderData:
            past_data = PastData()
        else:
            past_data = jsonpickle.decode(state.traderData)
        
        # Update market_data in traderData
        update_output = self.update_trader_data(past_data, state)
        product = 'STARFRUIT'

        # Orders to be placed on exchange matching engine
        result = {}
        positions = self.extract_positions(state.position)
        orders: List[Order] = []

        """RSI + MA for Starfruit"""
        # orders.append(Order(product, mid_price, -arbitrage_amount))
        # if past_data.rsi > 60 
        if update_output[product]['rsi_ma']:
            entry_conditions = past_data.rsi[product] > self.RSI_UPPER and past_data.past_buckets[product][-1] > past_data.ma[product]
            exit_conditions = past_data.rsi[product] > self.RSI_LOWER and past_data.past_buckets[product][-1] < past_data.ma[product]

            # Enter
            if positions[product] == 0 and entry_conditions:
                sell_orders = self.take_all('sell', state, product)
                for order in sell_orders:
                    orders.append(order)

            # Exit
            elif positions[product] < 0 and exit_conditions:
                buy_orders = self.take_all('buy', state, product)
                for order in buy_orders:
                    orders.append(order)
        
        """Wrap up for submitting orders"""
        # Some dummy data which is not used at the current stage
        conversions = None
        # Wrapping up and reporting
        traderData = jsonpickle.encode(past_data)
        result[product] = orders
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData