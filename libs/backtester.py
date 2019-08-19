import pandas as pd
import datetime as dt

class BackTester:

    def __init__(self, universe=pd.DataFrame(), start_date, end_date):
        self.universe = universe

        pass

    def get_optimal_holdings(self, date_):
        universe = self.get_universe(date_)



    def get_universe(self, date_):
        # should adjust the universe to only stocks > 1B or ones we already have a position
        return self.universe



    def build_tradelist(self):
        pass




