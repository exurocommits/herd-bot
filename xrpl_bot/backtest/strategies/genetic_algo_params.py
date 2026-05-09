import numpy as np

class GeneticAlgoParams:
    """
    Evolves EMA crossover parameters (fast, slow) using a population-based approach.
    Population: 20 pairs in [3, 50].
    Fitness: Rolling 50-bar Sharpe Ratio.
    Evolution: Every 50 bars (Selection, Crossover, Mutation).
    """
    def __init__(self, data):
        self.data = data
        self.pop_size = 20
        self.generation_interval = 50
        self.param_range = (3, 50)
        
        # Initialize population: each individual is [fast_ema, slow_ema]
        self.population = []
        for _ in range(self.pop_size):
            f = np.random.uniform(*self.param_range)
            s = np.random.uniform(*self.param_range)
            if f >= s: s = f + 1
            self.population.append(np.array([f, s]))
        
        self.best_params = self.population[0]
        self.best_fitness = -np.inf
        self.current_index = 0
        self.bars_since_evolution = 0

    def _calculate_sharpe(self, fast, slow, data_slice):
        if len(data_slice) < 2: return 0.0
        close = data_slice['close']
        ema_f = close.ewm(span=int(fast)).mean()
        ema_s = close.ewm(span=int(slow)).mean()
        
        # Signal: 1 if fast > slow else -1
        signals = (ema_f > ema_s).astype(int).diff().fillna(0)
        returns = close.pct_change() * signals.shift(1)
        
        if returns.std() == 0: return 0.0
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24) # Annualized approx
        return sharpe

    def _evolve(self):
        # 1. Evaluate Fitness
        fitnesses = []
        for ind in self.population:
            # Use recent 50-bar window for fitness
            window = self.data.iloc[max(0, self.current_index-50):self.current_index]
            fitnesses.append(self._calculate_sharpe(ind[0], ind[1], window))
        
        fitnesses = np.array(fitnesses)
        
        # 2. Selection (Top 5)
        indices = np.argsort(fitnesses)[-5:]
        top_parents = self.population[indices]
        
        # Update best
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_params = top_parents[np.where(indices == best_idx)[0][0]]

        # 3. Crossover & Mutation to refill population
        new_population = list(top_parents)
        while len(new_population) < self.pop_size:
            p1, p2 = top_parents[np.random.randint(0, 5)], top_parents[np.random.randint(0, 5)]
            # Crossover (swap fast/slow)
            child = np.array([p1[0], p2[1]])
            # Mutation (+/- 5 random)
            child += np.random.uniform(-5, 5, 2)
            # Clip to range
            child = np.clip(child, *self.param_range)
            # Ensure fast < slow (or vice versa, let's say fast < slow)
            if child[0] >= child[1]:
                child[0], child[1] = child[1], child[0]
            new_population.append(child)
            
        self.population = new_population

    def next(self, bar):
        idx = self.current_index
        price = self.data['close'].iloc[idx]
        
        self.current_index += 1
        self.bars_since_evolution += 1
        
        if self.bars_since_evolution >= self.generation_interval:
            self._evolve()
            self.bars_since_evolution = 0
            
        return {"price": price, "params": self.best_params}

    def get_signals(self):
        return []

    def get_params(self):
        return {"best_fast": self.best_params[0], "best_slow": self.best_params[1], "best_fitness": self.best_fitness}

    def get_signal_dict(self, fast, slow, price):
        # Check crossover logic
        # In actual backtest, this is done via EMA comparison
        # For this class, we assume the caller provides the current EMA status or we compute it
        return {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}

    def get_crossover_signal(self, close_series):
        # Helper for the backtest to check current state
        idx = self.current_index - 1
        if idx < 1: return "HOLD"
        
        fast_ema = close_series.ewm(span=int(self.best_params[0])).mean().iloc[idx]
        slow_ema = close_series.ewm(span=int(self.best_params[1])).mean().iloc[idx]
        
        if fast_ema > slow_ema:
            return "BUY"
        elif fast_ema < slow_ema:
            return "SELL"
        return "HOLD"
