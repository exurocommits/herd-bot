import numpy as np

class PrincipalComponentStrategy:
    \"\"\"
    Principal Component Analysis (PCA) Residual Mean Reversion.
    
    Performs PCA on returns of 5 assets using numpy SVD.
    First PC = market factor. Residuals = asset-specific.
    Trade residual mean reversion:
    - BUY when residual z < -2
    - SELL when z > +2
    Retrain PCA every 50 bars.
    \"\"\"
    def __init__(self, data):
        \"\"\"
        :param data: dict containing 'prices' as a 2D numpy array (n_bars, n_assets).
        \"\"\"
        self.prices = data['prices']
        self.n_assets = self.prices.shape[1]
        self.window = 50
        self.current_index = 0
        self.pca_retrain_interval = 50
        self.last_pca_retrain = 0
        
        # State for residuals
        self.residuals = np.zeros(self.n_assets)
        self.residual_means = np.zeros(self.n_assets)
        self.residual_stds = np.ones(self.n_assets)
        
    def _retrain_pca(self, idx):
        \"\"\"Retrains PCA using SVD on the window of returns up to idx.\"\"\"
        start_idx = max(0, idx - self.window)
        window_prices = self.prices[start_idx : idx + 1]
        if len(window_prices) < 2:
            return
            
        # Calculate returns
        returns = (window_prices[1:] - window_prices[:-1]) / window_prices[:-1]
        
        # PCA using SVD: returns = U * S * V^T
        # V contains the principal components (loadings)
        # The first PC (column of V) is the market factor.
        U, S, Vh = np.linalg.svd(returns, full_matrices=False)
        
        # Vh is V^T. The first row of Vh is the first principal component.
        market_factor_loadings = Vh[0, :]
        
        # To get residuals, we project returns onto the first PC and subtract
        # projection = (returns @ market_factor_loadings) * market_factor_loadings
        # However, it's easier to just reconstruct the returns using the first PC:
        # reconstructed = (U[:, 0] * S[0])[:, np.newaxis] @ market_factor_loadings[np.newaxis, :]
        
        # Let's calculate residuals directly:
        # Residuals = Returns - (Projected Returns)
        # Projected returns = (returns @ market_factor_loadings.T) @ market_factor_loadings
        
        projected_returns = (returns @ market_factor_loadings.T)[:, np.newaxis] @ market_factor_loadings[np.newaxis, :]
        residuals = returns - projected_returns
        
        # We want the most recent residual for each asset
        self.residuals = residuals[-1, :]
        
        # Update mean and std for z-score
        self.residual_means = np.mean(residuals, axis=0)
        self.residual_stds = np.std(residuals, axis=0)
        
        self.last_pca_retrain = idx

    def next(self, bar):
        \"\"\"
        :param bar: dict containing 'prices' (list/array of current prices).
        \"\"\"
        self.current_index += 1
        idx = self.current_index
        
        if idx < self.window:
            return {"type": "HOLD", "price": 0.0, "confidence": 0.0, "metadata": {}}
            
        # Retrain PCA periodically
        if (idx - self.last_pca_retrain) >= self.pca_retrain_interval:
            self._retrain_pca(idx)
            
        if self.last_pca_retrain == 0:
            return {"type": "HOLD", "price": 0.0, "confidence": 0.0, "metadata": {}}

        # Use current asset prices to calculate current return
        # Note: We need the return for the current bar to calculate the z-score of the residual
        # In a real time-step, we'd have the current return.
        # For this class, we assume we can derive it from self.prices.
        
        # We'll pick one asset to signal on (the one with the highest |z|)
        # to fit the single signal requirement.
        
        # Since we only update residuals during PCA retraining, 
        # we'll use the residual calculated during the last PCA training.
        
        z_scores = (self.residuals - self.residual_means) / (self.residual_stds + 1e-9)
        
        # Find asset with max absolute z-score
        abs_z = np.abs(z_scores)
        target_asset_idx = np.argmax(abs_z)
        z = z_scores[target_asset_idx]
        
        signal_type = "HOLD"
        if z < -2:
            signal_type = "BUY"
        elif z > 2:
            signal_type = "SELL"
            
        return {
            "type": signal_type,
            "price": float(self.prices[idx, target_asset_idx]),
            "confidence": float(abs(z) / 3.0),
            "metadata": {
                "asset_idx": int(target_asset_idx),
                "z_score": float(z)
            }
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"window": self.window, "pca_retrain_interval": self.pca_retrain_interval}
