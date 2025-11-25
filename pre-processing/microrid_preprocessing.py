import pandas as pd
import numpy as np
import os

class MicrogridPreprocessor:
    def __init__(self, uci_path=None):
        self.uci_path = uci_path
        # Bengaluru, India
        self.LATITUDE = 12.9716 
        self.SOLAR_CAPACITY_KW = 4.0  # Typical residential system
        self.PANEL_EFFICIENCY = 0.18  # 18% efficient panels

    # ==========================================
    # 🛠️ HELPER: PHYSICS-BASED SOLAR GENERATION
    # ==========================================
    def calculate_solar_physics(self, df):
        """
        Generates Irradiance and Solar Generation using a simplified 
        Clear Sky Radiation Model based on time of day and year.
        """
        # 1. Create Time Features
        # Day of year (1-365) and Hour (0-23)
        day_of_year = df.index.dayofyear
        hour_of_day = df.index.hour + (df.index.minute / 60.0)
        
        # 2. Simulate Solar Elevation Angle (Simplified)
        # declination angle (delta): varies between -23.45 and +23.45
        delta = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle (h): 15 degrees per hour from solar noon (assumed 12:00)
        h = 15 * (hour_of_day - 12)
        
        # Elevation angle (alpha)
        lat_rad = np.radians(self.LATITUDE)
        delta_rad = np.radians(delta)
        h_rad = np.radians(h)
        
        sin_alpha = (np.sin(lat_rad) * np.sin(delta_rad)) + \
                    (np.cos(lat_rad) * np.cos(delta_rad) * np.cos(h_rad))
        
        # Clamp negative values (night time) to 0
        sin_alpha = np.maximum(sin_alpha, 0)
        
        # 3. Calculate Clear Sky Irradiance (W/m^2)
        # Max theoretical is ~1000 W/m^2 at noon
        # We add some randomness for "Cloud Cover"
        clear_sky_irradiance = 1000 * sin_alpha
        
        # 4. Generate Cloud Cover (0.0 to 1.0, where 1.0 is full sun, 0.2 is dark cloud)
        # We simulate weather patterns with random walks to make it realistic (clouds stick around)
        noise = np.random.normal(0, 0.1, len(df))
        cloud_factor = np.clip(0.8 + np.cumsum(noise) * 0.05, 0.1, 1.0)
        
        # Reset cloud factor occasionally to prevent drifting to infinity
        if len(df) > 0:
            cloud_factor = np.clip(np.random.beta(5, 2, len(df)), 0.1, 1.0)

        # Final Irradiance
        df['irradiance'] = clear_sky_irradiance * cloud_factor
        df['cloud_cover'] = (1 - cloud_factor) * 100 # percentage

        # 5. Calculate Generation (kWh)
        # Formula: Capacity (kW) * (Irradiance / 1000) * EfficiencyLosses
        # We assume the capacity is rated at 1000 W/m^2
        df['generation_kwh'] = self.SOLAR_CAPACITY_KW * (df['irradiance'] / 1000)
        
        return df

    # ==========================================
    # 🛠️ HELPER: MOCK UCI DATA
    # ==========================================
    def generate_mock_uci(self):
        print("⚠️ No UCI file provided. Generating synthetic demand data...")
        dates = pd.date_range(start="2008-01-01", end="2008-01-10 23:00:00", freq="H")
        df = pd.DataFrame(index=dates)
        # Synthetic daily demand curve
        hour = df.index.hour
        base_load = 0.5
        peak_load = (hour > 17) & (hour < 22)
        df['demand_kwh'] = base_load + (peak_load * 1.5) + np.random.normal(0, 0.1, len(df))
        df['demand_kwh'] = df['demand_kwh'].clip(lower=0.1)
        return df

    # ==========================================
    # ⭐ MAIN PIPELINE
    # ==========================================
    def run_pipeline(self):
        print("Step 1: Loading Demand Data...")
        
        if self.uci_path and os.path.exists(self.uci_path):
            # Load Real UCI
            df = pd.read_csv(self.uci_path, sep=";", na_values="?", low_memory=False)
            
            # 🛠️ FIX: Normalize headers to lowercase to avoid KeyError (Global_active_power vs global_active_power)
            df.columns = df.columns.str.strip().str.lower()
            print(f"   - Columns found: {df.columns.tolist()}")

            # Updated to use lowercase column names
            df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d/%m/%Y %H:%M:%S")
            df = df.set_index("timestamp")
            df["demand_kwh"] = df["global_active_power"].astype(float) / 60
            
            # Resample to Hourly
            df = df["demand_kwh"].resample("H").sum().to_frame()
            
            # Filter for a clean 1-year period (e.g., 2008)
            # You can expand this range to '2007-01-01':'2010-12-31' for more data
            df = df.loc['2006-12-16':'2008-11-26']

            # 🛠️ MODERNIZATION STEP
            # Multiply by 1.2 to simulate 2025 usage levels (approx +20% vs 2008)
            print("   - Applying 1.2x factor for modern appliance usage...")
            df["demand_kwh"] = df["demand_kwh"] * 1.2
            
        else:
            df = self.generate_mock_uci()

        print("Step 2: Simulating Physics-Based Solar Generation...")
        df = self.calculate_solar_physics(df)

        print("Step 3: Generating Temperature...")
        # FIX: Align seasonal peak to April/May (Bengaluru Summer)
        # Bengaluru Peak heat is around April (Day 110)
        # Bengaluru is much warmer: Base ~24°C, varies by +/- 5°C seasonally
        day_of_year = df.index.dayofyear
        hour_of_day = df.index.hour
        
        # 1. Seasonal Component (Warmer overall, peak in April/May)
        # Shift peak to Day 100 (April)
        seasonal_temp = 24 + 5 * np.sin((day_of_year - 20) * 2 * np.pi / 365)
        
        # 2. Diurnal Component (Day/Night cycle)
        daily_temp = 6 * np.sin((hour_of_day - 9) * 2 * np.pi / 24)
        
        # 3. Random Weather Noise
        noise = np.random.normal(0, 2, len(df))
        
        df["temperature"] = seasonal_temp + daily_temp + noise

        print("Step 4: Engineering Features...")
        df["hour"] = df.index.hour
        df["month"] = df.index.month
        df["dayofweek"] = df.index.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        
        # Lag Features
        df["demand_lag_1"] = df["demand_kwh"].shift(1)
        df["generation_lag_1"] = df["generation_kwh"].shift(1)
        
        # Static Specs
        df["solar_capacity"] = self.SOLAR_CAPACITY_KW
        df["battery_capacity"] = 5.0 # kWh

        df = df.dropna()
        print("\n✅ PREPROCESSING COMPLETE!")
        print(f"Final Shape: {df.shape}")
        return df

# ==========================================
# 🚀 EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    uci_filename = "household_power_consumption.txt"

    if os.path.exists(uci_filename):
        print(f"📂 Found real UCI dataset.")
        processor = MicrogridPreprocessor(uci_path=uci_filename)
    else:
        processor = MicrogridPreprocessor()
    
    final_df = processor.run_pipeline()
    
    pd.set_option('display.max_columns', None) 
    print("\n--- DATA PREVIEW ---")
    print(final_df.head(12)) # Check first 12 hours
    print("\n--- STATISTICS ---")
    print(final_df[['demand_kwh', 'generation_kwh', 'irradiance', 'temperature']].describe())
    
    # SAVE THE FILE
    final_df.to_csv("processed_microgrid_data.csv")
    print("\n💾 Saved to 'processed_microgrid_data.csv'")