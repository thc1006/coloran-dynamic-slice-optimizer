# src/data/data_processor.py

import pandas as pd
import numpy as np
import gc
import warnings

warnings.filterwarnings('ignore')

class MemoryOptimizedProcessor:
    """
    對載入的網路切片資料進行記憶體優化的特徵工程。
    - 建立時間、資源、QoS 等多維度特徵。
    - 採用分批處理以應對大規模資料集。
    - 最佳化資料型別以降低記憶體佔用。
    """
    def __init__(self, slice_configs, batch_size=75000):
        self.slice_configs = slice_configs
        self.batch_size = batch_size
        self.feature_columns = [
            'num_ues', 'slice_id', 'sched_policy_num', 'allocated_rbgs',
            'bs_id', 'exp_id', 'sum_requested_prbs', 'sum_granted_prbs',
            'prb_utilization', 'throughput_efficiency', 'qos_score',
            'network_load', 'hour', 'minute', 'day_of_week'
        ]
        print(f"🔧 初始化記憶體優化處理器，批次大小: {self.batch_size:,}")

    def _process_single_batch(self, df):
        """處理單一批次的資料，建立所有工程特徵。"""
        # 1. 時間特徵
        timestamps = pd.to_datetime(df['Timestamp'], unit='ms', errors='coerce')
        df['hour'] = timestamps.dt.hour
        df['minute'] = timestamps.dt.minute
        df['day_of_week'] = timestamps.dt.dayofweek

        # 2. 排程策略編碼
        df['sched_policy_num'] = df['sched_policy'].map({'sched0': 0, 'sched1': 1, 'sched2': 2})

        # 3. RBG 配置
        config_map = {(cfg, i): rbg for cfg, rbgs in self.slice_configs.items() for i, rbg in enumerate(rbgs)}
        df['allocated_rbgs'] = pd.Series(zip(df['training_config'], df.get('slice_id', 0))).map(config_map).fillna(0)

        # 4. 資源利用率
        df['prb_utilization'] = np.where(df['sum_requested_prbs'] > 0, df['sum_granted_prbs'] / df['sum_requested_prbs'], 0).clip(0, 1)

        # 5. 吞吐量效率
        df['throughput_efficiency'] = np.where(df['sum_granted_prbs'] > 0, df['tx_brate downlink [Mbps]'].fillna(0) / df['sum_granted_prbs'], 0)

        # 6. QoS 評分
        dl_score = (100 - df['tx_errors downlink (%)'].fillna(50)) / 100
        ul_score = (100 - df['rx_errors uplink (%)'].fillna(50)) / 100
        cqi_score = df['dl_cqi'].fillna(7.5) / 15
        df['qos_score'] = (0.4 * dl_score + 0.3 * ul_score + 0.3 * cqi_score).clip(0, 1)

        # 7. 網路負載
        df['network_load'] = df.get('num_ues', 1).fillna(1) / 42.0

        # 8. 綜合效率指標 (目標變數)
        df['allocation_efficiency'] = (0.5 * df['throughput_efficiency'] + 0.3 * df['qos_score'] + 0.2 * df['prb_utilization']).clip(0, 1)

        return df[self.feature_columns + ['allocation_efficiency']].dropna(subset=['allocation_efficiency'])

    def process_data(self, slice_data):
        """對完整資料集進行分批處理和特徵工程。"""
        print(f"🚀 開始分批處理，總記錄數: {len(slice_data):,}")
        total_rows = len(slice_data)
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        
        processed_batches = []
        for i in range(num_batches):
            print(f" 📦 處理批次 {i + 1}/{num_batches}...")
            batch = slice_data.iloc[i * self.batch_size:(i + 1) * self.batch_size]
            processed_batches.append(self._process_single_batch(batch.copy()))
            gc.collect()
            
        print("🔗 合併所有批次結果...")
        final_df = pd.concat(processed_batches, ignore_index=True)
        print(f"✅ 特徵工程完成，共 {len(final_df):,} 筆有效記錄。")
        return self.optimize_datatypes(final_df)

    @staticmethod
    def optimize_datatypes(df):
        """最佳化資料型別以節省記憶體。"""
        print("🔧 最佳化資料型別...")
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        print(f" 💾 記憶體最佳化: {initial_memory:.1f} MB → {final_memory:.1f} MB (節省 {((initial_memory - final_memory) / initial_memory * 100):.1f}%)")
        return df

