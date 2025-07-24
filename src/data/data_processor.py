# src/data/data_processor.py

import pandas as pd
import numpy as np
import gc
import warnings
from typing import Dict, List, Tuple, Optional

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
            'slice_id', 'sched_policy_num', 'allocated_rbgs',
            'sum_requested_prbs', 'sum_granted_prbs',
            'prb_utilization', 'throughput_efficiency', 'qos_score',
            'network_load', 'hour', 'minute', 'day_of_week'
        ]
        print(f"Initializing memory optimized processor, batch size: {self.batch_size:,}")

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
        # Create slice_id if it doesn't exist
        if 'slice_id' not in df.columns:
            df['slice_id'] = 0
        df['allocated_rbgs'] = pd.Series(zip(df['training_config'], df['slice_id'])).map(config_map).fillna(0)

        # 4. 資源利用率
        df['prb_utilization'] = np.where(df['sum_requested_prbs'] > 0, df['sum_granted_prbs'] / df['sum_requested_prbs'], 0).clip(0, 1)

        # 5. 吞吐量效率
        df['throughput_efficiency'] = np.where(df['sum_granted_prbs'] > 0, df['tx_brate downlink [Mbps]'].fillna(0) / df['sum_granted_prbs'], 0)

        # 6. QoS 評分 - Use proper missing value handling
        # Use median imputation for missing error rates (conservative approach)
        dl_errors = df.get('tx_errors downlink (%)', pd.Series(dtype='float64')).fillna(2.5)  # Conservative 2.5% error rate
        ul_errors = df.get('rx_errors uplink (%)', pd.Series(dtype='float64')).fillna(2.5)  # Conservative 2.5% error rate
        dl_score = (100 - dl_errors) / 100
        ul_score = (100 - ul_errors) / 100
        
        # Use median CQI value for missing data (typical indoor environment)
        cqi_score = df.get('dl_cqi', pd.Series(dtype='float64')).fillna(10.0) / 15  # Median CQI ~10
        df['qos_score'] = (0.4 * dl_score + 0.3 * ul_score + 0.3 * cqi_score).clip(0, 1)

        # 7. 網路負載 - Use proper missing value handling
        # Use median user count for missing data
        num_ues = df.get('num_ues', pd.Series(dtype='float64')).fillna(5.0)  # Median UE count
        df['network_load'] = num_ues / 42.0

        # 8. 綜合效率指標 (目標變數)
        df['allocation_efficiency'] = (0.5 * df['throughput_efficiency'] + 0.3 * df['qos_score'] + 0.2 * df['prb_utilization']).clip(0, 1)

        return df[self.feature_columns + ['allocation_efficiency']].dropna(subset=['allocation_efficiency'])

    def process_data(self, slice_data):
        """對完整資料集進行分批處理和特徵工程。"""
        print(f"Starting batch processing, total records: {len(slice_data):,}")
        total_rows = len(slice_data)
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        
        processed_batches = []
        for i in range(num_batches):
            print(f"  Processing batch {i + 1}/{num_batches}...")
            batch = slice_data.iloc[i * self.batch_size:(i + 1) * self.batch_size]
            processed_batches.append(self._process_single_batch(batch.copy()))
            gc.collect()
            
        print("Merging all batch results...")
        final_df = pd.concat(processed_batches, ignore_index=True)
        print(f"Feature engineering completed, {len(final_df):,} valid records.")
        return self.optimize_datatypes(final_df)

    @staticmethod
    def optimize_datatypes(df):
        """最佳化資料型別以節省記憶體。"""
        print("Optimizing data types...")
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory optimization: {initial_memory:.1f} MB -> {final_memory:.1f} MB (saved {((initial_memory - final_memory) / initial_memory * 100):.1f}%)")
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Comprehensive data quality validation and reporting."""
        print("Executing data quality validation...")
        
        quality_report = {
            'total_records': len(df),
            'missing_values': {},
            'outliers': {},
            'data_ranges': {},
            'duplicates': 0,
            'quality_score': 0.0
        }
        
        # Check for missing values
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            quality_report['missing_values'][col] = missing_pct
            
        # Check for outliers using IQR method
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = len(df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))])
            quality_report['outliers'][col] = (outlier_count / len(df)) * 100
            
        # Check data ranges
        for col in numeric_cols:
            quality_report['data_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
            
        # Check for duplicates
        quality_report['duplicates'] = df.duplicated().sum()
        
        # Calculate overall quality score
        avg_missing = np.mean(list(quality_report['missing_values'].values()))
        avg_outliers = np.mean(list(quality_report['outliers'].values()))
        duplicate_pct = (quality_report['duplicates'] / len(df)) * 100
        
        quality_score = max(0, 100 - avg_missing - avg_outliers - duplicate_pct)
        quality_report['quality_score'] = quality_score
        
        print(f"Data quality score: {quality_score:.1f}/100")
        print(f"   - Average missing values: {avg_missing:.1f}%")
        print(f"   - Average outliers: {avg_outliers:.1f}%")
        print(f"   - Duplicate records: {quality_report['duplicates']:,} ({duplicate_pct:.1f}%)")
        
        return quality_report
    
    def clean_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Remove or cap outliers based on specified method."""
        print(f"Cleaning outliers (method: {method}, threshold: {threshold})...")
        
        cleaned_df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers instead of removing them
                cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                cleaned_df = cleaned_df[z_scores < threshold]
                
        print(f"Outlier cleaning completed, kept {len(cleaned_df):,} records")
        return cleaned_df

