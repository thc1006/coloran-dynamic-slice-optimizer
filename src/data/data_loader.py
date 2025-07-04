# src/data/data_loader.py

import os
import subprocess
import pandas as pd
import glob
import warnings

warnings.filterwarnings('ignore')

class ColoRANDataLoader:
    """
    ColO-RAN 資料集載入器。
    負責從遠端儲存庫下載資料集，並將原始 CSV 檔案載入到 pandas DataFrame 中。
    """
    def __init__(self, base_path="/content"):
        self.dataset_repo_url = "https://github.com/wineslab/colosseum-oran-coloran-dataset.git"
        self.dataset_local_path = os.path.join(base_path, "colosseum-oran-coloran-dataset")
        self.base_stations = [1, 8, 15, 22, 29, 36, 43]
        self.scheduling_policies = ['sched0', 'sched1', 'sched2'] # RR, WF, PF
        self.training_configs = [f'tr{i}' for i in range(28)]
        print(f"🚀 初始化 ColoRANDataLoader，資料路徑設定為: {self.dataset_local_path}")

    def download_dataset(self):
        """下載 ColO-RAN 資料集，如果本地不存在。"""
        print("檢查 ColO-RAN 資料集...")
        if os.path.exists(self.dataset_local_path):
            print(f"✅ 資料夾 '{self.dataset_local_path}' 已存在，跳過下載。")
            return self.dataset_local_path

        print("資料集不存在，開始下載...")
        try:
            result = subprocess.run(
                ["git", "clone", self.dataset_repo_url, self.dataset_local_path],
                capture_output=True, text=True, timeout=600, check=True
            )
            print("✅ 資料集下載成功！")
            print("\n📁 資料集結構預覽：")
            os.system(f"ls -la {self.dataset_local_path}")
            return self.dataset_local_path
        except subprocess.TimeoutExpired:
            print("❌ 下載超時，請檢查網路連線。")
        except subprocess.CalledProcessError as e:
            print(f"❌ Git clone 失敗: {e.stderr}")
        except Exception as e:
            print(f"❌ 下載過程中發生錯誤: {e}")
        return None

    def _auto_detect_structure(self):
        """自動偵測有效的資料集結構路徑。"""
        possible_paths = [
            os.path.join(self.dataset_local_path, "rome_static_medium"),
            self.dataset_local_path,
        ]
        for path in possible_paths:
            if os.path.exists(path) and any(d.startswith('sched') for d in os.listdir(path)):
                print(f"✅ 找到有效資料結構: {path}")
                return path
        print("❌ 未找到標準資料結構。")
        return None

    def load_raw_data(self):
        """使用 glob 載入所有原始資料。"""
        base_data_path = self._auto_detect_structure()
        if not base_data_path:
            return None, None, None

        slice_data_list = []
        total_combinations = len(self.scheduling_policies) * len(self.training_configs) * len(self.base_stations)
        print(f"\n🚀 開始載入完整 ColO-RAN 資料集（{total_combinations} 個組合）")

        for sched_policy in self.scheduling_policies:
            for training_config in self.training_configs:
                search_pattern = f"{base_data_path}/{sched_policy}/{training_config}/exp*/bs*/slices_bs*/*_metrics.csv"
                slice_files = glob.glob(search_pattern)
                for slice_file in slice_files:
                    try:
                        df = pd.read_csv(slice_file)
                        path_parts = slice_file.split('/')
                        exp_folder = next((p for p in path_parts if p.startswith('exp')), 'exp1')
                        bs_folder = next((p for p in path_parts if p.startswith('bs') and 'slices' not in p), 'bs1')
                        df['bs_id'] = int(bs_folder.replace('bs', ''))
                        df['exp_id'] = int(exp_folder.replace('exp', ''))
                        df['imsi'] = os.path.basename(slice_file).replace('_metrics.csv', '')
                        df['sched_policy'] = sched_policy
                        df['training_config'] = training_config
                        slice_data_list.append(df)
                    except Exception as e:
                        print(f" ❌ Slice 檔案載入失敗 {slice_file}: {e}")

        if not slice_data_list:
            print("❌ 未能載入任何切片資料。")
            return None

        print("\n🔗 合併資料中...")
        combined_slice_data = pd.concat(slice_data_list, ignore_index=True)
        memory_mb = combined_slice_data.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"✅ 切片資料載入完成: {len(combined_slice_data):,} 筆記錄, {memory_mb:.1f} MB")
        return combined_slice_data

if __name__ == '__main__':
    # 範例使用
    data_loader = ColoRANDataLoader()
    data_loader.download_dataset()
    slice_data = data_loader.load_raw_data()
    if slice_data is not None:
        print("\n資料載入器測試成功，已載入切片資料。")
        print(slice_data.head())

