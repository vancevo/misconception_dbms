import os
import pandas as pd
from datasets import Dataset

# 1. Điền Token của bạn (Lấy tại: https://huggingface.co/settings/tokens)
# Lưu ý: Token này phải có quyền "WRITE"
HF_TOKEN = "YOUR_HF_TOKEN" 

# 2. Điền tên repository trên HuggingFace của bạn
# Ví dụ: "vancevo/misconception_mining_asag"
REPO_ID = "vancevo/misconception_mining"

# Đường dẫn tới file dữ liệu gốc trên máy của bạn
CSV_FILE_PATH = "../../data-generate.csv"

def upload_dataset():
    print(f"Đang đọc dữ liệu từ {CSV_FILE_PATH}...")
    try:
        # Đọc file CSV bằng Pandas
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Đã đọc {len(df)} dòng dữ liệu.")
        
        # Chuyển đổi DataFrame sang định dạng Dataset của HuggingFace
        hf_dataset = Dataset.from_pandas(df)
        
        print(f"Đang đẩy dữ liệu lên HuggingFace Hub: {REPO_ID}...")
        # Đẩy lên hub
        hf_dataset.push_to_hub(
            repo_id=REPO_ID,
            token=HF_TOKEN,
            private=False # Set thành True nếu bạn muốn giấu dataset đi
        )
        print("✅ TẢI LÊN THÀNH CÔNG! Dataset của bạn đã có trên HuggingFace.")
        print(f"👉 Xem tại: https://huggingface.co/datasets/{REPO_ID}")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    # Đảm bảo thư mục được chạy đúng chỗ
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    upload_dataset()
