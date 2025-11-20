import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ======== 1. Tải lại kiến trúc model (Bắt buộc) ========
# PyTorch cần định nghĩa lớp để tải trọng số
z_dim = 100

class Generator(nn.Module):
    def __init__(self, input_dim, z_dim, p_max=0.5):
        super(Generator, self).__init__()
        self.p_max = p_max
        self.net = nn.Sequential(
            nn.Linear(input_dim + z_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, input_dim),
            nn.Tanh()
        )

    def forward(self, real_x, z):
        x = torch.cat([real_x, z], dim=1)
        perturb = self.net(x) * self.p_max
        adv = torch.clamp(real_x + perturb, 0.0, 1.0)
        return adv, perturb

# ======== 2. Cấu hình và đường dẫn ========
DATA_DIR = "/home/thangkb2024/processed"
GEN_PATH = "generator_malgan_best.pt" # Model tốt nhất bạn vừa lưu
BLACKBOX_PATH = os.path.join(DATA_DIR, "blackbox.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ======== 3. Tải Dữ liệu TEST (Dữ liệu "lạ") ========
print("Loading FINAL TEST data (X_test.npy)...")
X_test_all = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test_all = np.load(os.path.join(DATA_DIR, "y_test.npy"))

input_dim = X_test_all.shape[1] 

# Tách riêng các mẫu malware trong tập test
X_test_mal = X_test_all[y_test_all == 1]
X_test_ben = X_test_all[y_test_all == 0]

print(f"Test set loaded: Malware {X_test_mal.shape}, Benign {X_test_ben.shape}")

# ======== 4. Tải Black-Box và Generator ========
print(f"Loading Black-Box model: {BLACKBOX_PATH}")
blackbox = joblib.load(BLACKBOX_PATH)

print(f"Loading BEST Generator: {GEN_PATH}")
G = Generator(input_dim, z_dim).to(device)
G.load_state_dict(torch.load(GEN_PATH, map_location=device))
G.eval() # Chuyển sang chế độ đánh giá

# ======== 5. Đánh giá Baseline (Black-box vs Mẫu Gốc) ========
print("\n" + "="*40)
print(" 1. BASELINE PERFORMANCE (Original Files)")
print("="*40)

preds_mal_original = blackbox.predict(X_test_mal.astype(np.float64))
preds_ben_original = blackbox.predict(X_test_ben.astype(np.float64))

# Tỷ lệ phát hiện (Detection Rate) - Càng cao càng tốt
detection_rate = np.mean(preds_mal_original == 1)
# Tỷ lệ dương tính giả (False Positive Rate) - Càng thấp càng tốt
fp_rate = np.mean(preds_ben_original == 1)

print(f"🔥 Detection Rate (Malware): {detection_rate * 100:.2f}%")
print(f"   (Blackbox phát hiện đúng {detection_rate * 100:.2f}% malware gốc)")
print(f"🔥 False Positive Rate (Benign): {fp_rate * 100:.2f}%")
print(f"   (Blackbox phát hiện nhầm {fp_rate * 100:.2f}% file sạch)")

# ======== 6. Sinh Mẫu Đối Kháng và Đánh giá Evasion Rate ========
print("\n" + "="*40)
print(" 2. ADVERSARIAL PERFORMANCE (Generated Files)")
print("="*40)

# Chuyển dữ liệu malware sang tensor
X_test_mal_t = torch.tensor(X_test_mal, dtype=torch.float32).to(device)

# Sinh mẫu đối kháng
with torch.no_grad(): # Không cần tính gradient
    z_eval = torch.randn(X_test_mal_t.size(0), z_dim, device=device)
    adv_samples, perturb = G(X_test_mal_t, z_eval)
    adv_samples_np = adv_samples.cpu().numpy()

# Đưa mẫu đối kháng vào black-box
preds_adversarial = blackbox.predict(adv_samples_np.astype(np.float64))

# Evasion Rate là % mẫu mã độc (malware) bị black-box dự đoán nhầm là 0 (benign)
evasion_rate = np.mean(preds_adversarial == 0) 

print(f"🚀 EVASION RATE (Malware): {evasion_rate * 100:.2f}%")
print(f"   (Blackbox bị lừa, tin rằng {evasion_rate * 100:.2f}% malware là file sạch)")

# ======== 7. (Optional) Đo lường sự thay đổi (Perturbation) ========
perturb_np = perturb.cpu().numpy()
avg_perturb_l1 = np.mean(np.abs(perturb_np))
avg_perturb_l2 = np.mean(perturb_np**2)
print("\n" + "="*40)
print(" 3. PERTURBATION (Mức độ thay đổi file)")
print("="*40)
print(f"   Average L1 Perturbation: {avg_perturb_l1:.6f}")
print(f"   Average L2 Perturbation: {avg_perturb_l2:.6f}")

# ======== 8. (Optional) Vẽ biểu đồ kết quả ========
try:
    labels = ['Detected (1)', 'Evasion (0)']
    
    # Dữ liệu cho biểu đồ
    original_counts = [np.sum(preds_mal_original == 1), np.sum(preds_mal_original == 0)]
    adversarial_counts = [np.sum(preds_adversarial == 1), np.sum(preds_adversarial == 0)]

    df = pd.DataFrame({
        'Sample Type': ['Original Malware', 'Original Malware', 'Adversarial Malware', 'Adversarial Malware'],
        'Prediction': labels * 2,
        'Count': original_counts + adversarial_counts
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sample Type', y='Count', hue='Prediction', data=df)
    plt.title('Black-Box Performance: Original vs Adversarial (Test Set)')
    plt.ylabel('Number of Samples')
    plt.savefig("evaluation_results.png")
    print(f"\n✅ Đã lưu biểu đồ kết quả vào: evaluation_results.png")

except ImportError:
    print("\n(Vui lòng cài 'pip install pandas matplotlib seaborn' để vẽ biểu đồ)")

print("\n🎉 Đánh giá hoàn tất!")