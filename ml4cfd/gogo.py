import os
import json
import time
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyvista as pv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from scipy.spatial import cKDTree
from tqdm import tqdm

# ==============================================================================
# 🌙 [수면 모드] 밤샘 학습용 최종 설정 (GTX 1660 Super)
# ==============================================================================
NUM_SAMPLES = 10000     # [상향] 데이터 10,000개 (밤새 충분히 학습할 양)
SAMPLING_RATE = 50      # [유지] 고해상도 (타협 없음, 정밀도 최우선)
K_NEIGHBORS = 30        # [유지]
BATCH_SIZE = 1          # [유지] 메모리 터짐 방지
ACCUMULATION_STEPS = 16 # [유지] 학습 안정성 확보
EPOCHS = 200            # [설정] 데이터가 많으므로 200 에폭이면 충분함

HIDDEN_DIM = 128        # 모델 크기
LEARNING_RATE = 0.001   # 시작 학습률
# ==============================================================================

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, tensor_data):
        self.mean = torch.mean(tensor_data, dim=0)
        self.std = torch.std(tensor_data, dim=0)
        self.std[self.std == 0] = 1.0
    def transform(self, tensor_data):
        return (tensor_data - self.mean.to(tensor_data.device)) / self.std.to(tensor_data.device)
    def inverse_transform(self, tensor_data):
        return (tensor_data * self.std.to(tensor_data.device)) + self.mean.to(tensor_data.device)

def manual_knn_graph(pos, k):
    tree = cKDTree(pos)
    _, indices = tree.query(pos, k=k+1)
    indices = indices[:, 1:]
    source_nodes = np.repeat(np.arange(len(pos)), k)
    target_nodes = indices.flatten()
    return torch.tensor(np.stack([source_nodes, target_nodes], axis=0), dtype=torch.long)

def extract_params_from_name(folder_name):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", folder_name)
    params = [float(n) for n in numbers if '.' in n or n.isdigit()]
    if len(params) > 7: params = params[-7:]
    else: params = params + [0.0] * (7 - len(params))
    return np.array(params, dtype=np.float32)

def load_dataset(root_path, manifest_path):
    if not os.path.exists(manifest_path): return []
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    folder_names = manifest.get('full_train', [])[:NUM_SAMPLES]
    dataset = []
    
    print(f"\nzzz [수면 모드] 데이터 로드 시작 ({NUM_SAMPLES}개)... 시간이 좀 걸립니다.")
    for name in tqdm(folder_names):
        path = os.path.join(root_path, name)
        if not os.path.isdir(path): continue
        try:
            vtu_files = [f for f in os.listdir(path) if 'internal.vtu' in f.lower()]
            if not vtu_files: continue
            mesh = pv.read(os.path.join(path, vtu_files[0]))
            
            idx = np.arange(0, mesh.n_points, SAMPLING_RATE)
            pos = mesh.points[idx, :2] 
            keys = mesh.point_data.keys()
            p = mesh.point_data['p'][idx] if 'p' in keys else mesh.point_data['pressure'][idx]
            u = mesh.point_data['U'][idx, :2] if 'U' in keys else mesh.point_data['velocity'][idx, :2]
            
            params = extract_params_from_name(name)
            params_repeated = np.tile(params, (len(pos), 1))
            x_features = np.concatenate([pos, params_repeated], axis=1)
            
            dataset.append(Data(
                x=torch.tensor(x_features, dtype=torch.float),
                edge_index=manual_knn_graph(pos, k=K_NEIGHBORS),
                y=torch.tensor(np.concatenate([p.reshape(-1, 1), u], axis=-1), dtype=torch.float)
            ))
        except: continue
    print(f"✅ 로드 완료: {len(dataset)}개")
    return dataset

class AirfoilGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(AirfoilGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.skip_proj = nn.Linear(in_channels, hidden_channels * 4)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels * 2, heads=2, concat=True)
        self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.silu(self.conv1(x, edge_index) + self.skip_proj(x))
        x = F.silu(self.conv2(x, edge_index))
        x = F.silu(self.conv3(x, edge_index))
        return self.decoder(x)

def calculate_score(metrics_dict, speed_up):
    thresholds = {"x-velocity": (0.01, 0.02), "y-velocity": (0.01, 0.02), "pressure": (0.002, 0.01)}
    val_by_color = {"green": 2, "orange": 1, "red": 0}
    total_points = 0
    vars = metrics_dict["ML"]
    for k, v in vars.items():
        t_g, t_o = thresholds[k]
        if v < t_g: color = "green"
        elif v < t_o: color = "orange"
        else: color = "red"
        total_points += val_by_color[color]
    return (total_points / 6 * 0.75 + max(min(math.log10(speed_up) / 4, 1), 0) * 0.25) * 100

if __name__ == "__main__":
    BASE_DIR = r"C:\Users\dlals\Desktop\NeurIPS2024-ML4CFD-competition-Starting-Kit-main"
    DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
    MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")
    
    dataset = load_dataset(DATASET_DIR, MANIFEST_PATH)
    
    if dataset:
        INPUT_DIM = dataset[0].x.shape[1]
        x_norm, y_norm = Normalizer(), Normalizer()
        x_norm.fit(torch.cat([d.x for d in dataset], dim=0))
        y_norm.fit(torch.cat([d.y for d in dataset], dim=0))
        
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AirfoilGAT(INPUT_DIM, HIDDEN_DIM, 3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Patience를 20으로 설정하여 진득하게 기다림
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        
        print(f"\n🌙 밤샘 학습 시작! 주인님 안녕히 주무세요. (Epochs={EPOCHS})")
        model.train()
        
        for epoch in range(1, EPOCHS + 1):
            total_loss = 0
            optimizer.zero_grad()
            
            for i, batch in enumerate(loader):
                batch = batch.to(device)
                batch.x = x_norm.transform(batch.x)
                y_target = y_norm.transform(batch.y)
                
                pred = model(batch)
                loss = F.mse_loss(pred, y_target)
                loss = loss / ACCUMULATION_STEPS 
                loss.backward()
                
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * ACCUMULATION_STEPS
            
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
            
            # 진행 상황 출력 (10 에폭마다 or 첫 에폭)
            if epoch % 10 == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

        # --- [추가됨] 아침에 일어나서 쓸 모델 저장 ---
        print("\n💾 학습 완료! 모델을 저장합니다: best_model.pth")
        torch.save(model.state_dict(), "best_model.pth")

        # 평가 및 점수 계산
        print("🏆 최종 점수 계산 중...")
        model.eval()
        mse_sum, total_inf_time = np.zeros(3), 0
        with torch.no_grad():
            for data in dataset:
                data = data.to(device)
                t0 = time.time()
                data.x = x_norm.transform(data.x)
                pred = model(data)
                total_inf_time += (time.time() - t0)
                mse_sum += torch.mean((pred - y_norm.transform(data.y))**2, dim=0).cpu().numpy()
        
        avg_mse = mse_sum / len(dataset)
        score = calculate_score({"ML": {"pressure": avg_mse[0], "x-velocity": avg_mse[1], "y-velocity": avg_mse[2]}}, (100.0 * len(dataset)) / total_inf_time)
        
        print("-" * 50)
        print(f"📊 [아침 결과 확인]")
        print(f"   • Pressure   : {avg_mse[0]:.6f}")
        print(f"   • X-Velocity : {avg_mse[1]:.6f}")
        print(f"   • Y-Velocity : {avg_mse[2]:.6f}")
        print(f"🏆 Score : {score:.2f} / 100")
        print("-" * 50)