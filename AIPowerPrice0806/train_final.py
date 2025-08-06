# train.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import time
import os
import math
import warnings

# --- 0. 全局设置 (保持不变) ---
warnings.simplefilter(action='ignore', category=FutureWarning)
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 1. 参数配置 (为最终模型调整) ---
class Config:
    N_SAMPLES = 5000; INPUT_SEQ_LEN = 168; OUTPUT_SEQ_LEN = 24; N_FEATURES = 7

    MODEL_DIM = 128
    NUM_HEADS = 8        # 头数
    NUM_ENCODER_LAYERS = 4 # Transformer
    GRU_HIDDEN_SIZE = 128
    GRU_LAYERS = 2       # GRU
    
    DROPOUT_RATE = 0.25; BATCH_SIZE = 32; EPOCHS = 100 # 增加dropout和epochs
    LEARNING_RATE = 0.0003; TRAIN_SPLIT = 0.8; VAL_SPLIT = 0.1; SEED = 42
config = Config()
set_seed(config.SEED)

# --- 数据准备与预处理 (保持不变) ---
def generate_mock_data(n_samples):
    print("Step 1: Generating mock data...")
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=n_samples, freq='h'))
    time_component = np.arange(n_samples); daily_cycle = np.sin(2 * np.pi * time_component / 24); weekly_cycle = np.sin(2 * np.pi * time_component / (24 * 7))
    base_load = 20000 + 5000*daily_cycle + 3000*weekly_cycle + np.random.normal(0, 250, n_samples)
    capacity = np.full(n_samples, 40000) + np.random.normal(0, 200, n_samples)
    base_price = 30 + 20*(base_load/capacity)*2 + 10*daily_cycle + np.random.normal(0, 2, n_samples)
    spike_indices = np.random.choice(n_samples, size=int(n_samples*0.02), replace=False)
    base_price[spike_indices] *= np.random.uniform(2, 4, size=len(spike_indices))
    price = np.clip(base_price, 5, 300)
    df = pd.DataFrame({'timestamp': dates, 'price': price, 'load': base_load, 'capacity': capacity})
    df['price_delta'] = df['price'].diff().fillna(0); df['load_delta'] = df['load'].diff().fillna(0); df['load_capacity_ratio'] = df['load'] / df['capacity']; df['predicted_load'] = df['load'].rolling(window=5, min_periods=1).mean() + np.random.normal(0, 150, n_samples); df['hour'] = df['timestamp'].dt.hour
    feature_cols = ['price', 'price_delta', 'load', 'load_delta', 'load_capacity_ratio', 'predicted_load', 'hour']
    return df[feature_cols], df['price']

def preprocess_data(features, target_price):
    print("Step 2: Preprocessing data...")
    feature_scalers, scaled_features = {}, features.copy()
    for col in features.columns:
        scaler = MinMaxScaler(); scaled_features[col] = scaler.fit_transform(features[[col]]); feature_scalers[col] = scaler
    target_scaler = MinMaxScaler(); scaled_target = target_scaler.fit_transform(target_price.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_features) - config.INPUT_SEQ_LEN - config.OUTPUT_SEQ_LEN + 1):
        X.append(scaled_features.iloc[i : i + config.INPUT_SEQ_LEN].values); y.append(scaled_target[i + config.INPUT_SEQ_LEN : i + config.INPUT_SEQ_LEN + config.OUTPUT_SEQ_LEN].flatten())
    X, y = np.array(X), np.array(y)
    train_size, val_size = int(len(X) * config.TRAIN_SPLIT), int(len(X) * config.VAL_SPLIT)
    X_train, y_train = X[:train_size], y[:train_size]; X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]; X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    return (X_train, y_train, X_val, y_val, X_test, y_test), feature_scalers, target_scaler

# --- 4. 模型定义 (核心改动: 最终深度模型) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__(); self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model); position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1); self.register_buffer('pe', pe)
    def forward(self, x): x = x + self.pe[:x.size(0), :]; return self.dropout(x)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__(); self.W = nn.Linear(hidden_size, hidden_size); self.v = nn.Parameter(torch.rand(hidden_size)); self.softmax = nn.Softmax(dim=1)
    def forward(self, features): # Can be used on GRU or Transformer output
        uh = torch.tanh(self.W(features)); scores = torch.matmul(uh, self.v); attention_weights = self.softmax(scores)
        context = torch.bmm(attention_weights.unsqueeze(1), features).squeeze(1); return context

class UltimateHybridModel(nn.Module):
    def __init__(self, n_features, model_dim, num_heads, num_encoder_layers, gru_hidden_size, gru_layers, dropout, output_seq_len):
        super().__init__()

        self.input_projection = nn.Linear(n_features, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*4, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.gru = nn.GRU(model_dim, gru_hidden_size, gru_layers, batch_first=True, dropout=dropout, bidirectional=True)

        self.transformer_attention = TemporalAttention(model_dim)
        
        # 融合层
        # GRU是双向的，所以输出维度是 2 * gru_hidden_size
        # Transformer注意力输出维度是 model_dim

        fused_dim = 2 * gru_hidden_size + model_dim
        self.output_layer = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.BatchNorm1d(fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, output_seq_len)
        )
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # 1. 投影和位置编码
        x_projected = self.input_projection(x)
        x_pos = self.pos_encoder(x_projected.permute(1, 0, 2)).permute(1, 0, 2)
        
        # 2. Transformer Encoder (带残差连接)
        transformer_output = self.transformer_encoder(x_pos)
        transformer_output = x_pos + transformer_output # 残差连接
        
        # 3. GRU
        gru_output, _ = self.gru(transformer_output) # gru_output shape: (batch, seq_len, 2 * gru_hidden_size)
        
        # 4. 信息融合
        # a. 从GRU输出中提取最后一个时间步的隐藏状态
        # gru_output[:, -1, :] 包含了前向和后向的最后一个隐藏状态
        gru_last_hidden = gru_output[:, -1, :] 
        
        # b. 对Transformer的输出使用时间注意力，得到全局上下文
        transformer_context = self.transformer_attention(transformer_output)
        
        # c. 拼接两种信息
        fused_features = torch.cat([gru_last_hidden, transformer_context], dim=1)
        
        # 5. 输出
        prediction = self.output_layer(fused_features)
        return prediction

# --- 训练、评估、可视化函数 (保持不变, 但训练函数中更换学习率调度器) ---
def train_model(model, train_loader, val_loader, initial_lr, device):
    print("\nStep 5: Training the model...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-3) # AdamW更好
    # 使用余弦退火学习率，让模型在后期能更精细地搜索
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-7)
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf'); epochs_no_improve = 0; patience_early_stop = 20 # 更大的耐心

    for epoch in range(config.EPOCHS):
        model.train(); total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch); loss = criterion(y_pred, y_batch)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        model.eval(); total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                total_val_loss += criterion(model(X_batch), y_batch).item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.7f}")
        
        scheduler.step() # 余弦退火调度器每步都更新
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth'); print(f"  New best val loss: {best_val_loss:.6f}. Saving model.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience_early_stop:
            print(f"\nEarly stopping after {patience_early_stop} epochs."); break
            
    print("Loading best model weights."); model.load_state_dict(torch.load('best_model.pth')); return history

def evaluate_model(model, test_loader, scaler, device):
    # (此函数无需更改)
    print("\nStep 6: Evaluating the model...")
    model.eval(); predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device); y_pred = model(X_batch)
            predictions.extend(scaler.inverse_transform(y_pred.cpu().numpy())); actuals.extend(scaler.inverse_transform(y_batch.numpy()))
    predictions, actuals = np.array(predictions), np.array(actuals)
    pred_flat, act_flat = predictions.flatten(), actuals.flatten()
    rmse = np.sqrt(np.mean((pred_flat - act_flat) ** 2)); mae = np.mean(np.abs(pred_flat - act_flat))
    denominator = np.abs(act_flat) + np.abs(pred_flat)
    smape = (np.sum(np.divide(2 * np.abs(pred_flat - act_flat), denominator, out=np.zeros_like(denominator), where=denominator != 0)) / len(act_flat)) * 100
    print(f"Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}, Test SMAPE: {smape:.4f}%")
    return predictions, actuals

def plot_results(actuals, predictions, history):
    # (此函数无需更改)
    print("\nStep 7: Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid'); plt.figure(figsize=(12, 5)); plt.plot(history['train_loss'], label='Training Loss'); plt.plot(history['val_loss'], label='Validation Loss'); plt.title('Model Loss'); plt.legend(); plt.savefig("training_loss.png"); plt.show()
    plt.figure(figsize=(15, 7)); plot_len = 24 * 7; plt.plot(actuals.flatten()[:plot_len], label='Actual'); plt.plot(predictions.flatten()[:plot_len], label='Predicted'); plt.title('Price Prediction vs Actual'); plt.legend(); plt.savefig("prediction_vs_actual.png"); plt.show()


# --- 主程序入口 ---
if __name__ == '__main__':
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    features, target_price = generate_mock_data(config.N_SAMPLES)
    (X_train, y_train, X_val, y_val, X_test, y_test), feature_scalers, target_scaler = preprocess_data(features, target_price)
    
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_data = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True) # 使用pin_memory加速
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    print("\nStep 3: Initializing model...")
    model = UltimateHybridModel(
        n_features=config.N_FEATURES, model_dim=config.MODEL_DIM, num_heads=config.NUM_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS, gru_hidden_size=config.GRU_HIDDEN_SIZE,
        gru_layers=config.GRU_LAYERS, dropout=config.DROPOUT_RATE, output_seq_len=config.OUTPUT_SEQ_LEN
    ).to(device)
    
    print(model); num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    history = train_model(model, train_loader, val_loader, config.LEARNING_RATE, device)
    predictions, actuals = evaluate_model(model, test_loader, target_scaler, device)
    plot_results(actuals, predictions, history)

    print("\nStep 8: Saving final model and scalers...")
    save_dir, save_path = "saved_model", os.path.join("saved_model", "model_and_scalers_ultimate.pth")
    os.makedirs(save_dir, exist_ok=True)

    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_scalers': feature_scalers,
        'target_scaler': target_scaler,
        'config_dict': config_dict # <--- 修改为保存字典
    }, save_path)

    print(f"Ultimate model and config dictionary saved to {save_path}")
    
    
    print(f"Ultimate model saved to {save_path}")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
