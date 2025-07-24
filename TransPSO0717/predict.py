# predict.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import math

# ==============================================================================
# 1. 定义模型结构 (必须与训练时完全一致)
#    将训练脚本中的模型定义代码完整地复制过来
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        uh = torch.tanh(self.W(features))
        scores = torch.matmul(uh, self.v)
        attention_weights = self.softmax(scores)
        context = torch.bmm(attention_weights.unsqueeze(1), features).squeeze(1)
        return context

class UltimateHybridModel(nn.Module):
    def __init__(self, n_features, model_dim, num_heads, num_encoder_layers, gru_hidden_size, gru_layers, dropout, output_seq_len):
        super().__init__()
        self.input_projection = nn.Linear(n_features, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*4, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.gru = nn.GRU(model_dim, gru_hidden_size, gru_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.transformer_attention = TemporalAttention(model_dim)
        fused_dim = 2 * gru_hidden_size + model_dim
        self.output_layer = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.BatchNorm1d(fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, output_seq_len)
        )

    def forward(self, x):
        x_projected = self.input_projection(x)
        x_pos = self.pos_encoder(x_projected.permute(1, 0, 2)).permute(1, 0, 2)
        transformer_output = self.transformer_encoder(x_pos)
        transformer_output = x_pos + transformer_output
        gru_output, _ = self.gru(transformer_output)
        gru_last_hidden = gru_output[:, -1, :]
        transformer_context = self.transformer_attention(transformer_output)
        fused_features = torch.cat([gru_last_hidden, transformer_context], dim=1)
        prediction = self.output_layer(fused_features)
        return prediction

# ==============================================================================
# 2. 预测函数
# ==============================================================================

def predict(model_path, raw_input_data):
    """
    使用已训练的模型进行预测。

    Args:
        model_path (str): 保存的模型和缩放器文件的路径。
        raw_input_data (pd.DataFrame): 原始的输入数据，必须包含168行和7个特征列，
                                       且列名和顺序必须与训练时完全一致。

    Returns:
        np.ndarray: 包含24个预测电价值的numpy数组。
    """
    # --- 检查输入数据格式 ---
    expected_columns = ['price', 'price_delta', 'load', 'load_delta', 'load_capacity_ratio', 'predicted_load', 'hour']
    if list(raw_input_data.columns) != expected_columns:
        raise ValueError(f"输入数据的列名或顺序不正确！期望列: {expected_columns}")
    if len(raw_input_data) != 168:
        raise ValueError(f"输入数据必须包含168个时间步（行），但收到了 {len(raw_input_data)} 行。")

    # --- 加载 artifacts ---
    print(f"Loading model and scalers from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = torch.load(model_path, map_location=device)
    
    # === 核心修改：补充缺失的配置 ===
    
    # 1. 定义一个包含所有必需参数的默认配置字典 (这些值应与train.py中的Config类匹配)
    default_config = {
        'N_FEATURES': 7,
        'INPUT_SEQ_LEN': 168,
        'OUTPUT_SEQ_LEN': 24,
        'MODEL_DIM': 128,       # 默认值
        'NUM_HEADS': 8,         # 默认值
        'NUM_ENCODER_LAYERS': 4,# 默认值
        'GRU_HIDDEN_SIZE': 128, # 默认值
        'GRU_LAYERS': 2,        # 默认值
        'DROPOUT_RATE': 0.25,   # 默认值
    }

    # 2. 加载文件中不完整的配置字典
    saved_config = artifacts.get('config_dict', {}) # 使用 .get() 避免旧文件没有此键而报错

    # 3. 将加载的配置更新到默认配置中
    #    这样，文件中有的参数会覆盖默认值，文件中没有的参数会使用默认值
    final_config = default_config.copy()
    final_config.update(saved_config)
    
    # === 修改结束 ===

    feature_scalers = artifacts['feature_scalers']
    target_scaler = artifacts['target_scaler']
    model_state_dict = artifacts['model_state_dict']

    # --- 重建模型 (现在使用我们合并后的 final_config) ---
    print("Rebuilding model architecture with complete config...")
    model = UltimateHybridModel(
        n_features=final_config['N_FEATURES'],
        model_dim=final_config['MODEL_DIM'],
        num_heads=final_config['NUM_HEADS'],
        num_encoder_layers=final_config['NUM_ENCODER_LAYERS'],
        gru_hidden_size=final_config['GRU_HIDDEN_SIZE'],
        gru_layers=final_config['GRU_LAYERS'],
        dropout=final_config['DROPOUT_RATE'],
        output_seq_len=final_config['OUTPUT_SEQ_LEN']
    )

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()


    # ... (后续代码完全保持不变) ...
    print("Preprocessing input data...")
    scaled_input_data = raw_input_data.copy()
    for col in scaled_input_data.columns:
        scaler = feature_scalers[col]
        scaled_input_data[col] = scaler.transform(scaled_input_data[[col]])
    
    input_tensor = torch.from_numpy(scaled_input_data.values).float()
    input_tensor = input_tensor.unsqueeze(0).to(device)

    print("Making prediction...")
    with torch.no_grad():
        scaled_prediction = model(input_tensor)

    print("Postprocessing prediction...")
    prediction_numpy = scaled_prediction.cpu().numpy()
    final_prediction = target_scaler.inverse_transform(prediction_numpy)
    
    return final_prediction.flatten()



# ==============================================================================
# 3. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # --- 指定模型路径 ---
    MODEL_PATH = "saved_model/model_and_scalers_ultimate.pth"

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件未找到于 '{MODEL_PATH}'")
        print("请先运行训练脚本生成模型文件。")
    else:
        # --- 创建一个模拟的输入数据 ---
        # 在实际应用中，您会从数据库或API获取这部分数据
        # 这里我们从训练数据中取一段来模拟真实场景
        print("Creating mock input data for demonstration...")
        # 假设我们有一个包含所有历史数据的文件
        try:
            full_data_df = pd.read_csv("mock_market_data_for_prediction.csv")
            # 取其中一段作为我们的输入 (例如，从第1000小时开始的168小时)
            start_index = 1000
            end_index = start_index + 168
            feature_cols = ['price', 'price_delta', 'load', 'load_delta', 'load_capacity_ratio', 'predicted_load', 'hour']
            new_input_df = full_data_df[feature_cols].iloc[start_index:end_index]
            print(f"Using data from index {start_index} to {end_index-1} as input.")

            # --- 调用预测函数 ---
            predicted_prices = predict(MODEL_PATH, new_input_df)

            # --- 显示结果 ---
            print("\n" + "="*50)
            print("      未来24小时电价预测结果      ")
            print("="*50)
            for i, price in enumerate(predicted_prices):
                print(f"  > 未来第 {i+1:2d} 小时预测电价: {price:8.4f}")
            print("="*50)
        
        except FileNotFoundError:
             print("\n错误: mock_market_data_for_prediction.csv 未找到。")
             print("无法创建模拟输入数据。请确保该文件与 predict.py 在同一目录，")
             print("或手动创建一个符合格式的 `new_input_df` DataFrame。")

