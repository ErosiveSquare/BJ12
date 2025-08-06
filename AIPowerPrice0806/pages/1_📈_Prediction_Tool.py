import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import math
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go

#预测界面这个是
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
    def __init__(self, n_features, model_dim, num_heads, num_encoder_layers, gru_hidden_size, gru_layers, dropout,
                 output_seq_len):
        super().__init__()
        self.input_projection = nn.Linear(n_features, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 4,
                                                   dropout=dropout, batch_first=True, activation='gelu')
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
# 2. Streamlit 缓存、预测和插值函数
# ==============================================================================

@st.cache_resource
def load_model_and_artifacts(model_path):
    """
    加载模型和相关文件，并使用 Streamlit 的缓存。
    """
    if not os.path.exists(model_path):
        st.error(f"错误: 模型文件未在 '{model_path}' 找到！请检查路径。")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        artifacts = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        st.error(f"加载模型文件失败: {e}")
        st.info("这可能是由于 PyTorch 版本不兼容或文件损坏。请确保您使用的是与训练模型时相近的 PyTorch 版本。")
        st.stop()

    return artifacts, device


def predict(artifacts, device, raw_input_data):
    """
    修改后的预测函数，接收加载好的 artifacts 和数据。
    """
    default_config = {
        'N_FEATURES': 7, 'INPUT_SEQ_LEN': 168, 'OUTPUT_SEQ_LEN': 24,
        'MODEL_DIM': 128, 'NUM_HEADS': 8, 'NUM_ENCODER_LAYERS': 4,
        'GRU_HIDDEN_SIZE': 128, 'GRU_LAYERS': 2, 'DROPOUT_RATE': 0.25,
    }
    saved_config = artifacts.get('config_dict', {})
    final_config = default_config.copy()
    final_config.update(saved_config)

    feature_scalers = artifacts['feature_scalers']
    target_scaler = artifacts['target_scaler']
    model_state_dict = artifacts['model_state_dict']

    model = UltimateHybridModel(
        n_features=final_config['N_FEATURES'], model_dim=final_config['MODEL_DIM'],
        num_heads=final_config['NUM_HEADS'], num_encoder_layers=final_config['NUM_ENCODER_LAYERS'],
        gru_hidden_size=final_config['GRU_HIDDEN_SIZE'], gru_layers=final_config['GRU_LAYERS'],
        dropout=final_config['DROPOUT_RATE'], output_seq_len=final_config['OUTPUT_SEQ_LEN']
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    scaled_input_data = raw_input_data.copy()
    for col in scaled_input_data.columns:
        if col in feature_scalers:
            scaler = feature_scalers[col]
            scaled_input_data[col] = scaler.transform(scaled_input_data[[col]])

    input_tensor = torch.from_numpy(scaled_input_data.values).float().unsqueeze(0).to(device)

    with torch.no_grad():
        scaled_prediction = model(input_tensor)

    prediction_numpy = scaled_prediction.cpu().numpy()
    final_prediction = target_scaler.inverse_transform(prediction_numpy)

    return final_prediction.flatten()


def interpolate_to_15min(hourly_prices):
    """
    使用三次样条插值将每小时数据转换为15分钟数据。
    """
    # 定义原始时间点 (0-23小时) 和新的时间点 (0, 0.25, ..., 23.75)
    hourly_time_points = np.arange(24)
    finer_time_points = np.arange(0, 24, 0.25)

    # 创建三次样条插值函数
    cs = CubicSpline(hourly_time_points, hourly_prices)

    # 计算15分钟粒度的价格
    finer_prices = cs(finer_time_points)

    # 创建时间标签
    time_labels = []
    for h in range(24):
        for m in [0, 15, 30, 45]:
            if h == 0 and m == 0:
                # 为了与每小时的+1h对齐，我们把第一个点标记为+1h
                time_labels.append(f"+1h 0m")
                continue
            time_labels.append(f"+{h + 1}h {m}m")

    # 因为arange(0, 24, 0.25)会产生96个点，所以需要96个标签
    # 第一个标签是 "+1h 0m",最后一个是 "+24h 45m"
    # 为了UI显示更直观，我们将第一个点标记为+1h，最后一个点标记为+25h
    # 修正时间标签生成逻辑
    time_labels = []
    for i, t in enumerate(finer_time_points):
        hour = int(t) + 1
        minute = int((t * 60) % 60)
        time_labels.append(f"+{hour}h {minute:02d}m")

    return finer_prices, time_labels, finer_time_points


# ==============================================================================
# 3. Streamlit 应用界面
# ==============================================================================
# --- 页面基础配置 ---
st.set_page_config(
    page_title="电价预测系统 | AI-Powered Price Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 主标题和介绍 ---
st.title("📈 智能电价预测系统")
st.markdown("""
基于 **Transformer & GRU** 混合深度学习模型的电价预测工具。
本系统能够对未来24小时的电价进行小时级精准预测，并提供15分钟粒度的平滑趋势分析。
""")
st.markdown("---")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 参数配置")
    MODEL_PATH = "saved_model/model_and_scalers_ultimate.pth"
    uploaded_file = st.file_uploader(
        "上传您的输入数据 (CSV or XLSX)",
        type=['csv', 'xlsx']
    )

    with st.expander("💡 如何准备输入数据?", expanded=False):
        st.markdown("""
        1.  **文件格式**: 支持 CSV 或 Excel (`.xlsx`) 文件。
        2.  **数据行数**: 文件必须 **不多不少，正好包含 168 行** 历史数据，代表过去7天每小时的数据点。
        3.  **数据列数与顺序**: 文件必须包含以下 **7个特征列**，且**名称和顺序**必须完全一致:
        """)
        st.code("price, price_delta, load, load_delta, load_capacity_ratio, predicted_load, hour")
        st.warning("任何格式不符都可能导致预测失败或结果不准确。")
    st.info("当前模型路径: `saved_model/model_and_scalers_ultimate.pth`")

# --- 加载模型 ---
artifacts_tuple = load_model_and_artifacts(MODEL_PATH)
if artifacts_tuple is None:
    st.stop()
artifacts, device = artifacts_tuple

# --- 主逻辑：处理上传的文件并进行预测 ---
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)

        st.subheader("📄 数据预览与验证")
        st.dataframe(input_df.head(), use_container_width=True)

        is_valid = True
        expected_columns = ['price', 'price_delta', 'load', 'load_delta', 'load_capacity_ratio', 'predicted_load',
                            'hour']
        if len(input_df) != 168:
            st.error(f"❌ 格式错误: 数据必须包含 168 行，但您上传的文件有 {len(input_df)} 行。")
            is_valid = False
        if list(input_df.columns) != expected_columns:
            st.error("❌ 格式错误: 列名或顺序不正确。请参照侧边栏的说明进行检查。")
            st.write("期望列:", expected_columns)
            st.write("您的列:", list(input_df.columns))
            is_valid = False

        if is_valid:
            st.success("✅ 数据格式正确，已准备好进行预测。")
            if st.button("🚀 开始预测未来24小时电价", type="primary", use_container_width=True):
                with st.spinner('🧠 正在分析数据并生成预测，请稍候...'):
                    predicted_prices = predict(artifacts, device, input_df)

                st.subheader("📊 预测结果分析")

                # 1. 关键指标展示
                avg_price = predicted_prices.mean()
                max_price = predicted_prices.max()
                min_price = predicted_prices.min()
                col1, col2, col3 = st.columns(3)
                col1.metric("未来24h平均电价", f"{avg_price:.2f}")
                col2.metric("未来24h最高电价", f"{max_price:.2f}", "峰值")
                col3.metric("未来24h最低电价", f"{min_price:.2f}", "谷值")

                # --- 创建选项卡 ---
                tab1, tab2 = st.tabs(["🕒 每小时预测详情", "📈 15分钟平滑趋势"])

                # --- Tab 1: 每小时预测 ---
                with tab1:
                    st.header("每小时电价预测")
                    # 创建小时级结果DataFrame
                    hourly_results_df = pd.DataFrame({
                        '预测未来小时': range(1, 25),
                        '预测电价': predicted_prices
                    })

                    # 使用Plotly绘制小时级图表
                    fig_hourly = go.Figure()
                    fig_hourly.add_trace(go.Scatter(
                        x=hourly_results_df['预测未来小时'],
                        y=hourly_results_df['预测电价'],
                        mode='lines+markers',
                        name='预测电价',
                        hovertemplate='未来第 %{x} 小时<br>预测电价: %{y:.2f}<extra></extra>'
                    ))
                    fig_hourly.update_layout(
                        title='未来24小时电价预测趋势',
                        xaxis_title='未来小时数 (h)',
                        yaxis_title='电价',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_hourly, use_container_width=True)

                    with st.expander("查看详细预测数据 (每小时)", expanded=False):
                        st.dataframe(
                            hourly_results_df.style.format({'预测电价': '{:.4f}'}),
                            use_container_width=True
                        )
                        csv_hourly = hourly_results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 下载每小时预测结果 (CSV)",
                            data=csv_hourly,
                            file_name='predicted_hourly_prices.csv',
                            mime='text/csv',
                        )

                # --- Tab 2: 15分钟平滑趋势 ---
                with tab2:
                    st.header("15分钟粒度平滑趋势")
                    st.info(
                        "ℹ️ 这是通过对每小时预测点进行**三次样条插值**生成的曲线")

                    # 生成15分钟粒度数据
                    finer_prices, time_labels, finer_time_points = interpolate_to_15min(predicted_prices)

                    # 创建15分钟粒度结果DataFrame
                    finer_results_df = pd.DataFrame({
                        '时间点': time_labels,
                        '平滑电价': finer_prices
                    })

                    # 使用Plotly绘制15分钟粒度图表
                    fig_finer = go.Figure()
                    # 绘制平滑曲线
                    fig_finer.add_trace(go.Scatter(
                        x=finer_time_points + 1,  # x轴从1开始
                        y=finer_prices,
                        mode='lines',
                        name='15分钟平滑曲线',
                        line=dict(shape='spline', color='rgba(255, 127, 14, 0.8)'),  # 使用橙色平滑线
                        hovertemplate='时间: %{customdata}<br>平滑电价: %{y:.2f}<extra></extra>',
                        customdata=time_labels
                    ))
                    # 标记原始的小时预测点
                    fig_finer.add_trace(go.Scatter(
                        x=hourly_results_df['预测未来小时'],
                        y=hourly_results_df['预测电价'],
                        mode='markers',
                        name='每小时预测点',
                        marker=dict(size=8, color='rgba(31, 119, 180, 1)'),
                        hovertemplate='未来第 %{x} 小时 (原始预测)<br>预测电价: %{y:.2f}<extra></extra>'
                    ))
                    fig_finer.update_layout(
                        title='未来24小时电价平滑趋势 (15分钟粒度)',
                        xaxis_title='未来小时数 (h)',
                        yaxis_title='电价',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_finer, use_container_width=True)

                    with st.expander("查看详细平滑数据 (15分钟)", expanded=False):
                        st.dataframe(
                            finer_results_df.style.format({'平滑电价': '{:.4f}'}),
                            use_container_width=True,
                            height=300  # 设置一个最大高度
                        )
                        csv_finer = finer_results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 下载15分钟平滑数据 (CSV)",
                            data=csv_finer,
                            file_name='predicted_15min_prices.csv',
                            mime='text/csv',
                        )

    except Exception as e:
        st.error(f"处理文件或预测时发生错误: {e}")
        st.warning("请确保您的文件是标准的 CSV 或 XLSX 格式，并且没有损坏。")

else:
    st.info("👋 请在左侧上传您的历史数据文件以启动预测。")
