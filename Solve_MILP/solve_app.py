import streamlit as st
import pandas as pd
import pyomo.environ as pyo
import numpy as np
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# 0. Streamlit 页面配置 (放在最前面)
# =============================================================================
st.set_page_config(
    page_title="储能电站辅助优化决策系统",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 1. 核心优化器类 (保持不变)
# =============================================================================
class BessOptimizer:
    """
    储能电站辅助决策优化器-华北电力大学
    """

    def __init__(self, station_params, price_forecast_df):
        self.params = station_params
        price_series = price_forecast_df['平滑电价']
        self.prices = {i + 1: price_series.iloc[i] for i in range(len(price_series))}
        self.model = None
        self.results = None
        self.results_df = None
        self.summary = {}

    def _build_model(self):
        # --- 详细模型构建，与之前相同 ---
        self.model = pyo.ConcreteModel(name="BESS_Optimal_Scheduling_V2")
        T = self.params['T']
        self.model.T = pyo.RangeSet(1, T)
        self.model.lambda_forecast = pyo.Param(self.model.T, initialize=self.prices)
        self.model.k = pyo.Param(initialize=self.params['k'])
        self.model.C_op = pyo.Param(initialize=self.params['C_op'])
        self.model.delta_t = pyo.Param(initialize=self.params['delta_t'])
        self.model.E_rated = pyo.Param(initialize=self.params['E_rated'])
        self.model.E0 = pyo.Param(initialize=self.params['E0'])
        self.model.E_T_target = pyo.Param(initialize=self.params['E_T_target'])
        self.model.P_ch_max = pyo.Param(initialize=self.params['P_ch_max'])
        self.model.P_dis_max = pyo.Param(initialize=self.params['P_dis_max'])
        self.model.eta_ch = pyo.Param(initialize=self.params['eta_ch'])
        self.model.eta_dis = pyo.Param(initialize=self.params['eta_dis'])
        self.model.SOC_min = pyo.Param(initialize=self.params['SOC_min'])
        self.model.SOC_max = pyo.Param(initialize=self.params['SOC_max'])
        self.model.N_cycle_max = pyo.Param(initialize=self.params['N_cycle_max'])
        self.model.E_min = self.model.E_rated * self.model.SOC_min
        self.model.E_max = self.model.E_rated * self.model.SOC_max
        self.model.P_ch = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, self.model.P_ch_max))
        self.model.P_dis = pyo.Var(self.model.T, domain=pyo.NonNegativeReals, bounds=(0, self.model.P_dis_max))
        self.model.E = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
        self.model.alpha = pyo.Var(self.model.T, domain=pyo.Binary)
        self.model.beta = pyo.Var(self.model.T, domain=pyo.Binary)

        def objective_rule(model):
            market_revenue = sum(
                model.lambda_forecast[t] * (model.P_dis[t] - model.P_ch[t]) * model.delta_t for t in model.T)
            degradation_cost = model.k * sum((model.P_ch[t] + model.P_dis[t]) * model.delta_t for t in model.T)
            return market_revenue - degradation_cost - model.C_op

        self.model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        def energy_balance_rule(model, t):
            if t == 1:
                return model.E[t] == model.E0 + (
                            model.P_ch[t] * model.eta_ch - model.P_dis[t] / model.eta_dis) * model.delta_t
            return model.E[t] == model.E[t - 1] + (
                        model.P_ch[t] * model.eta_ch - model.P_dis[t] / model.eta_dis) * model.delta_t

        self.model.energy_balance_con = pyo.Constraint(self.model.T, rule=energy_balance_rule)

        def charge_power_rule(model, t):
            return model.P_ch[t] <= model.beta[t] * model.P_ch_max

        self.model.charge_power_con = pyo.Constraint(self.model.T, rule=charge_power_rule)

        def discharge_power_rule(model, t):
            return model.P_dis[t] <= model.alpha[t] * model.P_dis_max

        self.model.discharge_power_con = pyo.Constraint(self.model.T, rule=discharge_power_rule)

        def mutex_rule(model, t):
            return model.alpha[t] + model.beta[t] <= 1

        self.model.mutex_con = pyo.Constraint(self.model.T, rule=mutex_rule)

        def soc_limit_rule(model, t):
            return pyo.inequality(self.model.E_min, model.E[t], self.model.E_max)

        self.model.soc_limit_con = pyo.Constraint(self.model.T, rule=soc_limit_rule)

        def terminal_energy_rule(model):
            return model.E[self.params['T']] >= self.model.E_T_target

        self.model.terminal_energy_con = pyo.Constraint(rule=terminal_energy_rule)

        def cycle_limit_rule(model):
            total_throughput = sum((model.P_ch[t] + model.P_dis[t]) * model.delta_t for t in model.T)
            return total_throughput <= 2 * model.E_rated * model.N_cycle_max

        self.model.cycle_limit_con = pyo.Constraint(rule=cycle_limit_rule)

        def hourly_consistency_charge_rule(model, h, i):
            t = 4 * h + i
            if t + 1 <= T:
                return model.beta[t] == model.beta[t + 1]
            return pyo.Constraint.Skip

        self.model.hourly_consistency_charge_con = pyo.Constraint(pyo.RangeSet(0, 23), pyo.RangeSet(1, 3),
                                                                  rule=hourly_consistency_charge_rule)

        def hourly_consistency_discharge_rule(model, h, i):
            t = 4 * h + i
            if t + 1 <= T:
                return model.alpha[t] == model.alpha[t + 1]
            return pyo.Constraint.Skip

        self.model.hourly_consistency_discharge_con = pyo.Constraint(pyo.RangeSet(0, 23), pyo.RangeSet(1, 3),
                                                                     rule=hourly_consistency_discharge_rule)

    def solve(self, solver_name='cbc'):
        if self.model is None:
            self._build_model()
        try:
            solver = pyo.SolverFactory(solver_name)
            if not solver.available():
                st.error(f"求解器 '{solver_name}' 不可用。请确认已正确安装并配置。")
                return False, f"求解器 '{solver_name}' 不可用"
        except Exception as e:
            st.error(f"加载求解器 '{solver_name}' 时出错: {e}")
            st.error(
                "请确保求解器已安装并且在系统路径中。例如，对于CBC，可尝试 `conda install -c conda-forge coincbc` 或 `sudo apt-get install coinor-cbc`。")
            return False, f"加载求解器失败"
        self.results = solver.solve(self.model, tee=False)
        if (self.results.solver.status == pyo.SolverStatus.ok) and (
                self.results.solver.termination_condition == pyo.TerminationCondition.optimal):
            self._process_results()
            return True, "最优"
        else:
            return False, str(self.results.solver.termination_condition)

    def _process_results(self):
        if self.results.solver.termination_condition != pyo.TerminationCondition.optimal: return
        data = []
        for t in self.model.T:
            P_ch_val, P_dis_val, E_val = pyo.value(self.model.P_ch[t]), pyo.value(self.model.P_dis[t]), pyo.value(
                self.model.E[t])
            SOC_val = (E_val / self.params['E_rated']) * 100
            price_val = self.model.lambda_forecast[t]
            revenue = (price_val * (P_dis_val - P_ch_val)) * self.params['delta_t']
            data.append({'Time_Step': t, 'Price (元/MWh)': price_val, 'Charge_Power (MW)': P_ch_val,
                         'Discharge_Power (MW)': P_dis_val, 'Net_Power (MW)': P_dis_val - P_ch_val,
                         'Energy (MWh)': E_val, 'SOC (%)': SOC_val, 'Interval_Revenue (元)': revenue})
        self.results_df = pd.DataFrame(data)
        gross_revenue = self.results_df['Interval_Revenue (元)'].sum()
        total_throughput = pyo.value(
            sum((self.model.P_ch[t] + self.model.P_dis[t]) * self.model.delta_t for t in self.model.T))
        degradation_cost = self.params['k'] * total_throughput
        net_profit = pyo.value(self.model.objective)
        equivalent_cycles = total_throughput / (2 * self.params['E_rated']) if self.params['E_rated'] > 0 else 0
        self.summary = {"总市场收益 (元)": gross_revenue, "退化成本 (元)": degradation_cost,
                        "固定运维成本 (元)": self.params['C_op'], "总净利润 (元)": net_profit,
                        "总能量吞吐 (MWh)": total_throughput, "等效循环次数": equivalent_cycles}

    def generate_bidding_strategy(self, delta_ch, delta_dis):
        if self.results_df is None: return pd.DataFrame()
        bidding_df = self.results_df[
            ['Time_Step', 'Price (元/MWh)', 'Charge_Power (MW)', 'Discharge_Power (MW)']].copy()

        def calculate_bid_price(row):
            if row['Charge_Power (MW)'] > 1e-3:
                return row['Price (元/MWh)'] * (1 + delta_ch)
            elif row['Discharge_Power (MW)'] > 1e-3:
                return row['Price (元/MWh)'] * (1 - delta_dis)
            else:
                return np.nan

        bidding_df['Bid_Price (元/MWh)'] = bidding_df.apply(calculate_bid_price, axis=1)
        bidding_df['Bid_Type'] = np.where(bidding_df['Charge_Power (MW)'] > 1e-3, '充电报价',
                                          np.where(bidding_df['Discharge_Power (MW)'] > 1e-3, '放电报价', '不参与'))
        return bidding_df


# =============================================================================
# 2. Streamlit UI 辅助函数
# =============================================================================
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')


def plot_results_plotly(results_df, bess_params):
    """
    使用 Plotly 生成高级、交互式的可视化图表，并返回figure对象
    """
    if results_df.empty:
        return None

    # 创建带有双Y轴的图表
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 准备时间轴
    time_index = pd.to_datetime(pd.date_range(start='2024-01-01', periods=bess_params['T'], freq='15min'))

    # 1. 充放电功率 Bar 图 (主Y轴)
    # 使用负值表示充电，以在0轴下方显示
    fig.add_trace(
        go.Bar(
            x=time_index,
            y=results_df['Discharge_Power (MW)'],
            name='放电功率 (MW)',
            marker_color='#2ca02c',  # 专业绿色
            hovertemplate='<b>%{x|%H:%M}</b><br>放电: %{y:.2f} MW<extra></extra>'
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=time_index,
            y=-results_df['Charge_Power (MW)'],
            name='充电功率 (MW)',
            marker_color='#d62728',  # 专业红色
            hovertemplate='<b>%{x|%H:%M}</b><br>充电: %{customdata:.2f} MW<extra></extra>',
            customdata=results_df['Charge_Power (MW)']  # 在悬停信息中显示正值
        ),
        secondary_y=False,
    )

    # 2. SOC 折线图 (副Y轴)
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=results_df['SOC (%)'],
            name='SOC (%)',
            mode='lines',
            line=dict(color='#ff7f0e', width=3),  # 醒目的橙色
            hovertemplate='<b>%{x|%H:%M}</b><br>SOC: %{y:.2f}%<extra></extra>'
        ),
        secondary_y=True,
    )

    # 3. 电价 折线图 (副Y轴)
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=results_df['Price (元/MWh)'],
            name='预测电价 (元/MWh)',
            mode='lines+markers',
            line=dict(color='#1f77b4', dash='dash'),  # 专业的蓝色虚线
            marker=dict(size=4),
            hovertemplate='<b>%{x|%H:%M}</b><br>电价: %{y:.2f} 元/MWh<extra></extra>'
        ),
        secondary_y=True,
    )

    # 4. SOC 上下限水平线
    soc_min_pct = bess_params['SOC_min'] * 100
    soc_max_pct = bess_params['SOC_max'] * 100
    fig.add_hline(y=soc_min_pct, line_dash="dot", line_color='#ff7f0e',
                  annotation_text=f"SOC下限: {soc_min_pct}%",
                  annotation_position="bottom right", secondary_y=True)
    fig.add_hline(y=soc_max_pct, line_dash="dot", line_color='#ff7f0e',
                  annotation_text=f"SOC上限: {soc_max_pct}%",
                  annotation_position="top right", secondary_y=True)

    # 5. 更新图表布局和样式，使其更“高级”
    fig.update_layout(
        title=dict(
            text='<b>储能电站日前市场优化调度策略</b>',
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        xaxis=dict(
            title='调度时间',
            tickformat='%H:%M',
            dtick=3600000 * 2,  # 每2小时一个主刻度
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            title='<b>充/放电功率 (MW)</b>',
            titlefont=dict(color='#d62728'),
            tickfont=dict(color='#d62728'),
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
        ),
        yaxis2=dict(
            title='<b>电价 (元/MWh) / SOC (%)</b>',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
            overlaying='y',
            side='right',
            showgrid=False,  # 副Y轴网格通常可以省略，使图表更简洁
            range=[0, max(results_df['Price (元/MWh)'].max(), 100) * 1.1]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='relative',  # 使正负条形图从0轴向两边延伸
        template='plotly_white',  # 使用简洁的白色背景模板
        margin=dict(l=80, r=80, t=100, b=80),  # 增加边距，防止标签被截断
        hovermode='x unified'  # 统一X轴的悬停信息，非常高级！
    )

    return fig


# =============================================================================
# 3. Streamlit App 主体 (与之前版本一致, 只修改了绘图函数的调用)
# =============================================================================
# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    st.subheader("1. 电价数据")
    price_source = st.radio("选择电价数据来源", ('上传CSV文件', '使用内置示范数据'), horizontal=True,
                            key="price_source")
    uploaded_file = None
    if price_source == '上传CSV文件':
        uploaded_file = st.file_uploader("上传15分钟电价预测文件 (CSV)", type=['csv'],
                                         help="CSV文件需包含名为'平滑电价'的列，共96个数据点。")
    st.subheader("2. 电站核心参数")
    col1, col2 = st.columns(2)
    with col1:
        E_rated = st.number_input("额定容量 (MWh)", min_value=1.0, value=100.0, step=10.0)
        P_ch_max = st.number_input("最大充电功率 (MW)", min_value=1.0, value=50.0, step=5.0)
    with col2:
        E0 = st.number_input("初始能量 (MWh)", min_value=0.0, value=50.0, step=10.0, max_value=E_rated)
        P_dis_max = st.number_input("最大放电功率 (MW)", min_value=1.0, value=50.0, step=5.0)
    st.subheader("3. 运行与经济参数")
    SOC_min, SOC_max = st.slider("可用SOC范围 (%)", 0, 100, (10, 90), 5)
    col3, col4 = st.columns(2)
    with col3:
        eta_ch = st.slider("充电效率", 0.80, 1.00, 0.95, 0.01, key="eta_ch")
        k = st.number_input("度电退化成本 (元/MWh)", min_value=0.0, value=5.0, step=1.0)
    with col4:
        eta_dis = st.slider("放电效率", 0.80, 1.00, 0.95, 0.01, key="eta_dis")
        N_cycle_max = st.number_input("日最大等效循环", min_value=0.1, value=1.0, step=0.1)
    st.subheader("4. 报价策略参数")
    delta_ch = st.slider("充电报价上浮系数 (%)", 0, 20, 5, key="delta_ch") / 100.0
    delta_dis = st.slider("放电报价下浮系数 (%)", 0, 20, 5, key="delta_dis") / 100.0
    st.markdown("---")
    run_button = st.button("🚀 开始优化计算", type="primary", use_container_width=True)

# --- 主界面 ---
st.title("🔋 储能电站辅助优化决策系统")
st.markdown("欢迎使用本系统。请在左侧边栏配置您的电站参数和电价数据，然后点击“开始优化计算”按钮。")

if run_button:
    bess_params = {'T': 96, 'delta_t': 0.25, 'k': k, 'C_op': 0.0, 'E_rated': E_rated, 'E0': E0, 'E_T_target': E0,
                   'P_ch_max': P_ch_max, 'P_dis_max': P_dis_max, 'eta_ch': eta_ch, 'eta_dis': eta_dis,
                   'SOC_min': SOC_min / 100.0, 'SOC_max': SOC_max / 100.0, 'N_cycle_max': N_cycle_max}
    price_data = None
    if price_source == '上传CSV文件':
        if uploaded_file is not None:
            try:
                price_data = pd.read_csv(uploaded_file)
                if '平滑电价' not in price_data.columns: st.error(
                    "上传的CSV文件中未找到 '平滑电价' 列，请检查文件。"); st.stop()
                if len(price_data) != 96: st.warning(
                    f"电价数据应有96个点，但上传文件有 {len(price_data)} 个点。将使用前96个点。"); price_data = price_data.head(
                    96)
            except Exception as e:
                st.error(f"读取CSV文件失败: {e}"); st.stop()
        else:
            st.error("请上传一个CSV文件。"); st.stop()
    else:
        time_points = np.arange(bess_params['T']);
        base_price, peak_price, valley_price = 400, 900, 150
        prices = base_price - 150 * np.sin(2 * np.pi * time_points / 48)
        prices[8:25] = np.linspace(prices[8], valley_price, 17)
        prices[40:57] = np.linspace(prices[40], peak_price, 17)
        prices[72:85] = np.linspace(prices[72], peak_price, 13)
        price_data = pd.DataFrame({'平滑电价': prices})
        st.info("正在使用内置的示范电价数据。")

    with st.spinner('⏳ 正在进行优化计算，这可能需要一些时间...'):
        try:
            optimizer = BessOptimizer(bess_params, price_data)
            is_solved, status = optimizer.solve(solver_name='cbc')
        except Exception as e:
            st.error(f"优化过程中发生未知错误: {e}"); st.stop()

    if is_solved:
        st.success(f"🎉 优化成功！已找到最优解。求解器状态: {status}")
        results_df, summary = optimizer.results_df, optimizer.summary
        bidding_df = optimizer.generate_bidding_strategy(delta_ch, delta_dis)

        # *** ✨【核心改动】调用新的Plotly绘图函数 ✨ ***
        fig = plot_results_plotly(results_df, bess_params)

        tab1, tab2, tab3 = st.tabs(["📈 **核心看板**", "📊 **详细调度数据**", "💰 **市场报价策略**"])
        with tab1:
            st.header("关键性能指标 (KPIs)")
            kpi_cols = st.columns(4)
            kpi_cols[0].metric(label="💰 总净利润", value=f"{summary['总净利润 (元)']:,.2f} 元",
                               delta=f"{summary['总市场收益 (元)']:,.2f} 元收益")
            kpi_cols[1].metric(label="🔄 等效循环次数", value=f"{summary['等效循环次数']:.3f} 次",
                               help=f"日最大循环限制: {bess_params['N_cycle_max']}")
            kpi_cols[2].metric(label="⚡ 总能量吞吐", value=f"{summary['总能量吞吐 (MWh)']:.2f} MWh",
                               help=f"退化成本: {summary['退化成本 (元)']:,.2f} 元")
            kpi_cols[3].metric(label="💡 平均度电利润",
                               value=f"{summary['总净利润 (元)'] / summary['总能量吞吐 (MWh)'] if summary['总能量吞吐 (MWh)'] > 0 else 0:.2f} 元/MWh")
            st.header("优化调度策略可视化")
            if fig:
                # *** ✨【核心改动】使用st.plotly_chart渲染图表 ✨ ***
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("无法生成图表。")
        with tab2:
            st.header("详细调度计划")
            st.markdown("下表展示了每个15分钟时间段的详细调度结果。")
            st.dataframe(results_df.style.format(
                {'Price (元/MWh)': '{:.2f}', 'Charge_Power (MW)': '{:.2f}', 'Discharge_Power (MW)': '{:.2f}',
                 'Net_Power (MW)': '{:.2f}', 'Energy (MWh)': '{:.2f}', 'SOC (%)': '{:.1f}%',
                 'Interval_Revenue (元)': '{:,.2f}'}).background_gradient(cmap='viridis', subset=['Price (元/MWh)']),
                         use_container_width=True)
            csv_results = convert_df_to_csv(results_df)
            st.download_button(label="📥 下载详细数据 (CSV)", data=csv_results, file_name='bess_optimal_schedule.csv',
                               mime='text/csv')
        with tab3:
            st.header("生成的市场报价策略")
            st.markdown(
                f"基于优化结果，并考虑充电 **{delta_ch * 100:.1f}%** 的上浮和放电 **{delta_dis * 100:.1f}%** 的下浮，生成以下报价策略。")
            active_bids = bidding_df[bidding_df['Bid_Price (元/MWh)'].notna()].copy()
            active_bids['Quantity (MW)'] = active_bids.apply(
                lambda row: row['Charge_Power (MW)'] if row['Bid_Type'] == '充电报价' else row['Discharge_Power (MW)'],
                axis=1)
            st.dataframe(active_bids[['Time_Step', 'Bid_Type', 'Quantity (MW)', 'Bid_Price (元/MWh)']].style.format(
                {'Quantity (MW)': '{:.2f}', 'Bid_Price (元/MWh)': '{:.2f}'}).apply(lambda s: [
                'background-color: #d4edda' if v == '放电报价' else 'background-color: #f8d7da' if v == '充电报价' else ''
                for v in s], subset=['Bid_Type']), use_container_width=True)
            csv_bids = convert_df_to_csv(active_bids)
            st.download_button(label="📥 下载报价策略 (CSV)", data=csv_bids, file_name='bess_bidding_strategy.csv',
                               mime='text/csv')
    else:
        st.error(f"❌ 优化求解失败。")
        st.error(f"求解器终止条件: **{status}**")
        st.warning("请尝试以下操作：")
        st.markdown(
            "- 检查电站参数是否合理（例如，初始能量是否在SOC限制范围内）。\n- 调整日最大循环次数等约束条件，可能当前条件过于严格。\n- 确认您安装的CBC求解器版本与您的操作系统和Python环境兼容。")
else:
    st.info("👈 请在左侧配置参数后，点击按钮开始。")
    st.markdown(
        "--- \n### 系统功能简介\n本系统旨在帮助储能电站运营商制定最优的日前市场充放电策略...\n**技术栈**: `Python`, `Streamlit`, `Pyomo`, `CBC Solver`, `Pandas`, `Plotly`")
