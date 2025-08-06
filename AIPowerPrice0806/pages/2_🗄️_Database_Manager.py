import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import os

#数据库界面这个是
st.set_page_config(
    page_title="电力市场数据库管理",
    page_icon="🗄️",
    layout="wide"
)

st.title("🗄️ 电力市场数据库管理")
st.markdown(
    "本数据库借鉴美国PJM(Pennsylvania—New Jersey—Maryland)电力市场数据库的架构精髓，立足国内电力市场运行特性，针对与电价预测强关联的核心维度重构特征工程体系,为电价预测模型提供了更具解释力的特征基底与更贴近实际场景的训练数据支撑。")
st.markdown("---")
# --- 数据库连接与函数 ---
DB_FILE = 'market_data.db'
TABLE_NAME = 'electricity_data'


@st.cache_resource
def get_db_connection():
    """建立并缓存与SQLite数据库的连接。"""
    if not os.path.exists(DB_FILE):
        st.error(f"未找到数据库文件'{DB_FILE}'。请先运行设置脚本。")
        st.stop()
    return sqlite3.connect(DB_FILE, check_same_thread=False)


# 使用st.session_state管理事务性操作的连接
def get_session_conn():
    if 'db_conn' not in st.session_state or st.session_state.db_conn.is_connected() is False:
        st.session_state.db_conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    return st.session_state.db_conn


def execute_query(query, params=(), fetch=None):
    """执行SQL查询的辅助函数。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        if fetch == 'one':
            result = cursor.fetchone()
        elif fetch == 'all':
            result = cursor.fetchall()
        else:
            result = None
        conn.commit()
        return result
    except sqlite3.Error as e:
        st.error(f"数据库错误: {e}")
        return None
    finally:
        # 不关闭缓存的连接
        pass


@st.cache_data(ttl=30)  # 缓存数据30秒以便更快反映更改
def fetch_data(_conn):
    """从表中获取所有数据，将时间戳转换为datetime格式。"""
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", _conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp', ascending=False).reset_index(drop=True)


# 初始化行选择的会话状态
if 'selected_record_index' not in st.session_state:
    st.session_state.selected_record_index = None

# --- 主应用 ---
conn = get_db_connection()
df = fetch_data(conn)

# --- 不同操作的UI标签页 ---
tab1, tab2, tab3 = st.tabs(["📊 查看与选择数据", "➕ 添加新记录", "✏️ 编辑/删除所选记录"])

# --- 标签页1: 查看与选择数据 ---
with tab1:
    st.header("浏览和选择历史数据")
    st.info("点击下方表格中的一行，在后续标签页中对其进行编辑或删除。", icon="👆")

    with st.expander("🔎 应用筛选器", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_date, max_date = df['timestamp'].min().date(), df['timestamp'].max().date()
            date_range = st.date_input(
                "选择日期范围", value=(min_date, max_date),
                min_value=min_date, max_value=max_date, key="date_filter"
            )
        with col2:
            hours = sorted(df['hour'].unique())
            selected_hours = st.multiselect("按小时筛选", options=hours, default=[])

    # 应用筛选
    filtered_df = df.copy()
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]).replace(hour=23, minute=59)
        filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & (filtered_df['timestamp'] <= end_date)]
    if selected_hours:
        filtered_df = filtered_df[filtered_df['hour'].isin(selected_hours)]

    st.write(f"显示 **{len(filtered_df)}** 条记录，共 **{len(df)}** 条。")

    # 用于选择的交互式数据框
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=500,
        on_select="rerun",
        selection_mode="single-row"
    )

    # 将所选记录的索引存储在会话状态中
    if st.session_state.get("on_select", {}).get("rows"):
        st.session_state.selected_record_index = st.session_state.on_select["rows"][0]

    if st.session_state.selected_record_index is not None:
        st.success(
            f"已选择时间戳为 **{filtered_df.iloc[st.session_state.selected_record_index]['timestamp']}** 的行。请前往'编辑/删除'标签页。")

# --- 标签页2: 添加新记录 ---
with tab2:
    st.header("➕ 添加新数据记录")
    st.info("输入新的每小时数据点的详细信息。时间戳必须唯一。")

    with st.form("new_record_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            new_date = st.date_input("日期")
            new_price = st.number_input("价格", format="%.4f", step=0.0001)
            new_load = st.number_input("负荷", format="%.4f", step=0.0001)
        with c2:
            new_time = st.time_input("时间（小时）", step=3600)
            new_price_delta = st.number_input("价格变化量", format="%.4f", step=0.0001)
            new_load_delta = st.number_input("负荷变化量", format="%.4f", step=0.0001)
        with c3:
            new_capacity = st.number_input("容量", format="%.4f", step=0.0001)
            new_predicted_load = st.number_input("预测负荷", format="%.4f", step=0.0001)

        submitted = st.form_submit_button("💾 添加记录")

        if submitted:
            timestamp_str = f"{new_date} {new_time}"
            hour = new_time.hour
            load_capacity_ratio = (new_load / new_capacity) if new_capacity != 0 else 0

            existing = execute_query("SELECT 1 FROM electricity_data WHERE timestamp = ?", (timestamp_str,),
                                     fetch='one')
            if existing:
                st.error(f"错误：已存在时间戳为'{timestamp_str}'的记录。")
            else:
                query = """INSERT INTO electricity_data (timestamp, price, load, capacity, price_delta, load_delta, load_capacity_ratio, predicted_load, hour) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                params = (
                timestamp_str, new_price, new_load, new_capacity, new_price_delta, new_load_delta, load_capacity_ratio,
                new_predicted_load, hour)
                execute_query(query, params)
                st.success(f"✅ 已成功添加 {timestamp_str} 的记录！")
                st.cache_data.clear()  # 清除缓存以显示新数据

# --- 标签页3: 编辑/删除记录 ---
with tab3:
    st.header("✏️ 修改所选记录")

    if st.session_state.selected_record_index is None:
        st.warning("未选择记录。请前往'查看与选择数据'标签页，点击一行以选择它。",
                   icon="👈")
    else:
        # 从筛选后的数据框中检索所选行的完整数据
        selected_row = filtered_df.iloc[st.session_state.selected_record_index]
        original_timestamp = selected_row['timestamp']

        st.info(f"正在编辑时间戳为：**{original_timestamp.strftime('%Y-%m-%d %H:%M:%S')}** 的记录")

        with st.form("edit_form"):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                price = st.number_input("价格", value=float(selected_row['price']), format="%.4f")
                load = st.number_input("负荷", value=float(selected_row['load']), format="%.4f")
            with ec2:
                price_delta = st.number_input("价格变化量", value=float(selected_row['price_delta']), format="%.4f")
                load_delta = st.number_input("负荷变化量", value=float(selected_row['load_delta']), format="%.4f")
            with ec3:
                capacity = st.number_input("容量", value=float(selected_row['capacity']), format="%.4f")
                predicted_load = st.number_input("预测负荷", value=float(selected_row['predicted_load']),
                                                 format="%.4f")

            # 更新和删除按钮
            update_button = st.form_submit_button("🔄 更新记录")

            st.markdown("---")
            st.warning("⚠️ **注意：** 删除记录是永久性操作，无法撤销。", icon="🔥")
            if st.checkbox("我已了解并希望启用删除按钮。"):
                delete_button = st.form_submit_button("❌ 永久删除此记录")
            else:
                st.form_submit_button("❌ 永久删除此记录", disabled=True)

        if update_button:
            load_capacity_ratio = (load / capacity) if capacity != 0 else 0
            query = """UPDATE electricity_data SET price=?, load=?, capacity=?, price_delta=?, load_delta=?, load_capacity_ratio=?, predicted_load=? WHERE timestamp=?"""
            params = (price, load, capacity, price_delta, load_delta, load_capacity_ratio, predicted_load,
                      original_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            execute_query(query, params)
            st.success(f"✅ 已成功更新 {original_timestamp} 的记录！")
            st.cache_data.clear()
            st.session_state.selected_record_index = None  # 取消选择行
            st.rerun()

        if 'delete_button' in locals() and delete_button:
            query = "DELETE FROM electricity_data WHERE timestamp=?"
            params = (original_timestamp.strftime('%Y-%m-%d %H:%M:%S'),)
            execute_query(query, params)
            st.success(f"🔥 已永久删除 {original_timestamp} 的记录。")
            st.cache_data.clear()
            st.session_state.selected_record_index = None  # 取消选择行
            st.rerun()
