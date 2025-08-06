# 导入必要库
import streamlit as st
from PIL import Image  # 用于处理图片

#此为首页，执行：streamlit run Home.py

# 页面配置（保持不变）
st.set_page_config(
    page_title="智能市场电价预测分析 | 首页",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

logo = Image.open("NCEPU.png")

# 用columns创建一行两列布局，左侧放校徽，右侧放标题
col1, col2 = st.columns([1, 7])  # 比例可根据校徽大小调整（1:5表示左侧占1份，右侧占5份）
with col1:
    st.image(logo, width=80)  # width根据校徽实际尺寸调整（推荐60-100）
with col2:
    st.title("欢迎使用市场电价智能分析平台 ⚡")

st.markdown("---")

st.markdown("""
### 市场电价智能分析平台

本应用整合了自研深度学习电价预测工具与完善的数据库管理界面，作为题目的第一部分实现。

- 在侧边栏中导航至「预测工具」，使用我们先进的Transformer与GRU模型预测未来电价。

- 导航至「数据库管理」，查看、添加、编辑或删除为预测提供支持的历史市场数据记录。

**从侧边栏选择一个页面开始使用。**
""")

st.info("挑战杯“揭榜挂帅”", icon="⚡")
