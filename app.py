import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon
import io

st.set_page_config(layout="wide", page_title="Scientific Plot Generator Pro")

st.title("📈 Scientific Plot Generator Pro")
st.markdown("分布比較のための、左右背中合わせハーフバイオリン図を作成します。")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    uploaded_file = st.file_uploader("CSVアップロード", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        group_col = st.selectbox("グループ列 (X軸)", df.columns)
        val_col = st.selectbox("値の列 (Y軸)", df.columns)
        
        st.header("グラフデザイン")
        color1 = st.color_picker("左グループの色", "#6FA9E5")
        color2 = st.color_picker("右グループの色", "#FA8F7C")
        plot_title = st.text_input("タイトル", "Distribution Comparison")

# メイン処理
if uploaded_file:
    groups = df[group_col].unique()
    if len(groups) != 2:
        st.error("グループ列は必ず2つのカテゴリを含んでください。")
    else:
        g1, g2 = groups[0], groups[1]
        d1 = df[df[group_col] == g1][val_col].values
        d2 = df[df[group_col] == g2][val_col].values
        
        # 統計検定
        stat, p_val = mannwhitneyu(d1, d2)
        
        # 描画開始
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # 1. 左側(g1)のハーフバイオリン
        v1 = ax.violinplot(d1, positions=[0], vert=True, widths=0.8, showextrema=False)
        for pc in v1['bodies']:
            pc.set_facecolor(color1)
            pc.set_alpha(0.7)
            # 頂点を反転させて左側に寄せる
            m = np.mean(pc.get_paths()[0].vertices[:, 0])
            pc.get_paths()[0].vertices[:, 0] = 2 * m - pc.get_paths()[0].vertices[:, 0]
            
        # 2. 右側(g2)のハーフバイオリン
        v2 = ax.violinplot(d2, positions=[0], vert=True, widths=0.8, showextrema=False)
        for pc in v2['bodies']:
            pc.set_facecolor(color2)
            pc.set_alpha(0.7)

        # 3. ストリップとボックス (左右に微調整)
        sns.stripplot(x=np.zeros(len(d1)) - 0.1, y=d1, color=color1, alpha=0.5, jitter=0.05, ax=ax)
        sns.stripplot(x=np.zeros(len(d2)) + 0.1, y=d2, color=color2, alpha=0.5, jitter=0.05, ax=ax)
        
        ax.boxplot([d1, d2], positions=[-0.2, 0.2], widths=0.1, showfliers=False, 
                   patch_artist=True, boxprops={'facecolor':'white', 'edgecolor':'black'})

        # 装飾
        ax.set_xticks([0])
        ax.set_xticklabels([f"{g1} vs {g2} (p={p_val:.4f})"])
        ax.set_title(plot_title)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        
        st.pyplot(fig)
        
        # ダウンロード
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("PNGダウンロード", buf, "publication_plot.png", "image/png")
