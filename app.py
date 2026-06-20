import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
import io

# ページ設定
st.set_page_config(layout="wide", page_title="Scientific Plot Generator")

st.title("📈 Scientific Plot Generator (Half-Violin)")
st.markdown("論文品質の分布比較グラフを数秒で作成します。データはブラウザ上でのみ処理され、サーバーには保存されません。")

# サイドバー設定
with st.sidebar:
    st.header("1. データ設定")
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        group_col = st.selectbox("グループ列 (X軸)", df.columns)
        val_col = st.selectbox("値の列 (Y軸)", df.columns)
        
        st.header("2. グラフ設定")
        plot_title = st.text_input("グラフタイトル", "Comparison of Results")
        y_label = st.text_input("Y軸ラベル", "Value")
        
        # 色設定
        color1 = st.color_picker("グループ1の色", "#6FA9E5")
        color2 = st.color_picker("グループ2の色", "#FA8F7C")

# メイン処理
if uploaded_file:
    # データ準備
    groups = df[group_col].unique()
    if len(groups) != 2:
        st.error("現在、グループが2つのデータのみ対応しています。")
    else:
        g1, g2 = groups[0], groups[1]
        data1 = df[df[group_col] == g1][val_col]
        data2 = df[df[group_col] == g2][val_col]
        
        # 統計検定 (両側検定)
        _, p_val = mannwhitneyu(data1, data2)
        
        # グラフ描画
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # ハーフバイオリン描画用のデータ構造
        plot_data = df.copy()
        
        # violinplot + stripplot + boxplot
        sns.violinplot(data=plot_data, x=group_col, y=val_col, palette=[color1, color2],
                       inner=None, cut=0, split=True, ax=ax, alpha=0.6)
        sns.stripplot(data=plot_data, x=group_col, y=val_col, color="gray", jitter=True, alpha=0.5, ax=ax)
        sns.boxplot(data=plot_data, x=group_col, y=val_col, width=0.2, boxprops={'facecolor':'none', 'edgecolor':'black'}, ax=ax)
        
        ax.set_title(plot_title, fontsize=16)
        ax.set_ylabel(y_label, fontsize=12)
        
        # 有意差ラベル
        if p_val < 0.05:
            ax.text(0.5, plot_data[val_col].max(), f"p={p_val:.4f}", ha='center', va='bottom', fontsize=12)

        st.pyplot(fig)
        
        # ダウンロードボタン
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button("グラフをダウンロード (PNG)", buf, "plot.png", "image/png")

else:
    st.info("左側のサイドバーからCSVをアップロードしてください。")
