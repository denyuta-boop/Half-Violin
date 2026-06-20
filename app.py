import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import io

st.set_page_config(layout="wide", page_title="Scientific Plot Generator Pro")
st.title("📈 Scientific Plot Generator Pro")
st.markdown("**左右背中合わせのハーフバイオリン図**を作成します。")

# --- サイドバー ---
with st.sidebar:
    st.header("1. データ設定")
    use_demo = st.checkbox("サンプルデータを使う (Tips Dataset)", value=True)
   
    if use_demo:
        df = sns.load_dataset("tips")
        group_col = "time"
        val_col = "total_bill"
        st.success("✅ サンプルデータ(Tips)をロードしました")
    else:
        uploaded_file = st.file_uploader("CSVアップロード", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            group_col = st.selectbox("グループ列 (X軸)", df.columns)
            val_col = st.selectbox("値の列 (Y軸)", df.columns)
        else:
            df = None

    if df is not None:
        st.header("2. デザイン設定")
        color1 = st.color_picker("左グループの色", "#6FA9E5")
        color2 = st.color_picker("右グループの色", "#FA8F7C")
        plot_title = st.text_input("タイトル", "Distribution Comparison")

# --- メイン処理 ---
if df is not None:
    groups = df[group_col].unique()
    if len(groups) != 2:
        st.error("グループ列は**必ず2つのカテゴリ**を含んでください。")
    else:
        g1, g2 = sorted(groups)  # 安定させるためにsort
        d1 = df[df[group_col] == g1][val_col].dropna().values
        d2 = df[df[group_col] == g2][val_col].dropna().values
       
        # 統計検定
        _, p_val = mannwhitneyu(d1, d2, alternative='two-sided')
       
               # ==================== 描画 ====================
        fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
       
        # === ハーフバイオリン（正しい左右配置）===
        # 左側（g1）
        left_df = pd.DataFrame({g1: d1})
        sns.violinplot(data=left_df, x=np.full(len(d1), -0.3), y=g1,
                       color=color1, ax=ax, inner=None, cut=0, 
                       density_norm='count', width=0.6, linewidth=1.2)
        
        # 右側（g2）
        right_df = pd.DataFrame({g2: d2})
        sns.violinplot(data=right_df, x=np.full(len(d2), 0.3), y=g2,
                       color=color2, ax=ax, inner=None, cut=0, 
                       density_norm='count', width=0.6, linewidth=1.2)
        
        # === 散布点（左右にオフセット）===
        jitter = 0.07
        ax.scatter(np.random.normal(-0.3, jitter, len(d1)), d1,
                   color=color1, alpha=0.55, s=28, edgecolor='white', linewidth=0.4)
        ax.scatter(np.random.normal(0.3, jitter, len(d2)), d2,
                   color=color2, alpha=0.55, s=28, edgecolor='white', linewidth=0.4)
        
        # === ボックスプロット ===
        bp = ax.boxplot([d1, d2], positions=[-0.3, 0.3], widths=0.12,
                        showfliers=False, patch_artist=True, 
                        medianprops={'color': 'orange', 'linewidth': 2})
        
        for patch, color in zip(bp['boxes'], [color1, color2]):
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
            patch.set_alpha(0.9)
        
        # === 装飾 ===
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xticks([0])
        ax.set_xticklabels([f"{g1} vs {g2}\n(p = {p_val:.4f})"], 
                          fontsize=13, fontweight='bold')
        ax.set_xlim(-0.85, 0.85)
        ax.set_title(plot_title, fontsize=16, pad=20)
        ax.set_ylabel(val_col)
        ax.set_xlabel("")
        
        # 凡例追加（任意）
        # ax.legend([bp["boxes"][0], bp["boxes"][1]], [g1, g2], loc='upper right')
        
        st.pyplot(fig)
       
        # ダウンロードボタン
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button(
            label="📥 PNGダウンロード (Publication Quality)",
            data=buf,
            file_name="half_violin_plot.png",
            mime="image/png"
        )
else:
    st.info("👈 サイドバーからデータを選んでください")
