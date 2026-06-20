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
        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
       
        # === 左右ハーフバイオリン（位置・幅を最適化）===
        offset = 0.35
        
        # 左側 (g1 - 青)
        sns.violinplot(data=pd.DataFrame({g1: d1}), 
                       x=np.full(len(d1), -offset), y=g1,
                       color=color1, ax=ax, inner=None, cut=0, 
                       density_norm='count', width=0.65, linewidth=1.1)
        
        # 右側 (g2 - 赤)
        sns.violinplot(data=pd.DataFrame({g2: d2}), 
                       x=np.full(len(d2), offset), y=g2,
                       color=color2, ax=ax, inner=None, cut=0, 
                       density_norm='count', width=0.65, linewidth=1.1)
        
        # === 散布点 ===
        jitter = 0.06
        ax.scatter(np.random.normal(-offset, jitter, len(d1)), d1,
                   color=color1, alpha=0.6, s=22, edgecolor='white', linewidth=0.3)
        ax.scatter(np.random.normal(offset, jitter, len(d2)), d2,
                   color=color2, alpha=0.6, s=22, edgecolor='white', linewidth=0.3)
        
        # === ボックスプロット ===
        bp = ax.boxplot([d1, d2], positions=[-offset, offset], widths=0.13,
                        showfliers=False, patch_artist=True,
                        medianprops={'color': 'darkorange', 'linewidth': 2.5})
        
        for patch, color in zip(bp['boxes'], [color1, color2]):
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
            patch.set_alpha(0.95)
        
        # === 装飾 ===
        ax.axvline(0, color="gray", linestyle="--", alpha=0.6, lw=1)
        
        ax.set_xticks([0])
        ax.set_xticklabels([f"{g1} vs {g2}\n(p = {p_val:.4f})"], 
                          fontsize=13, fontweight='bold')
        
        ax.set_xlim(-0.95, 0.95)   # 余白を十分に確保
        ax.set_title(plot_title, fontsize=16, pad=20)
        ax.set_ylabel(val_col)
        ax.set_xlabel("")
        
        # 枠線を綺麗に
        sns.despine(left=False, bottom=False, ax=ax)
        
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
