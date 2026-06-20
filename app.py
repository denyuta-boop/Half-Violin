import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon
import io
import os
import urllib.request

st.set_page_config(layout="wide", page_title="Scientific Plot Generator Pro")
st.title("📈 Scientific Plot Generator Pro")
st.markdown("カテゴリごとに半分のバイオリン（**raincloud風**）＋散布図＋箱ひげ図を重ねたグラフを作成します。")


# ---------- フォント ----------
@st.cache_resource
def load_times_new_roman():
    font_path = "/tmp/TimesNewRoman.ttf"
    font_url = "https://github.com/google/fonts/raw/main/apache/timesnewroman/Times%20New%20Roman.ttf"
    try:
        if not os.path.exists(font_path):
            urllib.request.urlretrieve(font_url, font_path)
        fm.fontManager.addfont(font_path)
        return fm.FontProperties(fname=font_path).get_name()
    except Exception:
        return None


with st.sidebar:
    st.header("0. フォント")
    use_times = st.checkbox("Times New Romanを使う（ダウンロードが必要）", value=False)
    font_family = "serif"
    if use_times:
        loaded = load_times_new_roman()
        if loaded:
            font_family = loaded
        else:
            st.warning("フォントのダウンロードに失敗しました。デフォルトフォントを使用します。")
    plt.rcParams["font.family"] = font_family
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False

    # ===================== 1. データ形式 / データ =====================
    st.header("1. データ")
    data_mode = st.radio(
        "CSVのデータ形式",
        ["ワイド形式（基準値1列＋手法2列から誤差を計算）", "ロング形式（値1列＋グループ列）"],
    )

    df = None
    plot_df = None
    g1 = g2 = None
    d1 = d2 = None
    val_label_default = "Value"

    if data_mode.startswith("ワイド"):
        st.caption("例：meanSSDE（基準値）, SSDE_RAI, SSDE_center のような列構成のCSV")
        use_demo_wide = st.checkbox("サンプル誤差データを使う", value=True, key="demo_wide")
        if use_demo_wide:
            rng = np.random.default_rng(0)
            n = 200
            ref = rng.normal(30, 5, n)
            method1 = ref - rng.normal(0, 1, n)
            method2 = ref - rng.normal(-0.3, 1.3, n)
            df = pd.DataFrame({"meanSSDE": ref, "SSDE_RAI": method1, "SSDE_center": method2})
        else:
            up = st.file_uploader("CSVアップロード", type="csv", key="wide_csv")
            if up:
                df = pd.read_csv(up)

        if df is not None:
            ref_col = st.selectbox("基準値列", df.columns)
            other_cols = [c for c in df.columns if c != ref_col]
            m1_col = st.selectbox("手法1列（左側）", other_cols, index=0)
            m2_col = st.selectbox(
                "手法2列（右側）", [c for c in other_cols if c != m1_col], index=0
            )

            sub = df[[ref_col, m1_col, m2_col]].dropna()
            err1 = (sub[ref_col] - sub[m1_col]).values
            err2 = (sub[ref_col] - sub[m2_col]).values

            cc1, cc2 = st.columns(2)
            label1 = cc1.text_input("左ラベル", m1_col)
            label2 = cc2.text_input("右ラベル", m2_col)

            g1, g2 = label1, label2
            d1, d2 = err1, err2
            plot_df = pd.DataFrame(
                {"Group": [g1] * len(d1) + [g2] * len(d2), "Value": np.concatenate([d1, d2])}
            )
            val_label_default = f"{ref_col} - Method"

    else:
        use_demo_long = st.checkbox("サンプルデータを使う (Tips Dataset)", value=True, key="demo_long")
        if use_demo_long:
            df = sns.load_dataset("tips")
            group_col, val_col = "time", "total_bill"
        else:
            up = st.file_uploader("CSVアップロード", type="csv", key="long_csv")
            if up:
                df = pd.read_csv(up)
                group_col = st.selectbox("グループ列", df.columns)
                val_col = st.selectbox("値の列", [c for c in df.columns if c != group_col])
            else:
                df = None

        if df is not None:
            groups = df[group_col].dropna().unique()
            if len(groups) != 2:
                st.error("グループ列は必ず2カテゴリにしてください。")
                df = None
            else:
                g1_raw, g2_raw = sorted(groups)
                d1 = df[df[group_col] == g1_raw][val_col].dropna().values
                d2 = df[df[group_col] == g2_raw][val_col].dropna().values

                cc1, cc2 = st.columns(2)
                label1 = cc1.text_input("左ラベル", str(g1_raw))
                label2 = cc2.text_input("右ラベル", str(g2_raw))

                g1, g2 = label1, label2
                plot_df = pd.DataFrame(
                    {"Group": [g1] * len(d1) + [g2] * len(d2), "Value": np.concatenate([d1, d2])}
                )
                val_label_default = val_col

    # ===================== 2〜5. オプション類（データがある時だけ表示） =====================
    if plot_df is not None:
        st.header("2. 統計検定")
        test_options = ["Mann-Whitney U検定（対応なし）", "Wilcoxon符号順位検定（対応あり）"]
        default_index = 1 if data_mode.startswith("ワイド") else 0
        test_choice = st.radio("検定方法", test_options, index=default_index)
        if test_choice.startswith("Wilcoxon") and len(d1) != len(d2):
            st.caption(f"⚠️ 左右でサンプル数が異なります（{len(d1)} 件 / {len(d2)} 件）。"
                       "Wilcoxonは対応のあるデータ用なので、選択した場合は自動的にMann-Whitney Uにフォールバックします。")

        st.header("3. デザイン設定")
        color1 = st.color_picker("左グループの色", "#99BBE3")
        color2 = st.color_picker("右グループの色", "#FAB3A6")
        plot_title = st.text_input("タイトル", "Distribution Comparison")
        y_label = st.text_input("Y軸ラベル", val_label_default)
        use_mathtext = st.checkbox(
            "X軸ラベルを下付き文字風にする（例: SSDE_RAI → SSDE_RAI の RAI が下付き）", value=False
        )
        show_zero_line = st.checkbox("y = 0 の破線を表示", value=data_mode.startswith("ワイド"))
        show_sig = st.checkbox("有意差ブラケットを表示", value=True)

        st.header("4. グラフ調整")
        fig_width = st.slider("画像横幅 (inch)", 6.0, 14.0, 9.0, 0.5)
        fig_height = st.slider("画像縦幅 (inch)", 5.0, 12.0, 7.5, 0.5)
        violin_width = st.slider("バイオリン幅", 0.5, 1.2, 0.9, 0.05)
        box_width = st.slider("ボックス幅", 0.1, 0.4, 0.2, 0.02)
        jitter = st.slider("散布点のJitter量", 0.05, 0.4, 0.15, 0.01)
        point_size = st.slider("散布点サイズ", 2, 15, 6, 1)
        point_alpha = st.slider("散布点透明度", 0.2, 1.0, 0.7, 0.05)
        point_color_mode = st.radio("散布点の色", ["グレー（統一）", "グループ色"], index=0)

        st.header("5. フォントサイズ")
        fs_title = st.slider("タイトル", 10, 40, 20)
        fs_axis = st.slider("軸ラベル", 10, 40, 18)
        fs_tick = st.slider("目盛りラベル", 10, 40, 16)
        fs_sig = st.slider("有意差ラベル", 10, 40, 16)


# ===================== メイン描画 =====================
if plot_df is not None:
    # --- 統計検定 ---
    is_wilcoxon = test_choice.startswith("Wilcoxon")
    fell_back = False
    if is_wilcoxon and len(d1) == len(d2):
        _, p_val = wilcoxon(d1, d2)
        test_name_used = "Wilcoxon符号順位検定（対応あり）"
    else:
        if is_wilcoxon:
            fell_back = True
        _, p_val = mannwhitneyu(d1, d2, alternative="two-sided")
        test_name_used = "Mann-Whitney U検定（対応なし）"

    if p_val < 0.001:
        sig_label, p_label = "***", "p < 0.001"
    elif p_val < 0.01:
        sig_label, p_label = "**", "p < 0.01"
    elif p_val < 0.05:
        sig_label, p_label = "*", "p < 0.05"
    else:
        sig_label, p_label = "ns", f"p = {p_val:.3f}"

    # --- 描画 ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    sns.violinplot(
        data=plot_df, x="Group", y="Value", hue="Group", order=[g1, g2], hue_order=[g1, g2],
        split=True, inner=None, cut=0, density_norm="count",
        width=violin_width, palette=[color1, color2], linewidth=1.2,
        ax=ax, legend=False,
    )

    use_group_colors = point_color_mode.startswith("グループ")
    sns.stripplot(
        data=plot_df, x="Group", y="Value", order=[g1, g2],
        hue="Group" if use_group_colors else None,
        hue_order=[g1, g2] if use_group_colors else None,
        palette=[color1, color2] if use_group_colors else None,
        color=None if use_group_colors else "gray",
        size=point_size, jitter=jitter, alpha=point_alpha, ax=ax, legend=False,
        edgecolor="white", linewidth=0.3,
    )

    sns.boxplot(
        data=plot_df, x="Group", y="Value", order=[g1, g2], width=box_width,
        showcaps=True,
        boxprops={"facecolor": "none", "zorder": 10, "linewidth": 1.5},
        medianprops={"color": "black", "zorder": 11, "linewidth": 1.5},
        whiskerprops={"linewidth": 1.5, "zorder": 10},
        capprops={"linewidth": 1.5, "zorder": 10},
        showfliers=False, ax=ax,
    )

    ax.set_title(plot_title, fontsize=fs_title)
    ax.set_ylabel(y_label, fontsize=fs_axis)
    ax.set_xlabel("")

    def to_mathtext(label):
        if "_" in label:
            main, sub = label.split("_", 1)
            return f"${main}_{{\\mathrm{{{sub}}}}}$"
        return label

    ax.set_xticks([0, 1])
    if use_mathtext:
        ax.set_xticklabels([to_mathtext(g1), to_mathtext(g2)], fontsize=fs_tick)
    else:
        ax.set_xticklabels([g1, g2], fontsize=fs_tick)

    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.tick_params(axis="x", length=0)

    if show_zero_line:
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2)

    if show_sig:
        y_max = plot_df["Value"].max()
        y_min = plot_df["Value"].min()
        data_range = y_max - y_min
        top_padding = data_range * 0.25
        ax.set_ylim(y_min - data_range * 0.1, y_max + top_padding)
        y = y_max + top_padding * 0.2
        h = data_range * 0.03
        ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], lw=1.5, color="black")
        ax.text(0.5, y + h, f"{sig_label} ({p_label})", ha="center", va="bottom", fontsize=fs_sig)

    sns.despine(ax=ax)

    if fell_back:
        st.warning(
            f"左右でサンプル数が異なるため（{len(d1)}件 / {len(d2)}件）、"
            "Wilcoxon検定は実行できません。代わりにMann-Whitney U検定の結果を表示しています。"
        )
    st.caption(f"使用した検定：{test_name_used}　/　p = {p_val:.4g}")

    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        label="📥 PNGダウンロード (Publication Quality)",
        data=buf,
        file_name="raincloud_plot.png",
        mime="image/png",
    )
else:
    st.info("👈 サイドバーからデータを選んでください")
