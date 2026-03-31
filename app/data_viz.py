import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


@st.cache_data
def load_data():
    return pd.read_csv("data/raw/train.csv")


def render_data_viz():
    st.markdown("## Data Visualization")

    df = load_data()

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    feature_num_cols = [c for c in num_cols if c != "SalePrice"]

    tab1, tab2, tab3 = st.tabs(
        ["🔍 EDA Overview", "📊 Univariate Analysis", "🔗 Bivariate & Multivariate"]
    )

    # =========================================================
    # TAB 1 — EDA OVERVIEW
    # =========================================================
    with tab1:
        st.markdown("### Dataset at a Glance")

        # ── Quick stats row ──────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Numerical Features", len(num_cols))
        c4.metric("Categorical Features", len(cat_cols))

        st.divider()

        # ── Missing values ───────────────────────────────────
        st.markdown("#### Missing Values")
        missing = (
            df.isnull()
            .sum()
            .reset_index()
            .rename(columns={"index": "Feature", 0: "Missing Count"})
        )
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        missing = missing[missing["Missing Count"] > 0].sort_values(
            "Missing %", ascending=False
        )

        if missing.empty:
            st.success("No missing values found!")
        else:
            fig_miss = px.bar(
                missing,
                x="Feature",
                y="Missing %",
                text="Missing %",
                color="Missing %",
                color_continuous_scale="OrRd",
                title=f"{len(missing)} features have missing values",
            )
            fig_miss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_miss.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_miss, use_container_width=True)

        st.divider()

        # ── Target distribution ──────────────────────────────
        st.markdown("#### Target Variable — SalePrice")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_target = px.histogram(
                df,
                x="SalePrice",
                nbins=60,
                marginal="box",
                title="SalePrice Distribution (Raw)",
                color_discrete_sequence=["#636EFA"],
            )
            st.plotly_chart(fig_target, use_container_width=True)

        with col_b:
            fig_log = px.histogram(
                df,
                x=np.log1p(df["SalePrice"]),
                nbins=60,
                marginal="box",
                title="SalePrice Distribution (Log-transformed)",
                color_discrete_sequence=["#EF553B"],
                labels={"x": "log(SalePrice + 1)"},
            )
            st.plotly_chart(fig_log, use_container_width=True)

        skew_val = df["SalePrice"].skew()
        kurt_val = df["SalePrice"].kurt()
        st.caption(
            f"**Skewness:** {skew_val:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Kurtosis:** {kurt_val:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Mean:** ${df['SalePrice'].mean():,.0f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Median:** ${df['SalePrice'].median():,.0f}"
        )

        st.divider()

        # ── Skewness of numerical features ──────────────────
        st.markdown("#### Skewness of Numerical Features")
        skew_df = (
            df[feature_num_cols]
            .skew()
            .reset_index()
            .rename(columns={"index": "Feature", 0: "Skewness"})
            .sort_values("Skewness", key=abs, ascending=False)
        )
        skew_df["Abs Skewness"] = skew_df["Skewness"].abs()
        skew_df["Flag"] = skew_df["Abs Skewness"].apply(
            lambda x: "High (>1)" if x > 1 else ("Moderate (0.5–1)" if x > 0.5 else "Low (<0.5)")
        )

        fig_skew = px.bar(
            skew_df.head(25),
            x="Skewness",
            y="Feature",
            orientation="h",
            color="Flag",
            color_discrete_map={
                "High (>1)": "#EF553B",
                "Moderate (0.5–1)": "#FFA15A",
                "Low (<0.5)": "#00CC96",
            },
            title="Top 25 Most Skewed Features",
        )
        fig_skew.update_layout(yaxis=dict(autorange="reversed"), height=550)
        st.plotly_chart(fig_skew, use_container_width=True)

        st.divider()

        # ── Data types summary ───────────────────────────────
        st.markdown("#### Data Types & Sample")
        st.dataframe(
            df.dtypes.reset_index().rename(columns={"index": "Feature", 0: "DType"}),
            use_container_width=True,
            height=220,
        )
        st.markdown("**Sample rows**")
        st.dataframe(df.sample(5, random_state=42), use_container_width=True)

    # =========================================================
    # TAB 2 — UNIVARIATE ANALYSIS
    # =========================================================
    with tab2:

        u_type = st.radio(
            "Analyse",
            ["Numerical Feature", "Categorical Feature", "Top Correlations with SalePrice"],
            horizontal=True,
        )

        st.divider()

        # ── Numerical ─────────────────────────────────────────
        if u_type == "Numerical Feature":
            feature = st.selectbox("Select Numerical Feature", feature_num_cols)

            col1, col2 = st.columns(2)

            with col1:
                show_kde = st.toggle("Overlay KDE curve", value=True)
                nbins = st.slider("Number of bins", 10, 100, 50)

                fig_hist = px.histogram(
                    df,
                    x=feature,
                    nbins=nbins,
                    marginal="rug",
                    title=f"Distribution of {feature}",
                    color_discrete_sequence=["#636EFA"],
                )

                if show_kde:
                    kde_x = np.linspace(df[feature].dropna().min(), df[feature].dropna().max(), 300)
                    kde_y = stats.gaussian_kde(df[feature].dropna())(kde_x)
                    kde_y_scaled = kde_y * len(df[feature].dropna()) * (df[feature].max() - df[feature].min()) / nbins
                    fig_hist.add_scatter(
                        x=kde_x,
                        y=kde_y_scaled,
                        mode="lines",
                        line=dict(color="#EF553B", width=2),
                        name="KDE",
                    )

                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                fig_box = px.box(
                    df,
                    y=feature,
                    points="outliers",
                    title=f"Box Plot — {feature}",
                    color_discrete_sequence=["#AB63FA"],
                )
                st.plotly_chart(fig_box, use_container_width=True)

            # Summary stats
            desc = df[feature].describe().round(2)
            st.markdown("**Summary Statistics**")
            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.metric("Mean", f"{desc['mean']:.2f}")
            s2.metric("Std", f"{desc['std']:.2f}")
            s3.metric("Min", f"{desc['min']:.2f}")
            s4.metric("25%", f"{desc['25%']:.2f}")
            s5.metric("75%", f"{desc['75%']:.2f}")
            s6.metric("Max", f"{desc['max']:.2f}")

        # ── Categorical ───────────────────────────────────────
        elif u_type == "Categorical Feature":
            feature = st.selectbox("Select Categorical Feature", cat_cols)

            col1, col2 = st.columns(2)

            vc = df[feature].value_counts().reset_index()
            vc.columns = [feature, "Count"]
            vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)

            with col1:
                fig_bar = px.bar(
                    vc,
                    x=feature,
                    y="Count",
                    text="Count",
                    title=f"Value Counts — {feature}",
                    color="Count",
                    color_continuous_scale="Blues",
                )
                fig_bar.update_traces(textposition="outside")
                fig_bar.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                fig_pie = px.pie(
                    vc,
                    names=feature,
                    values="Count",
                    title=f"Proportion — {feature}",
                    hole=0.4,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # ── Top Correlations with SalePrice ───────────────────
        elif u_type == "Top Correlations with SalePrice":
            n = st.slider("Show top N features", 5, 30, 15)

            corr_series = (
                df[num_cols]
                .corr()["SalePrice"]
                .drop("SalePrice")
                .sort_values(key=abs, ascending=False)
                .head(n)
                .reset_index()
            )
            corr_series.columns = ["Feature", "Correlation"]
            corr_series["Direction"] = corr_series["Correlation"].apply(
                lambda x: "Positive" if x >= 0 else "Negative"
            )

            fig_corr = px.bar(
                corr_series,
                x="Correlation",
                y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={"Positive": "#00CC96", "Negative": "#EF553B"},
                title=f"Top {n} Features Correlated with SalePrice",
                text="Correlation",
            )
            fig_corr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_corr.update_layout(yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

    # =========================================================
    # TAB 3 — BIVARIATE & MULTIVARIATE
    # =========================================================
    with tab3:

        bv_type = st.radio(
            "Analysis Type",
            [
                "Scatter Plot",
                "Categorical vs SalePrice",
                "Correlation Matrix",
                "Pair Plot",
            ],
            horizontal=True,
        )

        st.divider()

        # ── Scatter ───────────────────────────────────────────
        if bv_type == "Scatter Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x = st.selectbox("X-axis", feature_num_cols, index=0)
            with col2:
                y_opts = num_cols
                y = st.selectbox("Y-axis", y_opts, index=y_opts.index("SalePrice") if "SalePrice" in y_opts else 1)
            with col3:
                color_by = st.selectbox("Colour by (optional)", ["None"] + cat_cols)

            trendline = st.toggle("Show trendline (OLS)", value=True)

            fig_scatter = px.scatter(
                df,
                x=x,
                y=y,
                color=None if color_by == "None" else color_by,
                trendline="ols" if trendline else None,
                opacity=0.65,
                title=f"{x} vs {y}",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            if trendline:
                r, p = stats.pearsonr(df[x].dropna(), df[[x, y]].dropna()[y])
                st.caption(f"**Pearson r:** {r:.4f} &nbsp;&nbsp;|&nbsp;&nbsp; **p-value:** {p:.4e}")

        # ── Categorical vs SalePrice ───────────────────────────
        elif bv_type == "Categorical vs SalePrice":
            feature = st.selectbox("Select Categorical Feature", cat_cols)

            plot_kind = st.radio("Plot type", ["Violin", "Box", "Bar (Mean)"], horizontal=True)

            # Sort categories by median SalePrice
            order = (
                df.groupby(feature)["SalePrice"]
                .median()
                .sort_values()
                .index.tolist()
            )

            if plot_kind == "Violin":
                fig = px.violin(
                    df,
                    x=feature,
                    y="SalePrice",
                    box=True,
                    points="outliers",
                    category_orders={feature: order},
                    title=f"SalePrice by {feature}",
                )
            elif plot_kind == "Box":
                fig = px.box(
                    df,
                    x=feature,
                    y="SalePrice",
                    points="outliers",
                    category_orders={feature: order},
                    title=f"SalePrice by {feature}",
                    color=feature,
                )
            else:
                mean_df = (
                    df.groupby(feature)["SalePrice"]
                    .agg(["mean", "count"])
                    .reset_index()
                    .rename(columns={"mean": "Mean SalePrice", "count": "Count"})
                    .sort_values("Mean SalePrice")
                )
                fig = px.bar(
                    mean_df,
                    x=feature,
                    y="Mean SalePrice",
                    text="Mean SalePrice",
                    color="Mean SalePrice",
                    color_continuous_scale="Viridis",
                    title=f"Mean SalePrice by {feature}",
                    category_orders={feature: mean_df[feature].tolist()},
                )
                fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
                fig.update_layout(coloraxis_showscale=False)

            fig.update_layout(xaxis_tickangle=-35, height=500)
            st.plotly_chart(fig, use_container_width=True)

        # ── Correlation Matrix ────────────────────────────────
        elif bv_type == "Correlation Matrix":
            default_feats = feature_num_cols[:8]
            selected = st.multiselect(
                "Select numerical features (include SalePrice to see target correlation)",
                num_cols,
                default=default_feats + ["SalePrice"],
            )

            if len(selected) < 2:
                st.warning("Please select at least 2 features.")
            else:
                corr = df[selected].corr()

                fig_hm = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Correlation Matrix",
                    aspect="auto",
                )
                fig_hm.update_layout(height=max(400, len(selected) * 45))
                st.plotly_chart(fig_hm, use_container_width=True)

        # ── Pair Plot ─────────────────────────────────────────
        elif bv_type == "Pair Plot":
            selected = st.multiselect(
                "Select 2–5 features (SalePrice auto-included)",
                feature_num_cols,
                default=feature_num_cols[:4],
            )
            color_cat = st.selectbox(
                "Colour by (optional categorical)", ["None"] + cat_cols, index=0
            )

            if not (2 <= len(selected) <= 5):
                st.warning("Select between 2 and 5 features.")
            else:
                cols_to_plot = selected + ["SalePrice"]
                color_col = None if color_cat == "None" else color_cat

                if color_col:
                    # Reduce cardinality for readability
                    top_cats = df[color_col].value_counts().head(6).index
                    plot_df = df[cols_to_plot + [color_col]].copy()
                    plot_df[color_col] = plot_df[color_col].apply(
                        lambda x: x if x in top_cats else "Other"
                    )
                else:
                    plot_df = df[cols_to_plot]

                fig_pair = px.scatter_matrix(
                    plot_df,
                    dimensions=cols_to_plot,
                    color=color_col,
                    opacity=0.5,
                    title="Pair Plot",
                )
                fig_pair.update_traces(diagonal_visible=True, showupperhalf=False)
                fig_pair.update_layout(height=700)
                st.plotly_chart(fig_pair, use_container_width=True)