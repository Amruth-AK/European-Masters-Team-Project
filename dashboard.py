import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import analyze


def create_dashboard(analysis_dict: dict):
    """
    Creates a visual and interactive dashboard from the analysis dictionary,
    including detailed outlier and duplicate analysis sections.

    Args:
        analysis_dict: The dictionary containing the analysis results.
    """
    st.set_page_config(layout="wide")
    st.title("📊 Automated Data Analysis Dashboard")

    # Sidebar for navigation
    st.sidebar.title("Analysis Sections")
    # Added 'duplicate_analysis' to the page options
    page_options = [
        'general_info', 'missing_values', 'descriptive_statistics',
        'distributions', 'categorical_info', 'correlations', 'outlier_info',
        'duplicate_analysis'
    ]
    page = st.sidebar.radio("Go to", page_options, format_func=lambda x: x.replace('_', ' ').title())

    st.header(page.replace('_', ' ').title())

    # --- General Info Page ---
    if page == 'general_info':
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Rows", analysis_dict['general_info']['shape'][0])
        col2.metric("Number of Columns", analysis_dict['general_info']['shape'][1])
        mem_usage_mb = analysis_dict['general_info']['memory_usage'] / (1024 * 1024)
        col3.metric("Memory Usage (MB)", f"{mem_usage_mb:.2f}")

        st.subheader("Data Types")
        df_types = pd.DataFrame(list(analysis_dict['general_info']['data_types'].items()),
                                columns=['Column', 'Data Type'])
        st.table(df_types)

    # --- Missing Values Page ---
    elif page == 'missing_values':
        st.subheader("Missing Values Analysis")
        missing_df = pd.DataFrame.from_dict(analysis_dict['missing_values'], orient='index').reset_index()
        missing_df.columns = ['Column', 'Missing Count', 'Missing Percentage']

        plot_df = missing_df[missing_df['Missing Percentage'] > 0]
        if not plot_df.empty:
            fig = px.bar(
                plot_df, x='Column', y='Missing Percentage',
                title='Percentage of Missing Values per Column',
                text=plot_df['Missing Percentage'].apply(lambda x: f'{x:.2f}%')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values found in the dataset!")
        st.dataframe(missing_df)

    # --- Descriptive Statistics Page ---
    elif page == 'descriptive_statistics':
        st.subheader("Descriptive Statistics for Numerical Columns")
        stats_dict = analysis_dict['descriptive_statistics']
        numerical_cols = list(stats_dict.keys())

        fig = go.Figure()
        for col in numerical_cols:
            stats = stats_dict[col]
            fig.add_trace(go.Box(
                x=[col], name=col, q1=[stats['25%']], median=[stats['50%']],
                q3=[stats['75%']], lowerfence=[stats['min']], upperfence=[stats['max']]
            ))
        fig.update_layout(title_text="Box Plots of Numerical Columns (from Summary Stats)")
        st.plotly_chart(fig, use_container_width=True)

        desc_stats_df = pd.DataFrame(stats_dict)
        st.dataframe(desc_stats_df)

    # --- Distributions Page ---
    elif page == 'distributions':
        st.subheader("Distribution Analysis for Numerical Columns")
        numerical_cols = list(analysis_dict['distributions'].keys())

        selected_col_hist = st.selectbox("Select a column to view its distribution", numerical_cols)
        if selected_col_hist:
            hist_data = analysis_dict['histogram_data'][selected_col_hist]
            fig = go.Figure(data=[go.Bar(
                y=hist_data['counts'], x=hist_data['bin_edges'][:-1],
                width=[hist_data['bin_edges'][i + 1] - hist_data['bin_edges'][i] for i in
                       range(len(hist_data['counts']))]
            )])
            fig.update_layout(title_text=f'Distribution of {selected_col_hist} (Reconstructed)',
                              xaxis_title=selected_col_hist, yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Skewness")
        dist_df = pd.DataFrame.from_dict(analysis_dict['distributions'], orient='index').reset_index()
        dist_df.columns = ['Column', 'Skewness']
        fig_skew = px.bar(
            dist_df, x='Column', y='Skewness', title='Skewness of Numerical Columns',
            color='Skewness', color_continuous_scale=px.colors.diverging.RdYlGn_r,
            color_continuous_midpoint=0
        )
        st.plotly_chart(fig_skew, use_container_width=True)
        st.info(
            "Skewness indicates the asymmetry of the data distribution. A value of 0 suggests a symmetrical distribution.")

    # --- Correlations Page ---
    elif page == 'correlations':
        st.subheader("Correlation Analysis")
        st.markdown("#### Correlation Heatmap")
        corr_matrix = pd.DataFrame(analysis_dict['correlations']['correlation_matrix'])
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Matrix of Numerical Columns", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Correlation with Target Column")
        target_corr_df = pd.DataFrame.from_dict(analysis_dict['correlations']['target_correlation'], orient='index',
                                                columns=['Correlation']).reset_index()
        target_corr_df.columns = ['Feature', 'Correlation']
        fig2 = px.bar(target_corr_df.sort_values('Correlation', ascending=False),
                      x='Feature', y='Correlation', title='Feature Correlation with Target',
                      color='Correlation', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig2, use_container_width=True)

    # --- Categorical Info Page ---
    elif page == 'categorical_info':
        st.subheader("Categorical Column Analysis")
        categorical_cols = list(analysis_dict['categorical_info'].keys())
        selected_cat_col = st.selectbox("Select a Categorical Column", categorical_cols)

        if selected_cat_col:
            cat_info = analysis_dict['categorical_info'][selected_cat_col]
            st.metric(f"Unique Values in {selected_cat_col}", cat_info['unique_values'])
            value_counts_df = pd.DataFrame(list(cat_info['value_counts'].items()), columns=['Category', 'Count'])
            plot_type = st.radio("Select plot type", ('Bar Chart', 'Pie Chart'))

            if plot_type == 'Bar Chart':
                fig = px.bar(value_counts_df, x='Category', y='Count', title=f"Value Counts for {selected_cat_col}")
            else:
                fig = px.pie(value_counts_df, names='Category', values='Count',
                             title=f"Value Counts for {selected_cat_col}")
            st.plotly_chart(fig, use_container_width=True)

    # --- ENHANCED: Outlier Info Page ---
    elif page == 'outlier_info':
        st.subheader("Outlier Analysis (IQR Method)")
        outlier_df = pd.DataFrame.from_dict(analysis_dict['outlier_info'], orient='index').reset_index()
        # Update columns to match new detailed structure
        outlier_df.columns = [
            'Column', 'Lower Bound', 'Upper Bound', 'Total Outlier Count', 'Total Outlier Percentage',
            'Lower Outlier Count', 'Lower Outlier Percentage',
            'Upper Outlier Count', 'Upper Outlier Percentage'
        ]

        # Create a stacked bar chart for lower vs. upper outliers
        plot_df = outlier_df[['Column', 'Lower Outlier Percentage', 'Upper Outlier Percentage']]
        plot_df = plot_df.melt(id_vars='Column', var_name='Outlier Type', value_name='Percentage')

        fig = px.bar(
            plot_df,
            x='Column',
            y='Percentage',
            color='Outlier Type',
            title='Percentage and Type of Outliers per Numerical Column',
            labels={'Percentage': 'Outlier Percentage (%)'},
            color_discrete_map={
                'Lower Outlier Percentage': 'blue',
                'Upper Outlier Percentage': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(outlier_df)
        st.info("Outliers are detected using the 1.5 * IQR rule.")

    # --- NEW: Duplicate Analysis Page ---
    elif page == 'duplicate_analysis':
        st.subheader("Duplicate Row Analysis")
        row_info = analysis_dict['row_duplicate_info']
        col1, col2 = st.columns(2)
        col1.metric("Total Duplicate Rows", row_info.get('total_row_duplicates', 0))
        col2.metric("Duplicate Row Percentage", f"{row_info.get('duplicate_row_percentage', 0):.2f}%")

        st.subheader("Duplicate Value Analysis (per Feature)")
        feature_dup_df = pd.DataFrame.from_dict(analysis_dict['feature_duplicate_info'], orient='index').reset_index()
        feature_dup_df.columns = ['Column', 'Duplicate Count', 'Duplicate Percentage', 'Most Frequent Value',
                                  'Most Frequent Count']

        fig = px.bar(
            feature_dup_df, x='Column', y='Duplicate Percentage',
            title='Percentage of Duplicate Values per Column',
            text=feature_dup_df['Duplicate Percentage'].apply(lambda x: f'{x:.2f}%')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(feature_dup_df)





