# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_dashboard(analysis_dict: dict, page: str):
    """
    Renders the specific analysis page requested by the user.

    Args:
        analysis_dict: The dictionary containing all analysis results.
        page: The string name of the page to display.
    """
    st.title("📊 Automated Data Analysis Dashboard")
    st.header(page.replace('_', ' ').title())

    # --- Page Rendering Logic ---

    if page == 'General Info':
        st.subheader("Dataset Overview")
        info = analysis_dict['general_info']
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Rows", info['shape'][0])
        col2.metric("Number of Columns", info['shape'][1])
        mem_usage_mb = info['memory_usage'] / (1024 * 1024)
        col3.metric("Memory Usage (MB)", f"{mem_usage_mb:.2f}")

        st.subheader("Data Types")
        df_types = pd.DataFrame(list(info['data_types'].items()), columns=['Column', 'Data Type'])
        st.table(df_types)

    elif page == 'Missing Values':
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

    elif page == 'Descriptive Statistics':
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

    elif page == 'Distributions':
        st.subheader("Distribution Analysis for Numerical Columns")
        numerical_cols = list(analysis_dict['distributions'].keys())
        selected_col_hist = st.selectbox("Select a column to view its distribution", numerical_cols)
        if selected_col_hist:
            hist_data = analysis_dict['histogram_data'][selected_col_hist]
            fig = go.Figure(data=[go.Bar(
                y=hist_data['counts'], x=hist_data['bin_edges'][:-1],
                width=[hist_data['bin_edges'][i + 1] - hist_data['bin_edges'][i] for i in range(len(hist_data['counts']))]
            )])
            fig.update_layout(title_text=f'Distribution of {selected_col_hist} (Reconstructed)',
                              xaxis_title=selected_col_hist, yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Skewness")
        dist_df = pd.DataFrame.from_dict(analysis_dict['distributions'], orient='index').reset_index()
        dist_df.columns = ['Column', 'Skewness']
        fig_skew = px.bar(
            dist_df, x='Column', y='Skewness', title='Skewness of Numerical Columns',
            color='Skewness', color_continuous_scale=px.colors.diverging.RdYlGn_r, color_continuous_midpoint=0
        )
        st.plotly_chart(fig_skew, use_container_width=True)
        st.info("Skewness indicates the asymmetry of the data distribution. A value of 0 suggests a symmetrical distribution.")

    elif page == 'Correlations':
        st.subheader("Correlation Analysis")
        st.markdown("#### Correlation Heatmap")
        corr_matrix = pd.DataFrame(analysis_dict['correlations']['correlation_matrix'])
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Correlation with Target Column")
        target_corr = analysis_dict['correlations']['target_correlation']
        if isinstance(target_corr, dict):
            target_corr_df = pd.DataFrame.from_dict(target_corr, orient='index', columns=['Correlation']).reset_index()
            target_corr_df.columns = ['Feature', 'Correlation']
            fig2 = px.bar(target_corr_df.sort_values('Correlation', ascending=False), x='Feature', y='Correlation',
                          title='Feature Correlation with Target', color='Correlation', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Target column is not numeric, so target correlation is not available.")

    elif page == 'Categorical Info':
        st.subheader("Categorical Column Analysis")
        categorical_cols = list(analysis_dict['categorical_info'].keys())
        if not categorical_cols:
            st.info("No categorical columns found.")
        else:
            selected_cat_col = st.selectbox("Select a Categorical Column", categorical_cols)
            if selected_cat_col:
                cat_info = analysis_dict['categorical_info'][selected_cat_col]
                st.metric(f"Unique Values in {selected_cat_col}", cat_info['unique_values'])
                value_counts_df = pd.DataFrame(list(cat_info['value_counts'].items()), columns=['Category', 'Count'])
                plot_type = st.radio("Select plot type", ('Bar Chart', 'Pie Chart'))
                if plot_type == 'Bar Chart':
                    fig = px.bar(value_counts_df, x='Category', y='Count', title=f"Value Counts for {selected_cat_col}")
                else:
                    fig = px.pie(value_counts_df, names='Category', values='Count', title=f"Value Counts for {selected_cat_col}")
                st.plotly_chart(fig, use_container_width=True)

    elif page == 'Outlier Info':
        st.subheader("Outlier Analysis (IQR Method)")
        outlier_df = pd.DataFrame.from_dict(analysis_dict['outlier_info'], orient='index').reset_index()
        outlier_df.columns = ['Column', 'Lower Bound', 'Upper Bound', 'Total Outlier Count', 'Total Outlier %',
                              'Lower Outlier Count', 'Lower Outlier %', 'Upper Outlier Count', 'Upper Outlier %']
        plot_df = outlier_df[['Column', 'Lower Outlier %', 'Upper Outlier %']].melt(
            id_vars='Column', var_name='Outlier Type', value_name='Percentage'
        )
        fig = px.bar(
            plot_df, x='Column', y='Percentage', color='Outlier Type', title='Percentage and Type of Outliers',
            color_discrete_map={'Lower Outlier %': 'blue', 'Upper Outlier %': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(outlier_df)
        st.info("Outliers are detected using the 1.5 * IQR rule.")

    elif page == 'Duplicate Analysis':
        st.subheader("Duplicate Row Analysis")
        row_info = analysis_dict['row_duplicate_info']
        
        # --- NEW: Inform user about ignored columns ---
        ignored_cols = row_info.get('ignored_columns')
        if ignored_cols:
            st.info(f"The following identifier columns were excluded from this analysis: `{'`, `'.join(ignored_cols)}`")
        
        col1, col2 = st.columns(2)
        # --- UPDATED: Use the correct keys from the new analysis results ---
        col1.metric("Total Duplicate Rows", row_info.get('total_duplicates', 0))
        col2.metric("Duplicate Row Percentage", f"{row_info.get('duplicate_percentage', 0):.2f}%")
        
        # Display sample of duplicate rows if they exist
        if row_info.get('total_duplicates', 0) > 0:
            st.write("Sample of duplicate rows found:")
            st.dataframe(pd.DataFrame(row_info['duplicate_rows']).head())

        st.subheader("Duplicate Value Analysis (per Feature)")
        feature_dup_df = pd.DataFrame.from_dict(analysis_dict['feature_duplicate_info'], orient='index').reset_index()
        feature_dup_df.columns = ['Column', 'Duplicate Count', 'Duplicate %', 'Most Frequent Value', 'Most Frequent Count']
        fig = px.bar(
            feature_dup_df, x='Column', y='Duplicate %', title='Percentage of Duplicate Values per Column',
            text=feature_dup_df['Duplicate %'].apply(lambda x: f'{x:.2f}%')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(feature_dup_df)
