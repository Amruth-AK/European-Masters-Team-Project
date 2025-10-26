import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_dashboard(analysis_dict: dict):
    """
    Creates a more visual and interactive dashboard from the analysis dictionary.
    This version fixes the UnboundLocalError and the overlapping box plots.

    Args:
        analysis_dict: The dictionary containing the analysis results.
    """
    st.set_page_config(layout="wide")
    st.title("📊 Automated Data Analysis Dashboard")

    # Sidebar for navigation
    st.sidebar.title("Analysis Sections")
    page_options = [
        'general_info', 'missing_values', 'descriptive_statistics',
        'distributions', 'categorical_info', 'correlations', 'outlier_info'
    ]
    page = st.sidebar.radio("Go to", page_options, format_func=lambda x: x.replace('_', ' ').title())

    st.header(page.replace('_', ' ').title())

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

    elif page == 'descriptive_statistics':
        st.subheader("Descriptive Statistics for Numerical Columns")
        stats_dict = analysis_dict['descriptive_statistics']
        numerical_cols = list(stats_dict.keys())

        fig = go.Figure()
        for col in numerical_cols:
            stats = stats_dict[col]
            # --- FIX: Added x=[col] to place each box plot in its own category ---
            fig.add_trace(go.Box(
                x=[col],  # Assigns the box to a specific category on the x-axis
                name=col,
                q1=[stats['25%']],
                median=[stats['50%']],
                q3=[stats['75%']],
                lowerfence=[stats['min']],
                upperfence=[stats['max']]
            ))
        # --- END FIX ---

        fig.update_layout(title_text="Box Plots of Numerical Columns (from Summary Stats)")
        st.plotly_chart(fig, use_container_width=True)

        desc_stats_df = pd.DataFrame(stats_dict)
        st.dataframe(desc_stats_df)

    elif page == 'distributions':
        st.subheader("Distribution Analysis for Numerical Columns")
        # --- FIX: Define numerical_cols here to fix the UnboundLocalError ---
        numerical_cols = list(analysis_dict['distributions'].keys())
        # --- END FIX ---

        selected_col_hist = st.selectbox("Select a column to view its distribution", numerical_cols)
        if selected_col_hist:
            hist_data = analysis_dict['histogram_data'][selected_col_hist]
            fig = go.Figure(data=[go.Bar(
                y=hist_data['counts'],
                x=hist_data['bin_edges'][:-1],
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

    elif page == 'correlations':
        st.subheader("Correlation Analysis")
        st.markdown("#### Correlation Heatmap")
        corr_matrix = pd.DataFrame(analysis_dict['correlations']['correlation_matrix'])
        fig = px.imshow(
            corr_matrix, text_auto=True, aspect="auto",
            title="Correlation Matrix of Numerical Columns",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Correlation with Target Column")
        target_corr = analysis_dict['correlations']['target_correlation']
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

    elif page == 'outlier_info':
        st.subheader("Outlier Analysis (IQR Method)")
        outlier_df = pd.DataFrame.from_dict(analysis_dict['outlier_info'], orient='index').reset_index()
        outlier_df.columns = ['Column', 'Lower Bound', 'Upper Bound', 'Outlier Count', 'Outlier Percentage']

        fig = px.bar(
            outlier_df, x='Column', y='Outlier Percentage',
            title='Percentage of Outliers per Numerical Column',
            text=outlier_df['Outlier Percentage'].apply(lambda x: f'{x:.2f}%')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(outlier_df)
        st.info("Outliers are detected using the 1.5 * IQR rule.")


if __name__ == '__main__':

    sample_df = pd.read_csv(r'train.csv')

    # Paste the dictionary output from your analysis function here
    analysis_output = analyze.analysis(sample_df, 'accident_risk')
    create_dashboard(analysis_output)


