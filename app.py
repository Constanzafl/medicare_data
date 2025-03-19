import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="HCPCS Rate Trends (2016-2025)", layout="wide")

# Application title
st.title("Healthcare Procedure Rate Trends (2016-2025)")
st.markdown("Analysis of procedure pricing trends by category from 2016 to 2025")

# Function to load data from a single file containing all years
@st.cache_data
def load_data():
    try:
        # Intenta cargar el archivo consolidado - ajusta el nombre según sea necesario
        file_name = "hcpcs_pivoted_filtered.csv"
        df = pd.read_csv(file_name)
        st.success(f"Successfully loaded data from {file_name}")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Display basic information about the dataset
    st.subheader("Dataset Overview")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data Summary", "Rate Trends", "Category Comparison", "Procedure Details"])
    
    with tab1:
        # Show dataset info
        st.write(f"Total records: {df.shape[0]}")
        st.write(f"Time period: {df['year'].min()} - {df['year'].max()}")
        st.write(f"Total procedures: {df['hcpc'].nunique()}")
        st.write(f"Total categories: {df['category'].nunique()}")
        
        # Display unique categories
        categories = df['category'].unique()
        st.write("Procedure Categories:")
        st.write(categories)
        
        # Show sample of the data
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
    
    with tab2:
        st.subheader("Rate Trends Analysis (2016-2025)")
        
        # Filter for National rates (MAC Locality = 0) and no modifiers
        national_df = df[(df['locality'] == 0) & (df['modifier'].isna())]
        
        # Create category groupings according to specifications
        non_facility_categories = ['Evaluation & Management', 'Medicine']
        facility_categories = [
            'Radiology', 'Surgery - Cardiovascular', 'Surgery - Digestive', 
            'Surgery - Ear', 'Surgery - Eye', 'Surgery - Female Genital',
            'Surgery - Integumentary', 'Surgery - Male Genital', 
            'Surgery - Musculoskeletal', 'Surgery - Nervous', 
            'Surgery - Respiratory', 'Surgery - Urinary'
        ]
        
        # Function to determine which rate to use based on category
        def select_rate(row):
            if row['category'] in non_facility_categories:
                return row['price']  # Non-facility rate
            elif row['category'] in facility_categories:
                return row['limit_charge']  # Facility rate
            else:
                return row['price']  # Default to non-facility
        
        # Apply the function to create a new column with the appropriate rate
        national_df['selected_rate'] = national_df.apply(select_rate, axis=1)
        
        # Group by year and category to get average rates
        yearly_avg = national_df.groupby(['year', 'category'])['selected_rate'].mean().reset_index()
        
        # Create selectors for visualization
        selected_categories = st.multiselect(
            "Select Categories to Display",
            options=sorted(national_df['category'].unique()),
            default=sorted(national_df['category'].unique())[0:3]
        )
        
        # Filter data based on selection
        if selected_categories:
            filtered_data = yearly_avg[yearly_avg['category'].isin(selected_categories)]
            
            # Create line chart for rate trends
            fig = px.line(
                filtered_data, 
                x='year', 
                y='selected_rate', 
                color='category',
                title="Average Rate Trends by Procedure Category (2016-2025)",
                labels={'selected_rate': 'Average Rate ($)', 'year': 'Year', 'category': 'Category'},
                markers=True,
                line_shape='linear'
            )
            
            # Customize the chart
            fig.update_layout(
                xaxis=dict(tickmode='linear', dtick=1),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display growth rates
            st.subheader("Rate Growth Analysis")
            
            # Get the first and last years
            min_year = filtered_data['year'].min()
            max_year = filtered_data['year'].max()
            
            # Calculate growth for each category
            growth_data = []
            
            for category in selected_categories:
                category_data = filtered_data[filtered_data['category'] == category]
                if len(category_data) > 1:
                    first_year_rate = category_data[category_data['year'] == min_year]['selected_rate'].values
                    last_year_rate = category_data[category_data['year'] == max_year]['selected_rate'].values
                    
                    if len(first_year_rate) > 0 and len(last_year_rate) > 0:
                        first = first_year_rate[0]
                        last = last_year_rate[0]
                        total_growth = (last - first) / first * 100 if first > 0 else 0
                        years_diff = max_year - min_year
                        annual_growth = ((last / first) ** (1 / years_diff) - 1) * 100 if first > 0 and years_diff > 0 else 0
                        
                        growth_data.append({
                            'Category': category,
                            'First Year Rate': round(first, 2),
                            'Last Year Rate': round(last, 2),
                            'Total Growth (%)': round(total_growth, 2),
                            'Annual Growth (%)': round(annual_growth, 2)
                        })
            
            if growth_data:
                growth_df = pd.DataFrame(growth_data)
                st.dataframe(growth_df, use_container_width=True)
                
                # Create a bar chart for total growth
                fig_growth = px.bar(
                    growth_df, 
                    x='Category', 
                    y='Total Growth (%)',
                    title=f"Total Rate Growth by Category ({min_year}-{max_year})",
                    color='Total Growth (%)',
                    color_continuous_scale='RdBu',
                    text_auto='.1f'
                )
                
                fig_growth.update_layout(height=500)
                st.plotly_chart(fig_growth, use_container_width=True)
        else:
            st.warning("Please select at least one category to display.")
    
    with tab3:
        st.subheader("Category Comparison")
        
        # Same filtering for National data
        national_df = df[(df['locality'] == 0) & (df['modifier'].isna())]
        national_df['selected_rate'] = national_df.apply(select_rate, axis=1)
        
        # Select years for comparison
        years = sorted(national_df['year'].unique())
        
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.selectbox("Select Start Year", options=years, index=0)
        with col2:
            end_year = st.selectbox("Select End Year", options=years, index=len(years)-1)
        
        if start_year and end_year:
            # Filter data for selected years
            start_data = national_df[national_df['year'] == start_year]
            end_data = national_df[national_df['year'] == end_year]
            
            # Group by category and calculate average rates
            start_avg = start_data.groupby('category')['selected_rate'].mean().reset_index()
            end_avg = end_data.groupby('category')['selected_rate'].mean().reset_index()
            
            # Merge the data
            comparison_df = pd.merge(start_avg, end_avg, on='category', suffixes=('_start', '_end'))
            comparison_df['growth'] = (comparison_df['selected_rate_end'] - comparison_df['selected_rate_start']) / comparison_df['selected_rate_start'] * 100
            
            # Sort by growth
            comparison_df = comparison_df.sort_values('growth', ascending=False)
            
            # Create visualization
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bars for start and end years
            fig.add_trace(
                go.Bar(
                    x=comparison_df['category'],
                    y=comparison_df['selected_rate_start'],
                    name=f'{start_year} Rate',
                    marker_color='lightblue'
                )
            )
            
            fig.add_trace(
                go.Bar(
                    x=comparison_df['category'],
                    y=comparison_df['selected_rate_end'],
                    name=f'{end_year} Rate',
                    marker_color='darkblue'
                )
            )
            
            # Add line for growth percentage
            fig.add_trace(
                go.Scatter(
                    x=comparison_df['category'],
                    y=comparison_df['growth'],
                    name='Growth %',
                    line=dict(color='red', width=2),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title=f"Rate Comparison by Category: {start_year} vs {end_year}",
                xaxis=dict(title='Category', tickangle=45),
                yaxis=dict(title='Average Rate ($)'),
                yaxis2=dict(title='Growth (%)', range=[min(comparison_df['growth'])-10, max(comparison_df['growth'])+10]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=700,
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the data in tabular format
            st.subheader(f"Category Rate Comparison: {start_year} vs {end_year}")
            comparison_display = comparison_df.copy()
            comparison_display = comparison_display.rename(columns={
                'selected_rate_start': f'{start_year} Avg Rate',
                'selected_rate_end': f'{end_year} Avg Rate',
                'growth': 'Growth (%)'
            })
            
            comparison_display[f'{start_year} Avg Rate'] = comparison_display[f'{start_year} Avg Rate'].round(2)
            comparison_display[f'{end_year} Avg Rate'] = comparison_display[f'{end_year} Avg Rate'].round(2)
            comparison_display['Growth (%)'] = comparison_display['Growth (%)'].round(2)
            
            st.dataframe(comparison_display, use_container_width=True)
    
    with tab4:
        st.subheader("Procedure-Level Analysis")
        
        # Filter for National data
        national_df = df[(df['locality'] == 0) & (df['modifier'].isna())]
        national_df['selected_rate'] = national_df.apply(select_rate, axis=1)
        
        # Group by procedure code and year
        proc_df = national_df.groupby(['hcpc', 'year', 'category', 'sdesc'])['selected_rate'].mean().reset_index()
        
        # Select a category
        selected_category = st.selectbox(
            "Select Procedure Category",
            options=sorted(national_df['category'].unique())
        )
        
        # Filter procedures for the selected category
        category_procs = proc_df[proc_df['category'] == selected_category]
        
        # Get unique procedure codes for the category
        proc_codes = sorted(category_procs['hcpc'].unique())
        
        # Allow selecting up to 5 procedures
        selected_procs = st.multiselect(
            "Select Procedures to Compare (up to 5)",
            options=proc_codes,
            max_selections=5,
            format_func=lambda x: f"{x} - {category_procs[category_procs['hcpc'] == x]['sdesc'].iloc[0] if len(category_procs[category_procs['hcpc'] == x]) > 0 else ''}"
        )
        
        if selected_procs:
            # Filter data for selected procedures
            selected_data = category_procs[category_procs['hcpc'].isin(selected_procs)]
            
            # Create line chart
            fig = px.line(
                selected_data,
                x='year',
                y='selected_rate',
                color='hcpc',
                hover_data=['sdesc'],
                title=f"Rate Trends for Selected {selected_category} Procedures (2016-2025)",
                labels={'selected_rate': 'Rate ($)', 'year': 'Year', 'hcpc': 'Procedure Code'},
                markers=True
            )
            
            # Custom hover template to include description
            fig.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>Year: %{x}<br>Rate: $%{y:.2f}<extra></extra>"
            )
            
            # Update layout
            fig.update_layout(
                xaxis=dict(tickmode='linear', dtick=1),
                legend_title="Procedure Code - Description",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate growth rates for selected procedures
            min_year = selected_data['year'].min()
            max_year = selected_data['year'].max()
            
            proc_growth = []
            
            for proc in selected_procs:
                proc_data = selected_data[selected_data['hcpc'] == proc]
                
                if len(proc_data) > 1:
                    desc = proc_data['sdesc'].iloc[0]
                    first_year = proc_data[proc_data['year'] == min_year]
                    last_year = proc_data[proc_data['year'] == max_year]
                    
                    if not first_year.empty and not last_year.empty:
                        first_rate = first_year['selected_rate'].values[0]
                        last_rate = last_year['selected_rate'].values[0]
                        
                        total_growth = (last_rate - first_rate) / first_rate * 100 if first_rate > 0 else 0
                        years_diff = max_year - min_year
                        annual_growth = ((last_rate / first_rate) ** (1 / years_diff) - 1) * 100 if first_rate > 0 and years_diff > 0 else 0
                        
                        proc_growth.append({
                            'Procedure': proc,
                            'Description': desc,
                            f'{min_year} Rate': round(first_rate, 2),
                            f'{max_year} Rate': round(last_rate, 2),
                            'Total Growth (%)': round(total_growth, 2),
                            'Annual Growth (%)': round(annual_growth, 2)
                        })
            
            if proc_growth:
                growth_df = pd.DataFrame(proc_growth)
                st.subheader(f"Growth Analysis for Selected {selected_category} Procedures")
                st.dataframe(growth_df, use_container_width=True)
        else:
            st.info("Please select at least one procedure to display trends.")
else:
    st.error("Failed to load data. Please check the file path and format.")

# Add information about data filtering
st.sidebar.title("Analysis Parameters")
st.sidebar.markdown("### Data Filters Applied:")
st.sidebar.markdown("- **MAC Locality**: 0 (National)")
st.sidebar.markdown("- **Modifiers**: None (excluded)")
st.sidebar.markdown("### Rate Types Used:")
st.sidebar.markdown("**Non-facility Rates** for:")
st.sidebar.markdown("- Evaluation & Management\n- Medicine")
st.sidebar.markdown("**Facility Rates** for:")
st.sidebar.markdown("- Radiology\n- All Surgery Categories")

# Add explanations
st.sidebar.markdown("### About This Analysis")
st.sidebar.markdown("""
This dashboard analyzes healthcare procedure rate trends 
from 2016 to 2025, focusing on national rates without modifiers.

The analysis follows specific guidelines on which rate type 
to use based on procedure category.
""")
