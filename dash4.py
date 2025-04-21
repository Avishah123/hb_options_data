import streamlit as st
import pandas as pd
import psycopg2
from psycopg2 import sql
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sqlalchemy import create_engine
import urllib.parse

# Set page configuration
st.set_page_config(
    page_title="Options Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Database connection functions
@st.cache_resource
def init_connection():
    """Initialize connection to PostgreSQL - cached to reuse connection"""
    try:
        # Check if secrets are available
        if not hasattr(st, "secrets"):
            st.error("Secrets not found. Make sure you have a .streamlit/secrets.toml file.")
            return None
            
        # Get database credentials from secrets.toml
        # Using get() to avoid errors if keys don't exist
        secrets = st.secrets
        host = secrets.get("db_host", "")
        port = secrets.get("db_port", "")
        dbname = secrets.get("db_name", "")
        user = secrets.get("db_user", "")
        password = secrets.get("db_password", "")
        
        # Verify all credentials are present
        if not all([host, port, dbname, user, password]):
            st.error("Some database credentials are missing in secrets.toml.")
            return None
        
        # Create connection string
        conn_string = f"postgresql://{user}:{urllib.parse.quote_plus(password)}@{host}:{port}/{dbname}"
        engine = create_engine(conn_string)
        return engine
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

@st.cache_data
def load_data_from_postgres(query):
    """Load data from PostgreSQL with caching for performance"""
    try:
        engine = init_connection()
        if engine:
            df = pd.read_sql(query, engine)
            
            # Convert date columns to datetime objects
            date_columns = ['date', 'expiry_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            return df
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def filter_nearest_expiry_and_sum(df):
    """
    Filters dataframe to nearest expiry for each stock and calculates metrics.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Convert dates to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    if not pd.api.types.is_datetime64_any_dtype(df['expiry_date']):
        df['expiry_date'] = pd.to_datetime(df['expiry_date'])
    
    # Get the current date
    current_date = datetime.now().date()
    
    # Filter to only include expiry dates that are on or after the current date
    df_filtered = df[df['expiry_date'].dt.date >= current_date]
    
    # Find the nearest expiry date for each symbol
    nearest_expiry_by_symbol = df_filtered.groupby('symbol')['expiry_date'].min().reset_index()
    nearest_expiry_by_symbol.rename(columns={'expiry_date': 'nearest_expiry'}, inplace=True)
    
    # Merge this back to get all rows with the nearest expiry for each symbol
    nearest_expiry_df = pd.merge(df_filtered, nearest_expiry_by_symbol, on='symbol')
    nearest_expiry_df = nearest_expiry_df[nearest_expiry_df['expiry_date'] == nearest_expiry_df['nearest_expiry']]
    
    # Calculate the aggregate metrics per Symbol
    total_metrics = nearest_expiry_df.groupby('symbol').agg({
        'expiry_date': 'first',
        'new_total': 'sum',
        'total_buy_clients': 'sum',
        'total_sell_clients': 'sum'
    }).reset_index()
    
    # Only consider 'CE' and 'PE' option types
    valid_opt_types = ['CE', 'PE']
    filtered_for_ce_pe = nearest_expiry_df[nearest_expiry_df['opt_type'].isin(valid_opt_types)]
    
    # Calculate the sum of net_qty_carry_fwd by Symbol and OptType
    net_qty_by_type = filtered_for_ce_pe.groupby(['symbol', 'opt_type'])['net_qty_carry_fwd'].sum().reset_index()
    
    # Pivot to have separate columns for CE and PE
    pivoted_net_qty = net_qty_by_type.pivot(index='symbol', columns='opt_type', values='net_qty_carry_fwd').reset_index()
    pivoted_net_qty.columns.name = None
    
    # Ensure CE and PE columns exist
    for col in ['CE', 'PE']:
        if col not in pivoted_net_qty.columns:
            pivoted_net_qty[col] = 0
    
    # Limit columns to only Symbol, CE, and PE
    pivoted_net_qty = pivoted_net_qty[['symbol'] + [col for col in pivoted_net_qty.columns if col in ['CE', 'PE']]]
    
    # Merge the pivoted net quantity data with the total metrics
    final_result = pd.merge(total_metrics, pivoted_net_qty, on='symbol')
    
    # Fill any NaN values with 0
    final_result = final_result.fillna(0)
    
    # Rename columns for clarity and consistency with the UI
    final_result = final_result.rename(columns={
        'symbol': 'Symbol',
        'expiry_date': 'ExpiryDate',
        'new_total': 'NewTotal',
        'total_buy_clients': 'BuyClients',
        'total_sell_clients': 'SellClients'
    })
    
    # Convert ExpiryDate to string format for better display
    if pd.api.types.is_datetime64_any_dtype(final_result['ExpiryDate']):
        final_result['ExpiryDate'] = final_result['ExpiryDate'].dt.strftime('%d-%m-%Y')
    
    return final_result

def main():
    # Header
    st.markdown("<div class='main-header'>Options Data Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-text'>This dashboard analyzes option data to show the nearest expiry metrics for each stock symbol.</div>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Trading Date Analysis", "Expiry Date Analysis"])
    
    # Sidebar information
    st.sidebar.markdown("## Data Source")
    data_source = st.sidebar.radio("Select Data Source", ["Database", "Upload CSV"])
    
    # Get data based on selected source
    df = None
    expiry_df = None  # For the second tab

    if data_source == "Database":
        try:
            # Get the most recent date from the database
            date_query = "SELECT MAX(date) as latest_date FROM options_data_hb"
            latest_date_df = load_data_from_postgres(date_query)
            
            if latest_date_df is not None and not latest_date_df.empty and latest_date_df['latest_date'].iloc[0] is not None:
                latest_date = latest_date_df['latest_date'].iloc[0]
                st.sidebar.info(f"Latest data date: {latest_date.strftime('%Y-%m-%d')}")
                
                # Get trading dates for first tab
                available_dates_query = "SELECT DISTINCT date FROM options_data_hb ORDER BY date DESC"
                dates_df = load_data_from_postgres(available_dates_query)
                
                # Get expiry dates for second tab
                expiry_dates_query = "SELECT DISTINCT expiry_date FROM options_data_hb ORDER BY expiry_date"
                expiry_dates_df = load_data_from_postgres(expiry_dates_query)
                
                if dates_df is not None and not dates_df.empty and expiry_dates_df is not None and not expiry_dates_df.empty:
                    # Process trading dates
                    if pd.api.types.is_datetime64_any_dtype(dates_df['date']):
                        available_dates = dates_df['date'].dt.strftime('%Y-%m-%d').tolist()
                    else:
                        dates_df['date'] = pd.to_datetime(dates_df['date'], errors='coerce')
                        available_dates = dates_df['date'].dt.strftime('%Y-%m-%d').tolist()
                    
                    # Process expiry dates
                    if pd.api.types.is_datetime64_any_dtype(expiry_dates_df['expiry_date']):
                        available_expiry_dates = expiry_dates_df['expiry_date'].dt.strftime('%Y-%m-%d').tolist()
                    else:
                        expiry_dates_df['expiry_date'] = pd.to_datetime(expiry_dates_df['expiry_date'], errors='coerce')
                        available_expiry_dates = expiry_dates_df['expiry_date'].dt.strftime('%Y-%m-%d').tolist()
                    
                    # Trading date selection for first tab
                    st.sidebar.markdown("### Select Trading Date")
                    st.sidebar.markdown("Choose a specific trading day to analyze options data for that day.")
                    selected_date = st.sidebar.selectbox("Trading Date", available_dates)
                    
                    # Expiry date selection for second tab
                    st.sidebar.markdown("### Select Expiry Date")
                    st.sidebar.markdown("Choose a specific expiry date to analyze options data across all trading days.")
                    selected_expiry = st.sidebar.selectbox("Expiry Date", available_expiry_dates)
                    
                    # Option to view all data
                    show_all = st.sidebar.checkbox("Show all data (warning: may be slow with large datasets)")
                    
                    if show_all:
                        query = "SELECT * FROM options_data_hb ORDER BY date DESC LIMIT 10000"
                        df = load_data_from_postgres(query)
                        if df is not None and not df.empty:
                            st.sidebar.success(f"Loaded {len(df)} records from all dates (limited to 10,000)")
                        else:
                            st.sidebar.error("No data found in the database")
                    else:
                        try:
                            # Load data for selected trading date (Tab 1)
                            parsed_date = pd.to_datetime(selected_date)
                            formatted_date = parsed_date.strftime('%Y-%m-%d')
                            
                            query = f"""
                            SELECT * FROM options_data_hb 
                            WHERE CAST(date AS DATE) = '{formatted_date}'
                            """
                            df = load_data_from_postgres(query)
                            
                            # Load data for selected expiry date (Tab 2)
                            parsed_expiry = pd.to_datetime(selected_expiry)
                            formatted_expiry = parsed_expiry.strftime('%Y-%m-%d')
                            
                            expiry_query = f"""
                            SELECT * FROM options_data_hb 
                            WHERE CAST(expiry_date AS DATE) = '{formatted_expiry}'
                            ORDER BY date
                            """
                            expiry_df = load_data_from_postgres(expiry_query)
                            
                            if df is not None and not df.empty:
                                st.sidebar.success(f"Loaded {len(df)} records for trading date {selected_date}")
                            else:
                                st.sidebar.warning("No data found for the selected trading date")
                                
                            if expiry_df is not None and not expiry_df.empty:
                                st.sidebar.success(f"Loaded {len(expiry_df)} records for expiry date {selected_expiry}")
                            else:
                                st.sidebar.warning("No data found for the selected expiry date")
                                
                        except Exception as e:
                            st.sidebar.error(f"Error loading data: {e}")
                else:
                    st.sidebar.error("Could not retrieve dates from database")
            else:
                st.sidebar.error("No data found in the database")
        except Exception as e:
            st.sidebar.error(f"Error loading data from database: {e}")
            
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your options data CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Check if the CSV uses camelCase or snake_case columns and standardize
                if 'Date' in df.columns and 'Symbol' in df.columns:  # CamelCase
                    # Rename columns to match snake_case
                    column_mapping = {
                        'Date': 'date',
                        'Symbol': 'symbol',
                        'ExpiryDate': 'expiry_date',
                        'StrikePrice': 'strike_price',
                        'OptType': 'opt_type',
                        'BtFrwdLongQty': 'bt_frwd_long_qty',
                        'BtFrwdShortQty': 'bt_frwd_short_qty',
                        'BtFrwdLongValue': 'bt_frwd_long_value',
                        'BtFrwdShortValue': 'bt_frwd_short_value',
                        'AvgBuyPriceCarryFwd': 'avg_buy_price_carry_fwd',
                        'AvgSellPriceCarryFwd': 'avg_sell_price_carry_fwd',
                        'NetQtyCarryFwd': 'net_qty_carry_fwd',
                        'TotalClients': 'total_clients',
                        'TotalBuyClients': 'total_buy_clients',
                        'TotalSellClients': 'total_sell_clients',
                        'NewTotal': 'new_total',
                        'BuyPercent': 'buy_percent',
                        'SellPercent': 'sell_percent',
                        'NetPercent': 'net_percent'
                    }
                    df.rename(columns={col: column_mapping.get(col, col) for col in df.columns}, inplace=True)
                
                # For CSV upload, also use the same data for expiry analysis
                expiry_df = df.copy()
                
                st.sidebar.success("Data loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading CSV: {e}")
                df = None
    
    # Process data if available
    if df is not None or expiry_df is not None:
        # TAB 1: TRADING DATE ANALYSIS
        with tab1:
            if df is not None:
                try:
                    # Process the data to get nearest expiry metrics
                    processed_df = filter_nearest_expiry_and_sum(df)
                    
                    # Symbol multi-select filter
                    symbols = sorted(processed_df['Symbol'].unique())
                    selected_symbols = st.multiselect("Select Symbols", symbols, default=[])
                    
                    # Filter data based on selection
                    if selected_symbols:
                        display_df = processed_df[processed_df['Symbol'].isin(selected_symbols)]
                    else:
                        display_df = processed_df  # Show all if nothing is selected
                    
                    # Dashboard layout
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("<div class='sub-header'>Nearest Expiry Options Data</div>", unsafe_allow_html=True)
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Download button for the processed data
                        csv_buffer = BytesIO()
                        display_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        st.download_button(
                            label="Download Processed Data as CSV",
                            data=csv_buffer,
                            file_name="options_nearest_expiry.csv",
                            mime="text/csv",
                        )
                    
                    with col2:
                        st.markdown("<div class='sub-header'>Summary Statistics</div>", unsafe_allow_html=True)
                        
                        # Total metrics
                        st.metric("Total Buy Clients", f"{display_df['BuyClients'].sum():,}")
                        st.metric("Total Sell Clients", f"{display_df['SellClients'].sum():,}")
                        st.metric("Total New Clients", f"{display_df['NewTotal'].sum():,}")
                    
                    # Visualizations
                    st.markdown("<div class='sub-header'>Data Visualization</div>", unsafe_allow_html=True)
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Call vs Put Net Quantity
                        fig_ce_pe = px.bar(
                            display_df,
                            x='Symbol',
                            y=['CE', 'PE'],
                            title='Call vs Put Net Quantity by Symbol',
                            barmode='group',
                            color_discrete_sequence=['#1E88E5', '#FFC107']
                        )
                        fig_ce_pe.update_layout(
                            xaxis_title='Symbol',
                            yaxis_title='Net Quantity',
                            legend_title='Option Type'
                        )
                        st.plotly_chart(fig_ce_pe, use_container_width=True)
                    
                    with viz_col2:
                        # Buy vs Sell Clients
                        fig_clients = px.bar(
                            display_df,
                            x='Symbol',
                            y=['BuyClients', 'SellClients'],
                            title='Buy vs Sell Clients by Symbol',
                            barmode='group',
                            color_discrete_sequence=['#4CAF50', '#F44336']
                        )
                        fig_clients.update_layout(
                            xaxis_title='Symbol',
                            yaxis_title='Number of Clients',
                            legend_title='Client Type'
                        )
                        st.plotly_chart(fig_clients, use_container_width=True)
                    
                    # Call-Put Ratio
                    display_df['Call-Put Ratio'] = display_df['CE'] / display_df['PE'].where(display_df['PE'] != 0, 1)
                    
                    fig_ratio = px.line(
                        display_df,
                        x='Symbol',
                        y='Call-Put Ratio',
                        title='Call-Put Ratio by Symbol',
                        markers=True,
                        line_shape='linear'
                    )
                    fig_ratio.update_layout(
                        xaxis_title='Symbol',
                        yaxis_title='Call-Put Ratio'
                    )
                    # Add a horizontal line at y=1 for reference
                    fig_ratio.add_shape(
                        type='line',
                        x0=0,
                        y0=1,
                        x1=len(display_df),
                        y1=1,
                        line=dict(color='red', width=1, dash='dash')
                    )
                    st.plotly_chart(fig_ratio, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing data for Trading Date tab: {e}")
                    st.info("Please check your data format. The dashboard expects columns like 'date', 'symbol', 'expiry_date', 'opt_type', 'net_qty_carry_fwd', etc.")
            else:
                st.info("No data available for Trading Date analysis. Please select a valid trading date.")
        
        # TAB 2: EXPIRY DATE ANALYSIS
        with tab2:
            if expiry_df is not None:
                try:
                    st.markdown("<div class='sub-header'>Options Data for Selected Expiry Date</div>", unsafe_allow_html=True)
                    st.markdown(f"Showing aggregated data for expiry date: **{selected_expiry if 'selected_expiry' in locals() else 'All Expiry Dates'}**")
                    
                    # Convert dates if they're not already datetime
                    if not pd.api.types.is_datetime64_any_dtype(expiry_df['date']):
                        expiry_df['date'] = pd.to_datetime(expiry_df['date'], errors='coerce')
                        
                    # Get unique symbols for filtering
                    symbols = sorted(expiry_df['symbol'].unique())
                    selected_symbols_expiry = st.multiselect("Select Symbols for Expiry Analysis", symbols, default=[], key="symbols_expiry")
                    
                    # Filter data based on symbol selection
                    if selected_symbols_expiry:
                        filtered_expiry_df = expiry_df[expiry_df['symbol'].isin(selected_symbols_expiry)]
                    else:
                        filtered_expiry_df = expiry_df  # Show all if nothing is selected
                    
                    # Calculate aggregated metrics for each symbol (summing across all dates)
                    symbol_metrics = filtered_expiry_df.groupby('symbol').agg({
                        'new_total': 'sum',
                        'total_buy_clients': 'sum',
                        'total_sell_clients': 'sum'
                    }).reset_index()
                    
                    # Calculate CE and PE values by symbol (summing across all dates)
                    valid_opt_types = ['CE', 'PE']
                    ce_pe_by_symbol = filtered_expiry_df[filtered_expiry_df['opt_type'].isin(valid_opt_types)]
                    ce_pe_metrics = ce_pe_by_symbol.groupby(['symbol', 'opt_type'])['net_qty_carry_fwd'].sum().reset_index()
                    
                    # Pivot to have CE and PE as columns
                    pivoted_ce_pe = ce_pe_metrics.pivot_table(index='symbol', columns='opt_type', values='net_qty_carry_fwd').reset_index()
                    
                    # Ensure CE and PE columns exist
                    for col in ['CE', 'PE']:
                        if col not in pivoted_ce_pe.columns:
                            pivoted_ce_pe[col] = 0
                    
                    # Merge the metrics
                    merged_metrics = pd.merge(symbol_metrics, pivoted_ce_pe, on='symbol')
                    
                    # Calculate Call-Put Ratio
                    merged_metrics['Call-Put Ratio'] = merged_metrics['CE'] / merged_metrics['PE'].where(merged_metrics['PE'] != 0, 1)
                    
                    # Rename columns for clarity
                    merged_metrics = merged_metrics.rename(columns={
                        'symbol': 'Symbol',
                        'new_total': 'Total New Clients',
                        'total_buy_clients': 'Total Buy Clients',
                        'total_sell_clients': 'Total Sell Clients'
                    })
                    
                    # Show aggregated data table
                    st.markdown("<div class='sub-header'>Aggregated Data by Symbol</div>", unsafe_allow_html=True)
                    st.dataframe(merged_metrics, use_container_width=True)
                    
                    # Download button for the aggregated data
                    csv_buffer = BytesIO()
                    merged_metrics.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    st.download_button(
                        label="Download Aggregated Data as CSV",
                        data=csv_buffer,
                        file_name="options_expiry_aggregated.csv",
                        mime="text/csv",
                    )
                    
                    # Visualizations - Individual charts for each metric
                    st.markdown("<div class='sub-header'>Individual Visualizations</div>", unsafe_allow_html=True)
                    
                    # CE Values Chart
                    fig_ce = px.bar(
                        merged_metrics,
                        x='Symbol',
                        y='CE',
                        title='CE (Call) Net Quantity by Symbol',
                        color_discrete_sequence=['#1E88E5']
                    )
                    fig_ce.update_layout(
                        xaxis_title='Symbol',
                        yaxis_title='CE Net Quantity'
                    )
                    st.plotly_chart(fig_ce, use_container_width=True)
                    
                    # PE Values Chart
                    fig_pe = px.bar(
                        merged_metrics,
                        x='Symbol',
                        y='PE',
                        title='PE (Put) Net Quantity by Symbol',
                        color_discrete_sequence=['#FFC107']
                    )
                    fig_pe.update_layout(
                        xaxis_title='Symbol',
                        yaxis_title='PE Net Quantity'
                    )
                    st.plotly_chart(fig_pe, use_container_width=True)
                    
                    # Buy Clients Chart
                    fig_buy = px.bar(
                        merged_metrics,
                        x='Symbol',
                        y='Total Buy Clients',
                        title='Total Buy Clients by Symbol',
                        color_discrete_sequence=['#4CAF50']
                    )
                    fig_buy.update_layout(
                        xaxis_title='Symbol',
                        yaxis_title='Number of Buy Clients'
                    )
                    st.plotly_chart(fig_buy, use_container_width=True)
                    
                    # Sell Clients Chart
                    fig_sell = px.bar(
                        merged_metrics,
                        x='Symbol',
                        y='Total Sell Clients',
                        title='Total Sell Clients by Symbol',
                        color_discrete_sequence=['#F44336']
                    )
                    fig_sell.update_layout(
                        xaxis_title='Symbol',
                        yaxis_title='Number of Sell Clients'
                    )
                    st.plotly_chart(fig_sell, use_container_width=True)
                    
                    # Call-Put Ratio Chart
                    fig_ratio = px.line(
                        merged_metrics,
                        x='Symbol',
                        y='Call-Put Ratio',
                        title='Call-Put Ratio by Symbol',
                        markers=True,
                        line_shape='linear'
                    )
                    fig_ratio.update_layout(
                        xaxis_title='Symbol',
                        yaxis_title='Call-Put Ratio'
                    )
                    # Add a horizontal line at y=1 for reference
                    fig_ratio.add_shape(
                        type='line',
                        x0=0,
                        y0=1,
                        x1=len(merged_metrics),
                        y1=1,
                        line=dict(color='red', width=1, dash='dash')
                    )
                    st.plotly_chart(fig_ratio, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing data for Expiry Date tab: {e}")
                    st.info("Please check your data format. The dashboard expects columns like 'date', 'symbol', 'expiry_date', 'opt_type', 'net_qty_carry_fwd', etc.")
            else:
                st.info("No data available for Expiry Date analysis. Please select a valid expiry date.")
    
    else:
        st.info("Please connect to a database or upload a CSV file to get started.")
    
    # Add upload to database option
    if data_source == "Upload CSV" and df is not None:
        if st.sidebar.button("Upload to Database"):
            try:
                engine = init_connection()
                if engine:
                    # Upload to the options_data_hb table
                    df.to_sql(
                        'options_data_hb',
                        engine,
                        if_exists='append',
                        index=False,
                        chunksize=1000
                    )
                    st.sidebar.success(f"Successfully uploaded {len(df)} rows to options_data_hb table")
                else:
                    st.sidebar.error("Database connection not available")
            except Exception as e:
                st.sidebar.error(f"Error uploading to database: {e}")

if __name__ == "__main__":
    main()