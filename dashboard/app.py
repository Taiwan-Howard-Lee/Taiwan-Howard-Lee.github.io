"""
Apple ML Trading Dashboard - Main Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    from data_collection.apple_collector import AppleDataCollector
    from feature_engineering.technical_indicators import TechnicalIndicators
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure all required modules are installed and the project structure is correct.")


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="🍎 Apple ML Trading Dashboard",
        page_icon="🍎",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def create_sidebar():
    """Create sidebar with navigation and controls."""
    st.sidebar.title("🍎 Apple ML Trading")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Overview", "📈 Performance", "🤖 Models", "⚠️ Risk", "🌍 Market"]
    )
    
    st.sidebar.markdown("---")
    
    # Date range selector
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Model selector
    selected_model = st.sidebar.selectbox(
        "Select Model",
        ["Ensemble", "XGBoost", "Random Forest", "LSTM", "Transformer"]
    )
    
    # Risk tolerance
    risk_level = st.sidebar.slider(
        "Risk Tolerance",
        min_value=0.1, max_value=2.0, value=1.0, step=0.1
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    return page, start_date, end_date, selected_model, risk_level


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_apple_data(period="1y"):
    """Load Apple stock data with caching."""
    try:
        collector = AppleDataCollector()
        data = collector.fetch_daily_data(period=period)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data(ttl=300)
def calculate_technical_indicators(data):
    """Calculate technical indicators with caching."""
    try:
        if data is not None and not data.empty:
            indicators = TechnicalIndicators(data)
            return indicators.calculate_all_indicators()
        return None
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return None


def create_price_chart(data, indicators=None):
    """Create interactive price chart with technical indicators."""
    if data is None or data.empty:
        return None
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Moving Averages', 'Volume', 'RSI'),
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        shared_xaxes=True
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="AAPL Price"
        ), row=1, col=1
    )
    
    # Add moving averages if indicators available
    if indicators is not None:
        if 'SMA_20' in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ), row=1, col=1
            )
        
        if 'SMA_50' in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ), row=1, col=1
            )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ), row=2, col=1
    )
    
    # RSI chart
    if indicators is not None and 'RSI_14' in indicators.columns:
        fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators['RSI_14'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ), row=3, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title="Apple (AAPL) Stock Analysis",
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    return fig


def display_metrics_cards(data):
    """Display key performance metrics in cards."""
    if data is None or data.empty:
        return
    
    # Calculate metrics
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
    daily_change = current_price - prev_price
    daily_change_pct = (daily_change / prev_price) * 100
    
    # Calculate additional metrics
    high_52w = data['High'].rolling(252).max().iloc[-1]
    low_52w = data['Low'].rolling(252).min().iloc[-1]
    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
    current_volume = data['Volume'].iloc[-1]
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=f"{daily_change_pct:.2f}%"
        )
    
    with col2:
        st.metric(
            label="52W High",
            value=f"${high_52w:.2f}",
            delta=f"{((current_price - high_52w) / high_52w * 100):.1f}%"
        )
    
    with col3:
        st.metric(
            label="52W Low",
            value=f"${low_52w:.2f}",
            delta=f"{((current_price - low_52w) / low_52w * 100):.1f}%"
        )
    
    with col4:
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        st.metric(
            label="Volume Ratio",
            value=f"{volume_ratio:.2f}x",
            delta=f"{((volume_ratio - 1) * 100):.1f}%"
        )


def show_overview_page(data, indicators):
    """Display the overview page."""
    st.title("📊 Portfolio Overview")
    
    if data is None:
        st.error("Unable to load data. Please check your connection and try again.")
        return
    
    # Display metrics cards
    display_metrics_cards(data)
    
    st.markdown("---")
    
    # Main price chart
    fig = create_price_chart(data, indicators)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent data table
    st.subheader("Recent Price Data")
    st.dataframe(data.tail(10), use_container_width=True)


def show_performance_page():
    """Display the performance analysis page."""
    st.title("📈 Performance Analytics")
    st.info("Performance analytics will be implemented when backtesting system is ready.")
    
    # Placeholder for performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Performance")
        st.write("• Total Return: Coming soon")
        st.write("• Sharpe Ratio: Coming soon")
        st.write("• Maximum Drawdown: Coming soon")
    
    with col2:
        st.subheader("Risk Metrics")
        st.write("• Value at Risk: Coming soon")
        st.write("• Expected Shortfall: Coming soon")
        st.write("• Volatility: Coming soon")


def show_models_page():
    """Display the model insights page."""
    st.title("🤖 Model Insights")
    st.info("Model insights will be available when ML models are trained.")
    
    # Placeholder for model information
    st.subheader("Model Performance")
    st.write("• Model accuracy tracking")
    st.write("• Feature importance analysis")
    st.write("• Prediction confidence scores")


def show_risk_page():
    """Display the risk analysis page."""
    st.title("⚠️ Risk Analysis")
    st.info("Risk analysis will be implemented with the risk management system.")
    
    # Placeholder for risk metrics
    st.subheader("Risk Metrics")
    st.write("• VaR and CVaR calculations")
    st.write("• Tail risk analysis")
    st.write("• Drawdown analysis")


def show_market_page():
    """Display the market context page."""
    st.title("🌍 Market Context")
    st.info("Market context analysis will be available when regime detection is implemented.")
    
    # Placeholder for market analysis
    st.subheader("Market Regime")
    st.write("• Current market state")
    st.write("• Volatility clustering")
    st.write("• Economic indicators")


def main():
    """Main application function."""
    setup_page_config()
    
    # Create sidebar and get user inputs
    page, start_date, end_date, selected_model, risk_level = create_sidebar()
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_apple_data(period="1y")
        indicators = calculate_technical_indicators(data) if data is not None else None
    
    # Route to different pages
    if page == "📊 Overview":
        show_overview_page(data, indicators)
    elif page == "📈 Performance":
        show_performance_page()
    elif page == "🤖 Models":
        show_models_page()
    elif page == "⚠️ Risk":
        show_risk_page()
    elif page == "🌍 Market":
        show_market_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🍎 **Apple ML Trading System** | Built with Streamlit | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
