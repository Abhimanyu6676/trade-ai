# Restrict the maximum amount of virtual memory available to this shell to 2.5 Gigabytes, run below command first every time we login to new venv session
# ulimit -v 2500000
# OPENALGO_HOST = "https://openalgo-1.sternetsmart.com"
# OPENALGO_API_KEY = "888c8a1c731f0847cda2b4c5a8b449e248b1305fdb304baad2e0ec7b4575dde8"

import gradio as ui

from trade_ai.ai_parser import extract_market_intent
from trade_ai.data_engine import fetch_and_clean_candles, get_openalgo_status_message


def controller_pipeline(user_query: str, interval: str, exchange: str):
    """Orchestrates requests across NLP parsing, data processing, OpenAlgo system tools and prediction neural networks."""
    if not user_query.strip():
        return "Please input a text query.", "N/A", "N/A", "N/A", "0.0%", None

    # 1. Ask local text AI what the user wants
    action, ticker, days_back = extract_market_intent(user_query)

    # Handle OpenAlgo Analyzer Status Intent Request
    if action == "openalgo_status":
        status_msg = get_openalgo_status_message()
        return status_msg, "OPENALGO", exchange, "SYSTEM_STATUS", "100.0%", None

    # 2. Run data engine fetch + forward calculation pass through your custom PyTorch model
    df, ai_signal, confidence = fetch_and_clean_candles(
        ticker, days_back, interval, exchange
    )

    if df is None:
        return (
            f"❌ Pipeline Failure for {ticker}.",
            ticker,
            exchange,
            "Error State",
            "0.0%",
            None,
        )

    status_msg = f"⚡ **[Full MVC Stack Run Succeeded]** Processed lookup range: `{days_back} days`"
    confidence_string = f"{confidence}%"

    return status_msg, ticker, exchange, ai_signal, confidence_string, df


# =====================================================================
# GRADIO VISUAL INTERFACE ASSEMBLY
# =====================================================================
with ui.Blocks(title="Proprietary AI Quant Suite") as demo:
    ui.Markdown("# 🤖 Proprietary Custom PyTorch AI Trading Platform")
    ui.Markdown(
        "Decoupled MVC Pipeline: NLP intent interpretation, list-based feature math arrays, and native PyTorch GRU Neural Network prediction brains."
    )

    with ui.Row():
        with ui.Column(scale=2):
            user_text = ui.Textbox(
                label="Conversational Prompt Command Request Input",
                placeholder="Try: 'Check history for SBIN for 30 days' or 'Check openalgo status'...",
            )
        with ui.Column(scale=1):
            interval_select = ui.Dropdown(
                choices=["1m", "3m", "5m", "15m", "30m", "1h", "D"],
                value="D",
                label="Interval Timeframe",
            )
            exchange_select = ui.Dropdown(
                choices=["NSE", "BSE", "NFO", "MCX"],
                value="NSE",
                label="Target Exchange",
            )

    submit_btn = ui.Button("Execute Full Stack Process Pipeline", variant="primary")

    status_display = ui.Markdown()

    # 🏛️ VISUAL METRIC ANCHORS: Real-Time Custom AI Output Monitoring Blocks
    with ui.Row():
        active_stock = ui.Textbox(label="Active Stock Target", interactive=False)
        active_exchange = ui.Textbox(label="Active Exchange Matrix", interactive=False)

    with ui.Row():
        model_signal = ui.Textbox(
            label="🧠 Custom PyTorch AI Forecast Signal", interactive=False
        )
        model_confidence = ui.Textbox(
            label="📊 Neural Classification Direction Confidence", interactive=False
        )

    data_grid_view = ui.DataFrame()

    submit_btn.click(
        fn=controller_pipeline,
        inputs=[user_text, interval_select, exchange_select],
        outputs=[
            status_display,
            active_stock,
            active_exchange,
            model_signal,
            model_confidence,
            data_grid_view,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
