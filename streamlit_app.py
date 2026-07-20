"""
PART 2: LLMOps (Model Evaluation, Monitoring, and User Interface Creation Using Streamlit)

7.2 Data Visualization: Create various plots and visualizations to present insights
7.3 Streamlit UI: Develop an intuitive user interface using Streamlit
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Add parent directory to path to import from corporate metrics insight
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("corporate_metrics_insight", "./corporate_metrics_insight.py")
capstone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capstone_module)
test_agent = capstone_module.test_agent
df = capstone_module.df

# Import evaluation module
from evaluation import run_evaluation, format_evaluation_report, TEST_QUERIES

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Corp Metrics Insights", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# 7.3 STREAMLIT UI: Initialize Session State
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit_session_{datetime.now().timestamp()}"

# ============================================================================
# 7.3 STREAMLIT UI: Header
# ============================================================================
st.title("🔍 Corp Metrics Insights - AI-Powered Business Intelligence Assistant")
st.markdown("**Analyze your data with AI-driven insights and interactive visualizations**")

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.header("📊 Navigation")
    page = st.radio("Select Page:", ["Dashboard", "Chat with AI", "Model Evaluation", "About"])

# ============================================================================
# PAGE 1: DASHBOARD WITH VISUALIZATIONS
# ============================================================================
if page == "Dashboard":
    st.header("📈 Business Intelligence Dashboard")

    col1, col2 = st.columns(2)

    # ========================================================================
    # 7.2 DATA VISUALIZATION: Sales Trends Over Time
    # ========================================================================
    with col1:
        st.subheader("📅 Sales Trends Over Time")
        df["Date"] = pd.to_datetime(df["Date"])
        daily_sales = df.groupby("Date")["Sales"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_sales["Date"], daily_sales["Sales"], linewidth=2, color="#1f77b4")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Sales ($)", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=True)

    # ========================================================================
    # 7.2 DATA VISUALIZATION: Product Performance Comparisons
    # ========================================================================
    with col2:
        st.subheader("🏆 Product Performance")
        product_sales = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        bars = ax.bar(product_sales.index, product_sales.values, color=colors)
        ax.set_ylabel("Total Sales ($)", fontsize=10)
        ax.set_xlabel("Product", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:,.0f}',
                   ha='center', va='bottom', fontsize=9)

        st.pyplot(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # ========================================================================
    # 7.2 DATA VISUALIZATION: Regional Analysis
    # ========================================================================
    with col3:
        st.subheader("🌍 Regional Analysis")
        regional_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_region = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        wedges, texts, autotexts = ax.pie(regional_sales.values,
                                           labels=regional_sales.index,
                                           autopct='%1.1f%%',
                                           colors=colors_region,
                                           startangle=90)

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')

        st.pyplot(fig, use_container_width=True)

    # ========================================================================
    # 7.2 DATA VISUALIZATION: Customer Demographics and Segmentation
    # ========================================================================
    with col4:
        st.subheader("👥 Customer Satisfaction by Product")
        satisfaction_by_product = df.groupby("Product")["Customer_Satisfaction"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_satisfaction = ["#2ca02c" if x > 3.0 else "#ff7f0e" if x > 2.8 else "#d62728"
                              for x in satisfaction_by_product.values]
        bars = ax.barh(satisfaction_by_product.index, satisfaction_by_product.values, color=colors_satisfaction)
        ax.set_xlabel("Average Satisfaction Score", fontsize=10)
        ax.set_xlim(0, 5)
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.2f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')

        st.pyplot(fig, use_container_width=True)

    # ========================================================================
    # 7.2 DATA VISUALIZATION: Key Metrics
    # ========================================================================
    st.subheader("📊 Key Metrics")
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        total_sales = df["Sales"].sum()
        st.metric("Total Sales", f"${total_sales:,.0f}")

    with col6:
        avg_satisfaction = df["Customer_Satisfaction"].mean()
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}/5.0")

    with col7:
        num_transactions = len(df)
        st.metric("Total Transactions", f"{num_transactions:,}")

    with col8:
        avg_age = df["Customer_Age"].mean()
        st.metric("Avg Customer Age", f"{avg_age:.1f} years")

# ============================================================================
# PAGE 2: CHAT WITH AI ASSISTANT
# ============================================================================
elif page == "Chat with AI":
    st.header("💬 Chat with Corp Metrics Insights AI Assistant")

    st.markdown("""
    Ask me questions about your business data! I can:
    - Analyze sales trends and patterns
    - Segment customers by demographics
    - Identify underperforming products
    - Provide strategic recommendations based on business documents
    """)

    # ====================================================================
    # 7.3 STREAMLIT UI: Example Queries
    # ====================================================================
    st.subheader("📝 Example Queries (Click to Try)")

    example_queries = {
        "CSV + PDF Analysis": "Calculate the average 'Customer_Satisfaction' score per 'Product' category. Then, search the PDF research documents to identify management strategies for improving customer satisfaction in product lines that are currently underperforming the company average.",
        "Product Performance": "Which products have the lowest sales and customer satisfaction? What does the business literature say about improving underperforming products?",
        "Regional Insights": "Compare sales and satisfaction across regions. Search the PDFs for regional business strategies.",
        "Customer Segmentation": "Analyze customer age groups and their satisfaction levels. What strategies from our business documents can help improve satisfaction?",
    }

    col_examples = st.columns(2)
    for idx, (title, query) in enumerate(example_queries.items()):
        with col_examples[idx % 2]:
            if st.button(f"📌 {title}", use_container_width=True):
                st.session_state.example_query = query
                st.rerun()

    # ====================================================================
    # 7.3 STREAMLIT UI: Chat History Display
    # ====================================================================
    st.subheader("Conversation History")
    chat_container = st.container()

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

    # ====================================================================
    # 7.3 STREAMLIT UI: User Input
    # ====================================================================
    st.subheader("Ask a Question")

    # Pre-fill with example query if one was selected
    default_query = st.session_state.get("example_query", "")
    user_input = st.text_area("Your question:", height=100, value=default_query,
                              placeholder="e.g., Which product has the lowest customer satisfaction? What strategies can improve it?")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🚀 Send Query", use_container_width=True):
            if user_input.strip():
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": user_input})

                # Clear example query after submission
                if "example_query" in st.session_state:
                    del st.session_state.example_query

                # Show loading spinner
                with st.spinner("🤔 Corp Metrics Insights is analyzing..."):
                    try:
                        # Call the agent
                        response = test_agent(query=user_input, thread_id=st.session_state.thread_id)

                        # Extract response text
                        assistant_response = response["messages"][-1].content

                        # Add assistant message to history
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                        # Rerun to display new message
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = f"streamlit_session_{datetime.now().timestamp()}"
            st.rerun()

# ============================================================================
# PAGE 3: MODEL EVALUATION
# ============================================================================
elif page == "Model Evaluation":
    st.header("🔍 Model Evaluation Report")

    st.markdown("""
    This section evaluates the Corp Metrics Insights AI agent's performance across multiple dimensions:
    - **Relevance**: Does it answer the question?
    - **Accuracy**: Is the information correct?
    - **Completeness**: Does it cover all aspects?
    - **Tool Usage**: Did it use the right tool(s)?
    """)

    col1, col2 = st.columns([2, 1])

    with col2:
        if st.button("🚀 Run Evaluation", use_container_width=True):
            st.session_state.running_eval = True

    if st.session_state.get("running_eval", False):
        st.write("### ⏳ Running Evaluation in Progress...")

        # Create placeholder for streaming results
        progress_placeholder = st.empty()
        results_placeholder = st.empty()

        try:
            results = []
            total_tests = 10

            for i, test_case in enumerate(TEST_QUERIES, 1):
                # Update progress
                progress_text = f"**Test {i}/{total_tests}**: {test_case['description']}"
                progress_placeholder.write(progress_text)

                try:
                    # Run the agent
                    response = test_agent(query=test_case["query"], thread_id=f"eval_session_{test_case['id']}")
                    response_text = response["messages"][-1].content

                    # Score the response
                    from evaluation import score_response
                    scores = score_response(
                        query=test_case["query"],
                        response=response_text,
                        expected_tool=test_case["expected_tool"],
                        query_id=test_case["id"]
                    )

                    result = {
                        "Test ID": test_case["id"],
                        "Category": test_case["category"],
                        "Expected Tool": test_case["expected_tool"],
                        "Relevance": scores["Relevance"],
                        "Accuracy": scores["Accuracy"],
                        "Completeness": scores["Completeness"],
                        "Tool Usage": scores["Tool Usage"],
                        "Average Score": scores["Average"],
                        "Status": "✅ Pass" if scores["Average"] >= 4.0 else "⚠️ Warning" if scores["Average"] >= 3.0 else "❌ Fail"
                    }
                    results.append(result)

                    # Stream results table
                    results_df = pd.DataFrame(results)
                    with results_placeholder.container():
                        st.dataframe(results_df[["Test ID", "Category", "Expected Tool", "Average Score", "Status"]], use_container_width=True)

                except Exception as e:
                    print(f"Test {i} error: {str(e)}")
                    result = {
                        "Test ID": test_case["id"],
                        "Category": test_case["category"],
                        "Expected Tool": test_case["expected_tool"],
                        "Relevance": 1,
                        "Accuracy": 1,
                        "Completeness": 1,
                        "Tool Usage": 1,
                        "Average Score": 1.0,
                        "Status": "❌ Error"
                    }
                    results.append(result)

            # Final report
            results_df = pd.DataFrame(results)
            report = {
                "results": results_df,
                "summary": {
                    "total_tests": len(results),
                    "passed": len([r for r in results if r["Average Score"] >= 4.0]),
                    "warnings": len([r for r in results if 3.0 <= r["Average Score"] < 4.0]),
                    "failed": len([r for r in results if r["Average Score"] < 3.0]),
                    "pass_rate": round((len([r for r in results if r["Average Score"] >= 4.0]) / len(results)) * 100, 1),
                    "overall_score": round(results_df["Average Score"].mean(), 2)
                },
                "metrics_by_tool": results_df.groupby("Expected Tool")["Average Score"].mean().to_dict(),
                "metrics_by_category": results_df.groupby("Category")["Average Score"].mean().to_dict(),
                "metrics_by_dimension": {
                    "Relevance": round(results_df["Relevance"].mean(), 2),
                    "Accuracy": round(results_df["Accuracy"].mean(), 2),
                    "Completeness": round(results_df["Completeness"].mean(), 2),
                    "Tool Usage": round(results_df["Tool Usage"].mean(), 2)
                }
            }

            st.session_state.eval_report = report
            st.session_state.running_eval = False
            progress_placeholder.success("✅ Evaluation Complete!")

        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.session_state.running_eval = False

    # Display results if available
    if "eval_report" in st.session_state:
        report = st.session_state.eval_report
        summary = report["summary"]
        results_df = report["results"]

        # Display summary metrics
        st.subheader("Summary")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Pass Rate", f"{summary['pass_rate']}%")
        with col2:
            st.metric("Overall Score", f"{summary['overall_score']}/5.0")
        with col3:
            st.metric("Passed", summary['passed'])
        with col4:
            st.metric("Warnings", summary['warnings'])
        with col5:
            st.metric("Failed", summary['failed'])

        st.divider()

        # Display metrics by dimension
        st.subheader("Performance by Dimension")
        dim_cols = st.columns(4)
        dims = ["Relevance", "Accuracy", "Completeness", "Tool Usage"]

        for i, dim in enumerate(dims):
            with dim_cols[i]:
                score = report['metrics_by_dimension'][dim]
                st.metric(dim, f"{score}/5.0")

        st.divider()

        # Display metrics by tool
        st.subheader("Performance by Tool Type")
        tool_data = pd.DataFrame({
            "Tool Type": list(report['metrics_by_tool'].keys()),
            "Average Score": list(report['metrics_by_tool'].values())
        })

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2ca02c" if x >= 4.0 else "#ff7f0e" if x >= 3.0 else "#d62728"
                 for x in tool_data["Average Score"]]
        bars = ax.barh(tool_data["Tool Type"], tool_data["Average Score"], color=colors)
        ax.set_xlim(0, 5)
        ax.set_xlabel("Average Score", fontsize=10)
        ax.grid(True, alpha=0.3, axis="x")

        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.2f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')

        st.pyplot(fig, use_container_width=True)

        st.divider()

        # Display detailed results table
        st.subheader("Detailed Test Results")
        results_display = report["results"][["Test ID", "Category", "Expected Tool",
                                            "Relevance", "Accuracy", "Completeness",
                                            "Tool Usage", "Average Score", "Status"]]
        st.dataframe(results_display, use_container_width=True)

        st.divider()

        # Display formatted report
        st.subheader("Text Report")
        report_text = format_evaluation_report(report)
        st.text(report_text)

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================
elif page == "About":
    st.header("About Corp Metrics Insights")

    st.markdown("""
    ### 🎯 Mission
    Corp Metrics Insights is an AI-powered Business Intelligence Assistant designed to help organizations
    transform their data into actionable insights.

    ### 🛠️ Technology Stack
    - **LLM**: GPT-4o (OpenAI)
    - **Framework**: LangChain + LanGraph
    - **RAG System**: FAISS Vector Database
    - **UI**: Streamlit
    - **Data Processing**: Pandas

    ### 📊 Features
    - **Advanced Data Analysis**: Pandas-powered statistical analysis
    - **RAG Integration**: Retrieve insights from business documents
    - **Memory Management**: Conversation context retention
    - **Interactive Visualizations**: Real-time business dashboards
    - **Chat Interface**: Natural language querying

    ### 📈 Capabilities
    1. Sales performance analysis by time period
    2. Product and regional performance analysis
    3. Customer segmentation by demographics
    4. Statistical calculations (median, standard deviation, etc.)
    5. Strategic recommendations based on business documents

    ### 👨‍💻 Built for
    Small to medium-sized enterprises (SMEs) that want to leverage AI for business intelligence
    without investing in expensive BI tools.
    """)

    st.divider()
    st.markdown("**Version**: 1.0 | **Last Updated**: July 2026")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Corp Metrics Insights © 2026 | AI-Powered Business Intelligence
</div>
""", unsafe_allow_html=True)
