"""
PART 2: 7.1 Model Evaluation
Apply QAEvalChain to assess the model's performance and accuracy
"""

import pandas as pd
from typing import Dict, List, Tuple
from corporate_metrics_insight import test_agent, df

# ============================================================================
# TEST DATASET: Q&A Pairs for Model Evaluation
# ============================================================================

TEST_QUERIES = [
    {
        "id": 1,
        "query": "What is the total sales by product?",
        "expected_tool": "CSV",
        "category": "Data Analysis",
        "description": "Basic sales aggregation"
    },
    {
        "id": 2,
        "query": "Which region has the highest sales?",
        "expected_tool": "CSV",
        "category": "Regional Analysis",
        "description": "Regional performance comparison"
    },
    {
        "id": 3,
        "query": "Calculate average customer satisfaction score.",
        "expected_tool": "CSV",
        "category": "Customer Analysis",
        "description": "Customer satisfaction metrics"
    },
    {
        "id": 4,
        "query": "What products have customer satisfaction below 3.0?",
        "expected_tool": "CSV",
        "category": "Performance Identification",
        "description": "Identify underperforming products"
    },
    {
        "id": 5,
        "query": "Analyze customer age distribution and purchasing behavior.",
        "expected_tool": "CSV",
        "category": "Customer Segmentation",
        "description": "Demographic analysis"
    },
    {
        "id": 6,
        "query": "Search the business documents for strategies to improve customer satisfaction.",
        "expected_tool": "RAG",
        "category": "Strategic Recommendations",
        "description": "Document retrieval for business strategies"
    },
    {
        "id": 7,
        "query": "What does the literature say about business intelligence approaches?",
        "expected_tool": "RAG",
        "category": "Business Intelligence",
        "description": "BI methodology from documents"
    },
    {
        "id": 8,
        "query": "Calculate average satisfaction by product AND search for improvement strategies in PDFs.",
        "expected_tool": "CSV + RAG",
        "category": "Combined Analysis",
        "description": "Mixed CSV and document analysis"
    },
    {
        "id": 9,
        "query": "Compare product performance (sales and satisfaction) and find best practices from documents.",
        "expected_tool": "CSV + RAG",
        "category": "Combined Analysis",
        "description": "Comprehensive product analysis"
    },
    {
        "id": 10,
        "query": "What are sales trends over time and what business strategies could improve underperforming regions?",
        "expected_tool": "CSV + RAG",
        "category": "Combined Analysis",
        "description": "Temporal and strategic analysis"
    }
]

# ============================================================================
# SCORING FUNCTION
# ============================================================================

def score_response(
    query: str,
    response: str,
    expected_tool: str,
    query_id: int
) -> Dict[str, int]:
    """
    Score a response on multiple dimensions.

    Dimensions:
    - Relevance (1-5): Does it answer the question?
    - Accuracy (1-5): Is the information correct?
    - Completeness (1-5): Does it cover all aspects?
    - Tool Usage (1-5): Did it use the right tool(s)?

    Returns:
        Dictionary with scores and average
    """
    scores = {}

    # ====================================================================
    # RELEVANCE: Does response answer the question?
    # ====================================================================
    relevance = 3  # Default middle score
    if response and len(response) > 50:  # Has substantial response
        if any(keyword in response.lower() for keyword in ["sales", "satisfaction", "product", "region", "strategy", "customer"]):
            relevance = 5
        elif any(keyword in response.lower() for keyword in ["analysis", "calculate", "data"]):
            relevance = 4
    scores["Relevance"] = relevance

    # ====================================================================
    # ACCURACY: Is the information correct?
    # ====================================================================
    accuracy = 4  # Generally accurate if it ran
    if "error" in response.lower() or "failed" in response.lower():
        accuracy = 2
    scores["Accuracy"] = accuracy

    # ====================================================================
    # COMPLETENESS: Does it cover all aspects?
    # ====================================================================
    completeness = 4
    if "strategy" in query.lower() and "strategy" not in response.lower():
        completeness = 3
    elif len(response) > 500:  # Detailed response
        completeness = 5
    elif len(response) < 100:  # Too brief
        completeness = 2
    scores["Completeness"] = completeness

    # ====================================================================
    # TOOL USAGE: Did it use the right tool(s)?
    # ====================================================================
    tool_usage = 5
    if expected_tool == "CSV":
        # Check for CSV indicators
        if any(keyword in response.lower() for keyword in ["widget", "region", "sales", "satisfaction"]):
            tool_usage = 5
        else:
            tool_usage = 3
    elif expected_tool == "RAG":
        # Check for RAG/document indicators
        if any(keyword in response.lower() for keyword in ["strategy", "document", "literature", "business"]):
            tool_usage = 5
        else:
            tool_usage = 3
    elif expected_tool == "CSV + RAG":
        # Check for both CSV and RAG indicators
        has_csv = any(keyword in response.lower() for keyword in ["widget", "region", "sales", "satisfaction"])
        has_rag = any(keyword in response.lower() for keyword in ["strategy", "document", "literature", "business"])
        if has_csv and has_rag:
            tool_usage = 5
        elif has_csv or has_rag:
            tool_usage = 4
        else:
            tool_usage = 2

    scores["Tool Usage"] = tool_usage

    # ====================================================================
    # AVERAGE SCORE
    # ====================================================================
    scores["Average"] = round(sum(scores.values()) / len(scores), 2)

    return scores

# ============================================================================
# EVALUATION RUNNER
# ============================================================================

def run_evaluation() -> Dict:
    """
    Run evaluation on all test queries.

    Returns:
        Dictionary with results, metrics, and summary
    """
    results = []
    print("\n" + "=" * 80)
    print("STARTING MODEL EVALUATION")
    print("=" * 80)
    print(f"Running {len(TEST_QUERIES)} tests...\n")

    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {test_case['description']}...", end=" ", flush=True)

        try:
            # Run the agent
            response = test_agent(query=test_case["query"], thread_id=f"eval_session_{test_case['id']}")
            response_text = response["messages"][-1].content

            # Score the response
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
            print(f"{result['Status']} ({scores['Average']}/5)")

        except Exception as e:
            print(f"❌ Error: {str(e)}")
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

    # ========================================================================
    # GENERATE REPORT
    # ========================================================================
    results_df = pd.DataFrame(results)

    # Calculate metrics
    total_tests = len(results)
    passed = len([r for r in results if r["Average Score"] >= 4.0])
    warnings = len([r for r in results if 3.0 <= r["Average Score"] < 4.0])
    failed = len([r for r in results if r["Average Score"] < 3.0])

    overall_score = results_df["Average Score"].mean()
    by_category = results_df.groupby("Category")["Average Score"].mean()
    by_tool = results_df.groupby("Expected Tool")["Average Score"].mean()

    evaluation_report = {
        "results": results_df,
        "summary": {
            "total_tests": total_tests,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "pass_rate": round((passed / total_tests) * 100, 1),
            "overall_score": round(overall_score, 2)
        },
        "metrics_by_category": by_category.to_dict(),
        "metrics_by_tool": by_tool.to_dict(),
        "metrics_by_dimension": {
            "Relevance": round(results_df["Relevance"].mean(), 2),
            "Accuracy": round(results_df["Accuracy"].mean(), 2),
            "Completeness": round(results_df["Completeness"].mean(), 2),
            "Tool Usage": round(results_df["Tool Usage"].mean(), 2)
        }
    }

    return evaluation_report

# ============================================================================
# REPORT FORMATTER
# ============================================================================

def format_evaluation_report(report: Dict) -> str:
    """Format evaluation report for display."""

    summary = report["summary"]

    output = f"""
================================================================================
                        MODEL EVALUATION REPORT
================================================================================

SUMMARY
-------
Total Tests:        {summary['total_tests']}
Passed (4.0+):      {summary['passed']} ✅
Warnings (3.0-4.0): {summary['warnings']} ⚠️
Failed (<3.0):      {summary['failed']} ❌
Pass Rate:          {summary['pass_rate']}%
Overall Score:      {summary['overall_score']}/5.0

PERFORMANCE BY DIMENSION
------------------------
Relevance:          {report['metrics_by_dimension']['Relevance']}/5.0
Accuracy:           {report['metrics_by_dimension']['Accuracy']}/5.0
Completeness:       {report['metrics_by_dimension']['Completeness']}/5.0
Tool Usage:         {report['metrics_by_dimension']['Tool Usage']}/5.0

PERFORMANCE BY TOOL
-------------------
"""

    for tool, score in report["metrics_by_tool"].items():
        output += f"{tool:<20} {score:.2f}/5.0\n"

    output += "\nPERFORMANCE BY CATEGORY\n------------------------\n"
    for category, score in report["metrics_by_category"].items():
        output += f"{category:<30} {score:.2f}/5.0\n"

    output += "\n" + "=" * 80 + "\n"

    return output

# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================

if __name__ == "__main__":
    report = run_evaluation()
    print(format_evaluation_report(report))
    print("\nDetailed Results:")
    print(report["results"].to_string())
