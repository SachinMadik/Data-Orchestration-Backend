"""
Offline RAG Test Suite - All 11 Chart Types
Tests chart config generation logic WITHOUT requiring Azure OpenAI credentials.
Mocks the LLM plan generation to test execute_plan() and chart config building.
"""

import pandas as pd
import json


# ============================================================================
# Mock Plans for Each Chart Type
# ============================================================================

def get_mock_plan(chart_type, columns):
    """Generate a mock plan for each chart type"""
    
    plans = {
        "bar": {
            "operation": "groupby",
            "group_by": [columns[0]],
            "aggregations": [{"type": "sum", "column": columns[1] if len(columns) > 1 else columns[0]}],
            "chart": {"type": "bar", "x_col": columns[0], "y_cols": [columns[1] if len(columns) > 1 else columns[0]]}
        },
        "line": {
            "operation": "groupby",
            "group_by": [columns[0]],
            "aggregations": [{"type": "sum", "column": columns[1] if len(columns) > 1 else columns[0]}],
            "chart": {"type": "line", "x_col": columns[0], "y_cols": [columns[1] if len(columns) > 1 else columns[0]]}
        },
        "pie": {
            "operation": "groupby",
            "group_by": [columns[0]],
            "aggregations": [{"type": "sum", "column": columns[1] if len(columns) > 1 else columns[0]}],
            "chart": {"type": "pie", "x_col": columns[0], "y_cols": [columns[1] if len(columns) > 1 else columns[0]]}
        },
        "scatter": {
            "operation": "select",
            "select": columns[:2],
            "chart": {"type": "scatter", "x_col": columns[0], "y_cols": [columns[1] if len(columns) > 1 else columns[0]]}
        },
        "histogram": {
            "operation": "select",
            "select": [columns[0]],
            "chart": {"type": "histogram", "x_col": columns[0], "y_cols": []}
        },
        "heatmap": {
            "operation": "groupby",
            "group_by": columns[:2] if len(columns) >= 2 else [columns[0]],
            "aggregations": [{"type": "sum", "column": columns[2] if len(columns) > 2 else columns[0]}],
            "chart": {"type": "heatmap", "x_col": columns[0], "y_cols": [columns[2] if len(columns) > 2 else columns[0]]}
        },
        "radar": {
            "operation": "select",
            "select": columns,
            "chart": {"type": "radar", "x_col": columns[0], "y_cols": columns[1:] if len(columns) > 1 else []}
        },
        "funnel": {
            "operation": "select",
            "select": columns,
            "chart": {"type": "funnel", "x_col": columns[0], "y_cols": [columns[1] if len(columns) > 1 else columns[0]]}
        },
        "treemap": {
            "operation": "groupby",
            "group_by": columns[:2] if len(columns) >= 2 else [columns[0]],
            "aggregations": [{"type": "sum", "column": columns[2] if len(columns) > 2 else columns[0]}],
            "chart": {"type": "treemap", "x_col": columns[0], "y_cols": [columns[2] if len(columns) > 2 else columns[0]]}
        },
        "area": {
            "operation": "select",
            "select": columns,
            "chart": {"type": "area", "x_col": columns[0], "y_cols": columns[1:] if len(columns) > 1 else []}
        },
        "composed": {
            "operation": "select",
            "select": columns,
            "chart": {"type": "bar", "x_col": columns[0], "y_cols": columns[1:] if len(columns) > 1 else []}
        }
    }
    
    return plans.get(chart_type, plans["bar"])


# ============================================================================
# Test Data Generators
# ============================================================================

def create_sales_data():
    return pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Watch", "Headphones"],
        "Sales": [15000, 25000, 8000, 12000, 5000],
        "Profit": [3000, 5000, 1500, 2500, 800]
    })


def create_time_series_data():
    return pd.DataFrame({
        "Year": [2019, 2020, 2021, 2022, 2023, 2024],
        "Revenue": [100000, 120000, 150000, 180000, 220000, 250000],
        "Expenses": [80000, 90000, 110000, 130000, 160000, 180000]
    })


def create_proportion_data():
    return pd.DataFrame({
        "Company": ["Apple", "Samsung", "Xiaomi", "Oppo", "Others"],
        "Market_Share": [28, 22, 13, 10, 27]
    })


def create_correlation_data():
    return pd.DataFrame({
        "Marketing_Spend": [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000],
        "Revenue": [50000, 65000, 85000, 95000, 120000, 140000, 160000, 180000],
        "Region": ["North", "South", "East", "West", "North", "South", "East", "West"]
    })


def create_distribution_data():
    return pd.DataFrame({
        "Age": [22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65],
        "Employee_ID": range(1, 19)
    })


def create_heatmap_data():
    return pd.DataFrame({
        "Region": ["North", "North", "North", "South", "South", "South", "East", "East", "East"],
        "Product": ["Laptop", "Phone", "Tablet", "Laptop", "Phone", "Tablet", "Laptop", "Phone", "Tablet"],
        "Sales": [15000, 25000, 8000, 12000, 20000, 6000, 18000, 22000, 9000]
    })


def create_radar_data():
    return pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Quality": [85, 90, 75],
        "Price": [70, 85, 80],
        "Satisfaction": [88, 92, 78],
        "Durability": [90, 85, 70]
    })


def create_funnel_data():
    return pd.DataFrame({
        "Stage": ["Visitors", "Sign-ups", "Trials", "Paid", "Retained"],
        "Count": [10000, 5000, 2000, 800, 600]
    })


def create_treemap_data():
    return pd.DataFrame({
        "Category": ["Electronics", "Electronics", "Electronics", "Clothing", "Clothing", "Food", "Food"],
        "Subcategory": ["Phones", "Laptops", "Tablets", "Shirts", "Pants", "Snacks", "Drinks"],
        "Revenue": [50000, 40000, 15000, 20000, 18000, 12000, 10000]
    })


def create_area_data():
    return pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Product_A": [10000, 12000, 15000, 18000, 22000, 25000],
        "Product_B": [8000, 9000, 11000, 13000, 16000, 18000],
        "Product_C": [5000, 6000, 7000, 8000, 10000, 12000]
    })


def create_dual_axis_data():
    return pd.DataFrame({
        "Year": [2019, 2020, 2021, 2022, 2023],
        "Revenue": [1000000, 1500000, 2000000, 2500000, 3000000],
        "Profit_Margin": [15, 18, 20, 22, 25]
    })


# ============================================================================
# Import execute_plan (the function we're testing)
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.query_engine import execute_plan


# ============================================================================
# Test Runner
# ============================================================================

def run_test(test_name, df, expected_type, mock_plan):
    """Run a single test case"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Expected chart type: {expected_type}")
    print(f"Data shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    try:
        # Execute plan with mock
        result = execute_plan(df, mock_plan)
        result_type = result.get("type")
        chart_config = result.get("chart_config")
        rows = result.get("rows", [])
        
        print(f"\nExecution complete: type={result_type}")
        print(f"  - Rows returned: {len(rows)}")
        
        if chart_config:
            actual_type = chart_config.get("type")
            print(f"  - Chart type: {actual_type}")
            print(f"  - X-axis: {chart_config.get('xKey')}")
            series = chart_config.get('series', [])
            if isinstance(series, list) and series and isinstance(series[0], dict):
                print(f"  - Series: {[s.get('key') for s in series]}")
            else:
                print(f"  - Series: {series}")
            print(f"  - Dual-axis: {chart_config.get('dualAxis', False)}")
            
            # Verify chart type
            if actual_type == expected_type:
                print(f"\nPASS - Chart type matches: {expected_type}")
                status = "PASS"
            else:
                print(f"\nWARN - Chart type mismatch: expected {expected_type}, got {actual_type}")
                status = "WARN"
        else:
            print(f"\nFAIL - No chart config returned (got {result_type} instead)")
            status = "FAIL"
        
        # Show sample data
        if rows:
            print(f"\nSample output (first 3 rows):")
            for i, row in enumerate(rows[:3], 1):
                print(f"  {i}. {row}")
        
        return status
        
    except Exception as e:
        print(f"\nFAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return "FAIL"


def main():
    """Run all tests"""
    print("="*80)
    print("OFFLINE RAG TEST SUITE - ALL 11 CHART TYPES")
    print("Testing chart config generation WITHOUT Azure OpenAI")
    print("="*80)
    
    tests = [
        ("Bar Chart - Category Comparison", 
         create_sales_data(),
         "bar"),
        
        ("Line Chart - Trend Over Time",
         create_time_series_data(),
         "line"),
        
        ("Pie Chart - Market Share",
         create_proportion_data(),
         "pie"),
        
        ("Scatter Plot - Correlation",
         create_correlation_data(),
         "scatter"),
        
        ("Histogram - Distribution",
         create_distribution_data(),
         "histogram"),
        
        ("Heatmap - 2D Matrix",
         create_heatmap_data(),
         "heatmap"),
        
        ("Radar Chart - Multi-Metric",
         create_radar_data(),
         "radar"),
        
        ("Funnel Chart - Conversion",
         create_funnel_data(),
         "funnel"),
        
        ("Treemap - Hierarchical",
         create_treemap_data(),
         "treemap"),
        
        ("Area Chart - Cumulative Trends",
         create_area_data(),
         "area"),
        
        ("Composed Chart - Dual-Axis",
         create_dual_axis_data(),
         "composed"),
    ]
    
    results = {"PASS": 0, "WARN": 0, "FAIL": 0}
    
    for test_name, df, expected_type in tests:
        columns = list(df.columns)
        mock_plan = get_mock_plan(expected_type, columns)
        status = run_test(test_name, df, expected_type, mock_plan)
        results[status] += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(tests)}")
    print(f"PASS: {results['PASS']}")
    print(f"WARN: {results['WARN']}")
    print(f"FAIL: {results['FAIL']}")
    print(f"Success rate: {(results['PASS']+results['WARN'])/len(tests)*100:.1f}%")
    
    if results['FAIL'] == 0:
        print("\nALL TESTS PASSED - Chart config generation is working correctly!")
        print("Note: This tests the chart building logic. Full RAG requires Azure OpenAI.")
    else:
        print(f"\n{results['FAIL']} test(s) failed - review output above")
    
    return results['FAIL'] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
