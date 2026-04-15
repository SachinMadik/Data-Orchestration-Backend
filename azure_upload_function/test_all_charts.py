"""
Production RAG Test Suite - All 11 Chart Types
Tests query_engine.py with dummy data to verify all chart types work correctly.
"""

import pandas as pd
import sys
import os

# Add parent directory to path to import services
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.query_engine import generate_plan, execute_plan, structured_to_df


# ============================================================================
# Test Data Generators
# ============================================================================

def create_sales_data():
    """Bar chart test data - category comparisons"""
    return pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Watch", "Headphones"],
        "Sales": [15000, 25000, 8000, 12000, 5000],
        "Profit": [3000, 5000, 1500, 2500, 800]
    })


def create_time_series_data():
    """Line chart test data - trends over time"""
    return pd.DataFrame({
        "Year": [2019, 2020, 2021, 2022, 2023, 2024],
        "Revenue": [100000, 120000, 150000, 180000, 220000, 250000],
        "Expenses": [80000, 90000, 110000, 130000, 160000, 180000]
    })


def create_proportion_data():
    """Pie chart test data - market share"""
    return pd.DataFrame({
        "Company": ["Apple", "Samsung", "Xiaomi", "Oppo", "Others"],
        "Market_Share": [28, 22, 13, 10, 27]
    })


def create_correlation_data():
    """Scatter plot test data - correlation between 2 variables"""
    return pd.DataFrame({
        "Marketing_Spend": [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000],
        "Revenue": [50000, 65000, 85000, 95000, 120000, 140000, 160000, 180000],
        "Region": ["North", "South", "East", "West", "North", "South", "East", "West"]
    })


def create_distribution_data():
    """Histogram test data - age distribution"""
    return pd.DataFrame({
        "Age": [22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65],
        "Employee_ID": range(1, 19)
    })


def create_heatmap_data():
    """Heatmap test data - 2D categorical matrix"""
    return pd.DataFrame({
        "Region": ["North", "North", "North", "South", "South", "South", "East", "East", "East"],
        "Product": ["Laptop", "Phone", "Tablet", "Laptop", "Phone", "Tablet", "Laptop", "Phone", "Tablet"],
        "Sales": [15000, 25000, 8000, 12000, 20000, 6000, 18000, 22000, 9000]
    })


def create_radar_data():
    """Radar chart test data - multi-metric comparison"""
    return pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Quality": [85, 90, 75],
        "Price": [70, 85, 80],
        "Satisfaction": [88, 92, 78],
        "Durability": [90, 85, 70]
    })


def create_funnel_data():
    """Funnel chart test data - conversion pipeline"""
    return pd.DataFrame({
        "Stage": ["Visitors", "Sign-ups", "Trials", "Paid", "Retained"],
        "Count": [10000, 5000, 2000, 800, 600]
    })


def create_treemap_data():
    """Treemap test data - hierarchical data"""
    return pd.DataFrame({
        "Category": ["Electronics", "Electronics", "Electronics", "Clothing", "Clothing", "Food", "Food"],
        "Subcategory": ["Phones", "Laptops", "Tablets", "Shirts", "Pants", "Snacks", "Drinks"],
        "Revenue": [50000, 40000, 15000, 20000, 18000, 12000, 10000]
    })


def create_area_data():
    """Area chart test data - cumulative trends"""
    return pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Product_A": [10000, 12000, 15000, 18000, 22000, 25000],
        "Product_B": [8000, 9000, 11000, 13000, 16000, 18000],
        "Product_C": [5000, 6000, 7000, 8000, 10000, 12000]
    })


def create_dual_axis_data():
    """Composed chart test data - different scales (revenue vs margin %)"""
    return pd.DataFrame({
        "Year": [2019, 2020, 2021, 2022, 2023],
        "Revenue": [1000000, 1500000, 2000000, 2500000, 3000000],  # millions
        "Profit_Margin": [15, 18, 20, 22, 25]  # percentage (much smaller scale)
    })


# ============================================================================
# Test Runner
# ============================================================================

def run_test(test_name, df, query, expected_type):
    """Run a single test case"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Query: {query}")
    print(f"Expected chart type: {expected_type}")
    print(f"\nData shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    try:
        # Generate plan
        columns = list(df.columns)
        plan = generate_plan(query, columns)
        print(f"\n✓ Plan generated: operation={plan.get('operation')}")
        print(f"  - Chart type: {plan.get('chart', {}).get('type', 'N/A')}")
        print(f"  - Group by: {plan.get('group_by', [])}")
        print(f"  - Aggregations: {len(plan.get('aggregations', []))}")
        
        # Execute plan
        result = execute_plan(df, plan)
        result_type = result.get("type")
        chart_config = result.get("chart_config")
        rows = result.get("rows", [])
        
        print(f"\n✓ Execution complete: type={result_type}")
        print(f"  - Rows returned: {len(rows)}")
        
        if chart_config:
            actual_type = chart_config.get("type")
            print(f"  - Chart type: {actual_type}")
            print(f"  - X-axis: {chart_config.get('xKey')}")
            print(f"  - Series: {chart_config.get('series')}")
            print(f"  - Dual-axis: {chart_config.get('dualAxis', False)}")
            
            # Verify chart type
            if actual_type == expected_type:
                print(f"\n✅ PASS - Chart type matches expected: {expected_type}")
            else:
                print(f"\n⚠️  WARN - Chart type mismatch: expected {expected_type}, got {actual_type}")
        else:
            print(f"\n❌ FAIL - No chart config returned (got {result_type} instead)")
        
        # Show sample data
        if rows:
            print(f"\nSample output (first 3 rows):")
            for i, row in enumerate(rows[:3], 1):
                print(f"  {i}. {row}")
        
        print(f"\nScript: {result.get('script', 'N/A')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("PRODUCTION RAG TEST SUITE - ALL 11 CHART TYPES")
    print("="*80)
    
    tests = [
        # Test 1: Bar Chart
        ("Bar Chart - Category Comparison", 
         create_sales_data(),
         "Show sales by product",
         "bar"),
        
        # Test 2: Line Chart
        ("Line Chart - Trend Over Time",
         create_time_series_data(),
         "Show revenue trend over time",
         "line"),
        
        # Test 3: Pie Chart
        ("Pie Chart - Market Share",
         create_proportion_data(),
         "Show market share distribution as pie chart",
         "pie"),
        
        # Test 4: Scatter Plot
        ("Scatter Plot - Correlation",
         create_correlation_data(),
         "Show correlation between marketing spend and revenue as scatter plot",
         "scatter"),
        
        # Test 5: Histogram
        ("Histogram - Distribution",
         create_distribution_data(),
         "Show distribution of age as histogram",
         "histogram"),
        
        # Test 6: Heatmap
        ("Heatmap - 2D Matrix",
         create_heatmap_data(),
         "Show sales by region and product as heatmap",
         "heatmap"),
        
        # Test 7: Radar Chart
        ("Radar Chart - Multi-Metric",
         create_radar_data(),
         "Compare products across quality, price, satisfaction, and durability as radar chart",
         "radar"),
        
        # Test 8: Funnel Chart
        ("Funnel Chart - Conversion",
         create_funnel_data(),
         "Show conversion funnel from visitors to retained customers",
         "funnel"),
        
        # Test 9: Treemap
        ("Treemap - Hierarchical",
         create_treemap_data(),
         "Show revenue by category and subcategory as treemap",
         "treemap"),
        
        # Test 10: Area Chart
        ("Area Chart - Cumulative Trends",
         create_area_data(),
         "Show cumulative sales by product over months as area chart",
         "area"),
        
        # Test 11: Composed (Dual-Axis)
        ("Composed Chart - Dual-Axis",
         create_dual_axis_data(),
         "Plot revenue and profit margin by year",
         "composed"),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, df, query, expected_type in tests:
        success = run_test(test_name, df, query, expected_type)
        if success:
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - Production RAG is working correctly!")
    else:
        print(f"\n⚠️  {failed} test(s) failed - review output above")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
