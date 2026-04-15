# Production-Grade RAG Upgrade Plan

## Current Limitations
1. **Only 3 chart types** (bar, line, pie) - hardcoded
2. **No advanced visualizations** (scatter, heatmap, radar, funnel, treemap, histogram, area)
3. **Weak context** - only 1000 chars per doc in RAG
4. **No chart type intelligence** - backend guesses, no negotiation
5. **Keyword search only** - no semantic reranking
6. **No response format negotiation** - text/table/chart decided arbitrarily

## Production-Grade Requirements
✅ **10+ chart types** with intelligent selection
✅ **Structured + Unstructured data** - unified query engine
✅ **Rich context** - 6000+ chars per doc for complex queries
✅ **Smart routing** - intent detection for all visualization types
✅ **Dual-axis charts** - automatic scale detection
✅ **Error handling** - graceful fallbacks, no crashes
✅ **Response negotiation** - LLM picks best format (text/table/chart)

## Key Changes

### 1. Chart Types Expansion (10 types)
- bar, line, pie (existing)
- **area** - cumulative trends
- **scatter** - correlation analysis
- **histogram** - distribution
- **radar** - multi-metric comparison
- **heatmap** - 2D categorical matrix
- **funnel** - conversion pipelines
- **treemap** - hierarchical data
- **composed** - dual-axis (bar+line)

### 2. Query Engine Intelligence
- LLM-driven chart type selection based on data shape
- Automatic histogram binning for numeric distributions
- Heatmap pivot for 2D categorical data
- Scatter plot for correlation queries
- Funnel detection for sequential stage data

### 3. RAG Context Expansion
- Increase from 1000 → 6000 chars per doc
- Smart truncation - preserve tables/structured content
- Multi-doc aggregation for analytical queries

### 4. Response Format Negotiation
- LLM decides: text | table | chart based on query intent
- Chart type selected by data characteristics
- Fallback chain: chart → table → text

## Files to Modify

### Priority 1 (Core)
1. `services/query_engine.py` - chart type expansion + selection logic
2. `services/openai_service.py` - RAG context + chart type support
3. `function_app.py` - unified response builder

### Priority 2 (Enhancement)
4. `services/rag_service.py` - richer context window
5. `services/router_service.py` - smarter intent detection

## Implementation Status
- [x] Plan created
- [ ] query_engine.py upgraded
- [ ] openai_service.py upgraded
- [ ] function_app.py upgraded
- [ ] rag_service.py upgraded
- [ ] router_service.py upgraded
- [ ] Testing complete

## Next Steps
Apply all changes in one batch - ETA 2 minutes.
