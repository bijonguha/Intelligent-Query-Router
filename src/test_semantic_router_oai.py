#!/usr/bin/env python3
"""
Simple test script for semantic_router_oai.py
Tests routing functionality without interactive mode
"""

from semantic_router_oai import DirectOpenAISemanticRouter

def test_routing():
    """Test the semantic router with a few queries"""
    print("Testing DirectOpenAISemanticRouter...")
    
    # Initialize router
    router = DirectOpenAISemanticRouter()
    
    # Test queries
    test_queries = [
        "Fix this bug in my Python code",
        "Show me quarterly revenue analysis", 
        "Help customer with refund request",
        "Write a blog post about AI",
        "Research market trends",
        "Hello, how can you help me?"
    ]
    
    print(f"\nTesting {len(test_queries)} queries:")
    print("-" * 50)
    
    for query in test_queries:
        result = router.route_query(query)
        print(f"Query: {query}")
        print(f"  → Route: {result['route']}")
        print(f"  → Latency: {result['latency_ms']}ms")
        print()
    
    # Show statistics
    stats = router.get_performance_stats()
    print("Performance Summary:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Average latency: {stats['average_latency_ms']:.2f}ms")
    print(f"  Success rate: {stats['success_rate']}%")

if __name__ == "__main__":
    test_routing()