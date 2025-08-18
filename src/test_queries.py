"""
Test Query Sets for Router Testing
Provides categorized queries for testing different routing scenarios
"""

# Product-related queries
PRODUCT_QUERIES = [
    "What are the main features of the iPhone 15 Pro?",
    "Show me the technical specifications for Model XYZ",
    "How do I install this software?",
    "What's included in the premium package?",
    "Is this product compatible with Windows 11?",
    "Product warranty and return policy",
    "User manual for the smart home device",
    "Compare different product models",
    "What materials is this product made from?",
    "Product safety certifications"
]

# Analytics/Data queries
ANALYTICS_QUERIES = [
    "Show me total revenue for last quarter",
    "What's the customer acquisition cost?",
    "Monthly active users trend",
    "Calculate conversion rate by channel",
    "Top performing products by region",
    "Year-over-year growth analysis",
    "Customer lifetime value metrics",
    "Churn rate by subscription tier",
    "Average order value by category",
    "Sales forecast for next quarter"
]

# Support queries
SUPPORT_QUERIES = [
    "How do I reset my password?",
    "My order hasn't arrived yet",
    "I want to return this item",
    "Payment method not working",
    "Account suspended without reason",
    "Need help with installation",
    "Product is defective",
    "Cancel my subscription",
    "Update shipping address",
    "Contact customer service"
]

# Ambiguous queries (could go multiple ways)
AMBIGUOUS_QUERIES = [
    "Tell me about performance",  # Product performance or business performance?
    "Show me the latest updates",  # Product updates or data updates?
    "What are the current issues?",  # Product issues or business issues?
    "I need help with integration",  # Technical or business process?
    "Review the recent changes",  # Code or business changes?
    "Check the status",  # Order status or system status?
    "Analyze the trends",  # Market or technical trends?
    "What's the capacity?",  # Product or system capacity?
    "Show me the dashboard",  # Analytics or product dashboard?
    "Generate a report"  # What kind of report?
]

# Complex queries requiring strong model (for RouteLLM testing)
COMPLEX_QUERIES = [
    "Design a scalable microservices architecture for a real-time trading platform with sub-millisecond latency requirements",
    "Analyze the implications of GDPR compliance on our data architecture and suggest implementation strategies",
    "Compare different machine learning approaches for fraud detection in financial transactions",
    "Develop a comprehensive disaster recovery plan for multi-region cloud deployment",
    "Evaluate the trade-offs between different consensus algorithms for our blockchain implementation",
    "Create a detailed migration strategy from monolithic to serverless architecture",
    "Optimize our recommendation engine algorithm for better personalization",
    "Design a real-time data pipeline for processing IoT sensor data at scale",
    "Implement a zero-trust security model for our enterprise infrastructure",
    "Architect a CQRS and event sourcing solution for our e-commerce platform"
]

# Simple queries (for RouteLLM weak model)
SIMPLE_QUERIES = [
    "What is machine learning?",
    "Define API",
    "What's the date today?",
    "Convert 100 USD to EUR",
    "What's 15% of 200?",
    "Capital of Japan",
    "How many days in February?",
    "What does HTTP stand for?",
    "List primary colors",
    "What's 2+2?"
]

def get_test_suite(category: str = "all"):
    """Get test queries by category"""
    suites = {
        "product": PRODUCT_QUERIES,
        "analytics": ANALYTICS_QUERIES,
        "support": SUPPORT_QUERIES,
        "ambiguous": AMBIGUOUS_QUERIES,
        "complex": COMPLEX_QUERIES,
        "simple": SIMPLE_QUERIES,
        "all": (PRODUCT_QUERIES + ANALYTICS_QUERIES + SUPPORT_QUERIES + 
                AMBIGUOUS_QUERIES + COMPLEX_QUERIES + SIMPLE_QUERIES)
    }
    return suites.get(category, [])

if __name__ == "__main__":
    # Example: Print all ambiguous queries
    print("Ambiguous Test Queries:")
    for i, query in enumerate(AMBIGUOUS_QUERIES, 1):
        print(f"{i}. {query}")