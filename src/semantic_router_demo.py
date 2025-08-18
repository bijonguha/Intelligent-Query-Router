#!/usr/bin/env python3
"""
Semantic Router Demo with OpenAI
Demonstrates ultra-fast routing using vector similarity
"""

import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer
from colorama import init, Fore, Style
import json

# Initialize colorama for colored output
init(autoreset=True)

# Load environment variables
load_dotenv()

class SemanticRouterDemo:
    def __init__(self):
        """Initialize the Semantic Router with OpenAI embeddings"""
        print(f"{Fore.CYAN}Initializing Semantic Router with OpenAI...{Style.RESET_ALL}")
        
        # Initialize OpenAI encoder
        self.encoder = OpenAIEncoder(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )
        
        # Define routes
        self.routes = self._create_routes()
        
        # Create route layer
        self.router = RouteLayer(encoder=self.encoder, routes=self.routes)
        
        print(f"{Fore.GREEN}✓ Semantic Router initialized with {len(self.routes)} routes{Style.RESET_ALL}")
    
    def _create_routes(self) -> List[Route]:
        """Create routing definitions with example utterances"""
        
        # Product documentation route
        product_route = Route(
            name="product_documentation",
            utterances=[
                "What are the features of this product?",
                "Show me the product specifications",
                "How does this product work?",
                "What's included in the package?",
                "Product installation guide",
                "User manual for this device",
                "Product warranty information",
                "What materials is this made of?",
                "Product dimensions and weight",
                "Is this product compatible with",
                "How to maintain this product",
                "Product safety instructions",
                "What are the technical specifications?",
                "Show me the product datasheet",
                "Product configuration options"
            ]
        )
        
        # Analytics/SQL route
        analytics_route = Route(
            name="data_analytics",
            utterances=[
                "Show me sales data for last month",
                "What's our total revenue?",
                "How many customers do we have?",
                "Calculate the average order value",
                "Show me trends in sales",
                "Compare metrics between regions",
                "Generate a performance report",
                "What's the count of active users?",
                "Group sales by category",
                "Year over year growth analysis",
                "Show me monthly statistics",
                "What are our top selling products?",
                "Filter orders by status",
                "Aggregate results by quarter",
                "Show me customer demographics"
            ]
        )
        
        # Customer support route
        support_route = Route(
            name="customer_support",
            utterances=[
                "I have a problem with my order",
                "How can I return this item?",
                "Where is my package?",
                "I need help with my account",
                "The product is not working",
                "How do I contact support?",
                "I want to file a complaint",
                "Can I cancel my order?",
                "Refund request",
                "Track my shipment",
                "Update my delivery address",
                "I forgot my password",
                "My payment failed",
                "Report a damaged product",
                "Speaking to a human agent"
            ]
        )
        
        # General conversation route
        general_route = Route(
            name="general_conversation",
            utterances=[
                "Hello, how are you?",
                "What's the weather like?",
                "Tell me a joke",
                "What can you do?",
                "Who are you?",
                "Thank you for your help",
                "Goodbye",
                "What time is it?",
                "How's your day going?",
                "Nice to meet you",
                "Can you help me?",
                "What's new?",
                "How does this work?",
                "Tell me more",
                "I don't understand"
            ]
        )
        
        return [product_route, analytics_route, support_route, general_route]
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route a single query and return results with timing"""
        start_time = time.time()
        
        # Get route decision
        route_decision = self.router(query)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        result = {
            "query": query,
            "route": route_decision.name if route_decision else "no_match",
            "latency_ms": round(latency_ms, 2),
            "confidence": route_decision.score if hasattr(route_decision, 'score') else None
        }
        
        return result
    
    def run_demo(self):
        """Run interactive demo"""
        print(f"\n{Fore.YELLOW}=== Semantic Router Demo ==={Style.RESET_ALL}\n")
        
        # Test queries
        test_queries = [
            "What are the features of the iPhone 15?",
            "Show me revenue for Q4 2023",
            "I need to return my order",
            "Hello, how can you help me?",
            "Calculate total sales by region",
            "The product manual is missing",
            "How many active users last month?",
            "Track order #12345",
            "Product compatibility with Windows 11",
            "What's the average customer lifetime value?"
        ]
        
        print(f"Testing {len(test_queries)} queries:\n")
        
        results = []
        total_latency = 0
        
        for query in test_queries:
            result = self.route_query(query)
            results.append(result)
            total_latency += result['latency_ms']
            
            # Color code by route
            route_colors = {
                "product_documentation": Fore.BLUE,
                "data_analytics": Fore.GREEN,
                "customer_support": Fore.YELLOW,
                "general_conversation": Fore.MAGENTA,
                "no_match": Fore.RED
            }
            
            color = route_colors.get(result['route'], Fore.WHITE)
            print(f"{color}Query: {query[:50]}...")
            print(f"  → Route: {result['route']}")
            print(f"  → Latency: {result['latency_ms']}ms{Style.RESET_ALL}")
            print()
        
        # Summary statistics
        avg_latency = total_latency / len(test_queries)
        print(f"\n{Fore.CYAN}=== Performance Summary ==={Style.RESET_ALL}")
        print(f"Total queries: {len(test_queries)}")
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Total time: {total_latency:.2f}ms")
        
        # Route distribution
        print(f"\n{Fore.CYAN}=== Route Distribution ==={Style.RESET_ALL}")
        route_counts = {}
        for result in results:
            route = result['route']
            route_counts[route] = route_counts.get(route, 0) + 1
        
        for route, count in route_counts.items():
            percentage = (count / len(test_queries)) * 100
            print(f"{route}: {count} ({percentage:.1f}%)")
        
        # Interactive mode
        print(f"\n{Fore.YELLOW}=== Interactive Mode ==={Style.RESET_ALL}")
        print("Enter queries to test routing (type 'quit' to exit):\n")
        
        while True:
            try:
                user_query = input(f"{Fore.CYAN}Query> {Style.RESET_ALL}")
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                
                result = self.route_query(user_query)
                color = route_colors.get(result['route'], Fore.WHITE)
                print(f"{color}  → Route: {result['route']}")
                print(f"  → Latency: {result['latency_ms']}ms{Style.RESET_ALL}\n")
                
            except KeyboardInterrupt:
                break
        
        print(f"\n{Fore.GREEN}Demo completed!{Style.RESET_ALL}")

if __name__ == "__main__":
    demo = SemanticRouterDemo()
    demo.run_demo()