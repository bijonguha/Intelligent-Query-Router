#!/usr/bin/env python3
"""
Combined Comparison: Semantic Router vs RouteLLM
Direct comparison of both routing approaches
"""

import os
import time
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer
from colorama import init, Fore, Style
from tabulate import tabulate

init(autoreset=True)
load_dotenv()

class RouterComparison:
    def __init__(self):
        """Initialize both routers for comparison"""
        print(f"{Fore.CYAN}Initializing Router Comparison Suite...{Style.RESET_ALL}\n")
        
        # Initialize Semantic Router
        self._init_semantic_router()
        
        # Initialize RouteLLM (simplified version)
        self._init_routellm()
        
        print(f"{Fore.GREEN}✓ Both routers initialized{Style.RESET_ALL}\n")
    
    def _init_semantic_router(self):
        """Initialize Semantic Router"""
        print("Setting up Semantic Router...")
        
        encoder = OpenAIEncoder(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )
        
        # Create simple routes for comparison
        routes = [
            Route(
                name="technical",
                utterances=[
                    "Debug this code",
                    "Optimize algorithm",
                    "System architecture",
                    "API integration",
                    "Database schema"
                ]
            ),
            Route(
                name="business",
                utterances=[
                    "Revenue analysis",
                    "Market trends",
                    "Customer metrics",
                    "Sales forecast",
                    "ROI calculation"
                ]
            ),
            Route(
                name="general",
                utterances=[
                    "Tell me about",
                    "What is",
                    "How does",
                    "Explain",
                    "Describe"
                ]
            )
        ]
        
        self.semantic_router = RouteLayer(encoder=encoder, routes=routes)
    
    def _init_routellm(self):
        """Initialize RouteLLM simulator"""
        print("Setting up RouteLLM...")
        self.routellm_threshold = 0.7
    
    def semantic_route(self, query: str) -> Dict[str, Any]:
        """Route using Semantic Router"""
        start = time.time()
        
        result = self.semantic_router(query)
        
        latency = (time.time() - start) * 1000
        
        return {
            "method": "Semantic Router",
            "route": result.name if result else "no_match",
            "latency_ms": round(latency, 2),
            "cost": 0.00001,  # Only embedding cost
            "accuracy": "High for defined routes"
        }
    
    def routellm_route(self, query: str) -> Dict[str, Any]:
        """Route using RouteLLM approach"""
        start = time.time()
        
        # Simulate complexity classification
        complexity = len(query) / 100  # Simplified
        model = "gpt-4" if complexity > self.routellm_threshold else "gpt-3.5-turbo"
        
        # Simulate processing time
        time.sleep(0.05)  # Simulate classifier latency
        
        latency = (time.time() - start) * 1000
        
        return {
            "method": "RouteLLM",
            "route": model,
            "latency_ms": round(latency, 2),
            "cost": 0.001 if model == "gpt-4" else 0.0001,
            "accuracy": "Adaptive to any query"
        }
    
    async def compare_routers(self):
        """Run comprehensive comparison"""
        print(f"{Fore.YELLOW}=== Router Comparison Analysis ==={Style.RESET_ALL}\n")
        
        # Test queries
        queries = [
            "Debug this Python async function that's causing deadlocks",
            "What's our Q3 revenue compared to last year?",
            "Explain machine learning",
            "Design a microservices architecture for e-commerce",
            "Calculate customer acquisition cost",
            "How does photosynthesis work?",
            "Optimize this SQL query for better performance",
            "Show me user engagement metrics",
            "What is the capital of France?"
        ]
        
        print(f"Comparing {len(queries)} queries across both routers:\n")
        print("-" * 80)
        
        # Store results for comparison
        semantic_results = []
        routellm_results = []
        
        for query in queries:
            print(f"\nQuery: {Fore.CYAN}{query[:60]}...{Style.RESET_ALL}")
            
            # Test Semantic Router
            sem_result = self.semantic_route(query)
            semantic_results.append(sem_result)
            print(f"  {Fore.GREEN}Semantic Router:{Style.RESET_ALL}")
            print(f"    Route: {sem_result['route']}")
            print(f"    Latency: {sem_result['latency_ms']}ms")
            
            # Test RouteLLM
            route_result = self.routellm_route(query)
            routellm_results.append(route_result)
            print(f"  {Fore.YELLOW}RouteLLM:{Style.RESET_ALL}")
            print(f"    Model: {route_result['route']}")
            print(f"    Latency: {route_result['latency_ms']}ms")
        
        print("\n" + "=" * 80)
        
        # Summary comparison
        print(f"\n{Fore.CYAN}=== Performance Comparison ==={Style.RESET_ALL}\n")
        
        # Calculate averages
        sem_avg_latency = sum(r['latency_ms'] for r in semantic_results) / len(semantic_results)
        route_avg_latency = sum(r['latency_ms'] for r in routellm_results) / len(routellm_results)
        
        sem_total_cost = sum(r['cost'] for r in semantic_results)
        route_total_cost = sum(r['cost'] for r in routellm_results)
        
        # Create comparison table
        comparison_data = [
            ["Metric", "Semantic Router", "RouteLLM", "Winner"],
            ["Avg Latency", f"{sem_avg_latency:.2f}ms", f"{route_avg_latency:.2f}ms", 
             "Semantic Router ✓" if sem_avg_latency < route_avg_latency else "RouteLLM ✓"],
            ["Total Cost", f"${sem_total_cost:.6f}", f"${route_total_cost:.4f}",
             "Semantic Router ✓" if sem_total_cost < route_total_cost else "RouteLLM ✓"],
            ["Flexibility", "Predefined routes", "Any query type", "RouteLLM ✓"],
            ["Setup Complexity", "Simple", "Moderate", "Semantic Router ✓"],
            ["Accuracy", "High for known patterns", "Adaptive", "Depends on use case"],
        ]
        
        print(tabulate(comparison_data, headers="firstrow", tablefmt="grid"))
        
        # Use case recommendations
        print(f"\n{Fore.CYAN}=== Recommendations ==={Style.RESET_ALL}\n")
        
        print(f"{Fore.GREEN}Use Semantic Router when:{Style.RESET_ALL}")
        print("  ✓ You have well-defined routing categories")
        print("  ✓ Ultra-low latency is critical (<50ms)")
        print("  ✓ Cost minimization is important")
        print("  ✓ Query patterns are predictable")
        print("  ✓ You need deterministic routing\n")
        
        print(f"{Fore.YELLOW}Use RouteLLM when:{Style.RESET_ALL}")
        print("  ✓ Query patterns are diverse and unpredictable")
        print("  ✓ You need to balance cost vs quality dynamically")
        print("  ✓ Routing logic is complex (not just categories)")
        print("  ✓ You want to optimize model selection per query")
        print("  ✓ Slight latency increase is acceptable\n")
        
        print(f"{Fore.MAGENTA}Hybrid Approach (Best of Both):{Style.RESET_ALL}")
        print("  1. Use Semantic Router for initial categorization")
        print("  2. Use RouteLLM within categories for model selection")
        print("  3. Cache frequent queries to reduce costs further")
        print("  4. Monitor and adjust thresholds based on feedback")

if __name__ == "__main__":
    comparison = RouterComparison()
    asyncio.run(comparison.compare_routers())