#!/usr/bin/env python3
"""
Combined Comparison: Semantic Router vs RouteLLM (OpenAI API Integration)
Direct comparison of both routing approaches using real OpenAI API calls
"""

import os
import time
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
from colorama import init, Fore, Style
from tabulate import tabulate

# Import our OAI implementations
from semantic_router_oai import DirectOpenAISemanticRouter
from routellm_oai import DirectOpenAIRouteLLM

init(autoreset=True)
load_dotenv()

class OAIRouterComparison:
    def __init__(self, api_key: str = None):
        """Initialize both routers for comparison using OpenAI API"""
        print(f"{Fore.CYAN}Initializing OpenAI Router Comparison Suite...{Style.RESET_ALL}\n")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize both routers
        print(f"{Fore.YELLOW}Setting up Semantic Router...{Style.RESET_ALL}")
        self.semantic_router = DirectOpenAISemanticRouter(api_key=self.api_key)
        
        print(f"\n{Fore.YELLOW}Setting up RouteLLM...{Style.RESET_ALL}")  
        self.routellm = DirectOpenAIRouteLLM(api_key=self.api_key)
        
        print(f"\n{Fore.GREEN}✓ Both OpenAI routers initialized successfully{Style.RESET_ALL}\n")
    
    async def semantic_route_test(self, query: str) -> Dict[str, Any]:
        """Test query with Semantic Router"""
        start = time.time()
        
        try:
            result = self.semantic_router.route_query(query, include_confidence=False)
            latency = result['latency_ms']
            success = result['success']
            route = result['route']
            
            return {
                "method": "Semantic Router",
                "query": query,
                "route": route,
                "latency_ms": latency,
                "cost": 0.00002,  # Approximate embedding cost
                "success": success,
                "response_type": "Route Classification Only",
                "model_used": "text-embedding-3-small"
            }
            
        except Exception as e:
            return {
                "method": "Semantic Router",
                "query": query,
                "route": "error",
                "latency_ms": (time.time() - start) * 1000,
                "cost": 0,
                "success": False,
                "response_type": f"Error: {str(e)}",
                "model_used": "text-embedding-3-small"
            }
    
    async def routellm_test(self, query: str) -> Dict[str, Any]:
        """Test query with RouteLLM"""
        try:
            result = await self.routellm.route_and_execute(query, max_tokens=100)
            
            return {
                "method": "RouteLLM",
                "query": query,
                "route": result['model'],
                "latency_ms": result['latency_ms'],
                "cost": result['cost'],
                "success": result['success'],
                "response_type": "Full LLM Response",
                "model_used": result['model'],
                "complexity": result['complexity']['score'],
                "tokens_used": result['tokens']['total'],
                "response_preview": result['response'][:100] + "..." if len(result['response']) > 100 else result['response']
            }
            
        except Exception as e:
            return {
                "method": "RouteLLM", 
                "query": query,
                "route": "error",
                "latency_ms": 0,
                "cost": 0,
                "success": False,
                "response_type": f"Error: {str(e)}",
                "model_used": "unknown",
                "complexity": 0,
                "tokens_used": 0,
                "response_preview": ""
            }
    
    async def compare_routers_comprehensive(self):
        """Run comprehensive comparison between both routing approaches"""
        print(f"{Fore.YELLOW}=== OpenAI Router Comparison Analysis ==={Style.RESET_ALL}\n")
        
        # Comprehensive test queries covering different complexity levels
        test_queries = [
            # Simple queries (should use weak model in RouteLLM)
            "What is Python?",
            "Define artificial intelligence",
            "Who created Linux?",
            
            # Medium complexity queries
            "How does machine learning work?", 
            "Explain REST API principles",
            "List benefits of microservices",
            
            # High complexity queries (should use strong model in RouteLLM)
            "Design a scalable database architecture for high-traffic applications",
            "Compare and analyze different consensus algorithms in distributed systems",
            "Implement an optimization strategy for real-time data processing pipelines"
        ]
        
        print(f"Comparing {len(test_queries)} queries across both routing approaches:\n")
        print("=" * 90)
        
        # Store results for analysis
        semantic_results = []
        routellm_results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{Fore.MAGENTA}[{i}/{len(test_queries)}] Query: {query[:60]}...{Style.RESET_ALL}")
            print("-" * 70)
            
            # Test both routers concurrently for fair comparison
            semantic_task = self.semantic_route_test(query)
            routellm_task = self.routellm_test(query)
            
            semantic_result, routellm_result = await asyncio.gather(semantic_task, routellm_task)
            
            semantic_results.append(semantic_result)
            routellm_results.append(routellm_result)
            
            # Display results side by side
            print(f"{Fore.GREEN}Semantic Router:{Style.RESET_ALL}")
            print(f"  → Route: {semantic_result['route']}")
            print(f"  → Latency: {semantic_result['latency_ms']:.2f}ms")
            print(f"  → Cost: ${semantic_result['cost']:.6f}")
            print(f"  → Success: {'✓' if semantic_result['success'] else '✗'}")
            
            print(f"{Fore.YELLOW}RouteLLM:{Style.RESET_ALL}")
            print(f"  → Model: {routellm_result['model_used']}")
            if 'complexity' in routellm_result:
                print(f"  → Complexity: {routellm_result['complexity']:.3f}")
            print(f"  → Latency: {routellm_result['latency_ms']:.2f}ms")
            print(f"  → Cost: ${routellm_result['cost']:.6f}")
            print(f"  → Success: {'✓' if routellm_result['success'] else '✗'}")
            if 'tokens_used' in routellm_result:
                print(f"  → Tokens: {routellm_result['tokens_used']}")
            if routellm_result['success'] and routellm_result['response_preview']:
                print(f"  → Preview: {routellm_result['response_preview']}")
        
        print("\n" + "=" * 90)
        
        # Generate comprehensive comparison analysis
        await self._generate_comparison_analysis(semantic_results, routellm_results)
        
        # Interactive comparison mode
        await self._run_interactive_comparison()
    
    async def _generate_comparison_analysis(self, semantic_results: List[Dict], routellm_results: List[Dict]):
        """Generate detailed comparison analysis"""
        print(f"\n{Fore.CYAN}=== Detailed Performance Analysis ==={Style.RESET_ALL}\n")
        
        # Calculate metrics
        sem_avg_latency = sum(r['latency_ms'] for r in semantic_results) / len(semantic_results)
        route_avg_latency = sum(r['latency_ms'] for r in routellm_results) / len(routellm_results)
        
        sem_total_cost = sum(r['cost'] for r in semantic_results)
        route_total_cost = sum(r['cost'] for r in routellm_results)
        
        sem_success_rate = (sum(1 for r in semantic_results if r['success']) / len(semantic_results)) * 100
        route_success_rate = (sum(1 for r in routellm_results if r['success']) / len(routellm_results)) * 100
        
        # Model usage analysis for RouteLLM
        strong_model_usage = sum(1 for r in routellm_results if 'gpt-4' in r.get('model_used', ''))
        weak_model_usage = len(routellm_results) - strong_model_usage
        
        # Create detailed comparison table
        comparison_data = [
            ["Metric", "Semantic Router", "RouteLLM", "Winner"],
            ["Average Latency", f"{sem_avg_latency:.2f}ms", f"{route_avg_latency:.2f}ms", 
             "Semantic Router ✓" if sem_avg_latency < route_avg_latency else "RouteLLM ✓"],
            ["Total Cost", f"${sem_total_cost:.6f}", f"${route_total_cost:.4f}",
             "Semantic Router ✓" if sem_total_cost < route_total_cost else "RouteLLM ✓"],
            ["Success Rate", f"{sem_success_rate:.1f}%", f"{route_success_rate:.1f}%",
             "Semantic Router ✓" if sem_success_rate >= route_success_rate else "RouteLLM ✓"],
            ["Functionality", "Route classification only", "Full LLM responses", "RouteLLM ✓"],
            ["Setup Complexity", "Simple", "Moderate", "Semantic Router ✓"],
            ["Scalability", "High (embedding only)", "Medium (full inference)", "Semantic Router ✓"],
            ["Response Quality", "Categories only", "Complete answers", "RouteLLM ✓"]
        ]
        
        print(tabulate(comparison_data, headers="firstrow", tablefmt="grid"))
        
        # RouteLLM specific analysis
        if routellm_results:
            print(f"\n{Fore.CYAN}=== RouteLLM Model Usage Analysis ==={Style.RESET_ALL}")
            print(f"Strong model usage: {strong_model_usage}/{len(routellm_results)} queries ({(strong_model_usage/len(routellm_results)*100):.1f}%)")
            print(f"Weak model usage: {weak_model_usage}/{len(routellm_results)} queries ({(weak_model_usage/len(routellm_results)*100):.1f}%)")
            
            # Cost savings analysis
            if route_total_cost > 0:
                hypothetical_all_strong = len(routellm_results) * 0.002  # Approximate strong model cost
                savings = hypothetical_all_strong - route_total_cost
                savings_percent = (savings / hypothetical_all_strong) * 100 if hypothetical_all_strong > 0 else 0
                print(f"Cost savings vs all strong model: ${savings:.6f} ({savings_percent:.1f}%)")
        
        # Use case recommendations
        print(f"\n{Fore.CYAN}=== Use Case Recommendations ==={Style.RESET_ALL}\n")
        
        print(f"{Fore.GREEN}Use Semantic Router when:{Style.RESET_ALL}")
        print("  ✓ You need ultra-fast query classification (~400-800ms)")
        print("  ✓ Route-based logic is sufficient (no LLM responses needed)")
        print("  ✓ Cost minimization is critical (~$0.00002 per query)")
        print("  ✓ High throughput is essential")
        print("  ✓ Query patterns are predictable and fit defined categories")
        print("  ✓ Building intent classification systems\n")
        
        print(f"{Fore.YELLOW}Use RouteLLM when:{Style.RESET_ALL}")
        print("  ✓ You need complete LLM responses, not just routing")
        print("  ✓ Cost optimization between different model tiers")
        print("  ✓ Query complexity varies significantly")
        print("  ✓ Acceptable latency is 1-3 seconds")
        print("  ✓ You want dynamic model selection based on complexity")
        print("  ✓ Building conversational AI or Q&A systems\n")
        
        print(f"{Fore.MAGENTA}Hybrid Approach (Best of Both):{Style.RESET_ALL}")
        print("  1. Use Semantic Router for initial intent classification")
        print("  2. Use RouteLLM within specific intent categories")
        print("  3. Cache frequent query patterns")
        print("  4. Implement fallback mechanisms for edge cases")
        print("  5. Monitor and tune thresholds based on usage patterns")
    
    async def _run_interactive_comparison(self):
        """Run interactive comparison mode"""
        print(f"\n{Fore.YELLOW}=== Interactive Comparison Mode ==={Style.RESET_ALL}")
        print("Enter queries to compare both routing approaches (type 'quit' to exit):")
        print("Commands: 'stats' for current statistics, 'help' for information\n")
        
        while True:
            try:
                user_query = input(f"{Fore.CYAN}Query> {Style.RESET_ALL}")
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_query.lower() == 'stats':
                    semantic_stats = self.semantic_router.get_performance_stats()
                    routellm_stats = self.routellm.get_performance_stats()
                    print(f"\nSemantic Router: {semantic_stats['total_queries']} queries, {semantic_stats['average_latency_ms']:.2f}ms avg")
                    print(f"RouteLLM: {routellm_stats['total_queries']} queries, ${routellm_stats['total_cost']:.6f} total cost")
                    continue
                elif user_query.lower() == 'help':
                    print("\nComparison modes:")
                    print("- Semantic Router: Fast route classification")
                    print("- RouteLLM: Full LLM responses with cost optimization")
                    print("- Both approaches use real OpenAI API calls\n")
                    continue
                elif not user_query.strip():
                    continue
                
                print(f"\n{Fore.MAGENTA}Comparing: {user_query}{Style.RESET_ALL}")
                print("-" * 60)
                
                # Test both routers
                semantic_task = self.semantic_route_test(user_query)
                routellm_task = self.routellm_test(user_query)
                
                semantic_result, routellm_result = await asyncio.gather(semantic_task, routellm_task)
                
                # Display side-by-side results
                print(f"{Fore.GREEN}Semantic Router:{Style.RESET_ALL}")
                print(f"  Route: {semantic_result['route']}")
                print(f"  Latency: {semantic_result['latency_ms']:.2f}ms")
                print(f"  Cost: ${semantic_result['cost']:.6f}")
                
                print(f"{Fore.YELLOW}RouteLLM:{Style.RESET_ALL}")
                print(f"  Model: {routellm_result['model_used']}")
                if 'complexity' in routellm_result:
                    print(f"  Complexity: {routellm_result['complexity']:.3f}")
                print(f"  Latency: {routellm_result['latency_ms']:.2f}ms")
                print(f"  Cost: ${routellm_result['cost']:.6f}")
                if routellm_result['success']:
                    print(f"  Response: {routellm_result['response_preview']}")
                print()
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}\n")
        
        print(f"\n{Fore.GREEN}Comparison completed!{Style.RESET_ALL}")

async def main():
    """Main execution function"""
    try:
        # Initialize comparison suite
        comparison = OAIRouterComparison()
        
        # Run comprehensive comparison
        await comparison.compare_routers_comprehensive()
        
    except Exception as e:
        print(f"{Fore.RED}Error initializing comparison: {str(e)}{Style.RESET_ALL}")
        print("Make sure OPENAI_API_KEY is set in your environment or .env file")

if __name__ == "__main__":
    asyncio.run(main())