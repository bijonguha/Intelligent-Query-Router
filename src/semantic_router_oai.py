#!/usr/bin/env python3
"""
Semantic Router with Direct OpenAI API Integration
Demonstrates ultra-fast routing using OpenAI embeddings with direct API calls
"""

import os
import time
import openai
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder
from colorama import init, Fore, Style
import json

# Initialize colorama for colored output
init(autoreset=True)

# Load environment variables
load_dotenv()

class DirectOpenAISemanticRouter:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Router with direct OpenAI API integration
        
        Args:
            api_key: OpenAI API key. If None, loads from OPENAI_API_KEY environment variable
        """
        print(f"{Fore.CYAN}Initializing Direct OpenAI Semantic Router...{Style.RESET_ALL}")
        
        # Set up OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        openai.api_key = self.api_key
        self.client = openai
        
        # Initialize encoder with direct API key
        self.encoder = OpenAIEncoder(
            openai_api_key=self.api_key,
            name="text-embedding-3-small"  # Latest, faster embedding model
        )
        
        # Define comprehensive routes
        self.routes = self._create_comprehensive_routes()
        
        # Create semantic router with auto initialization
        print(f"{Fore.YELLOW}Building route index...{Style.RESET_ALL}")
        self.router = SemanticRouter(encoder=self.encoder, routes=self.routes)
        
        # Add routes and ensure index is ready
        if hasattr(self.router, 'add'):
            for route in self.routes:
                self.router.add(route)
        
        # Check if index is ready, if not, try to initialize it
        if hasattr(self.router, 'index') and self.router.index is not None:
            if hasattr(self.router.index, 'sync'):
                self.router.index.sync()
            elif hasattr(self.router.index, '_init'):
                self.router.index._init()
        
        # Performance tracking
        self.total_queries = 0
        self.total_latency = 0
        self.route_usage = {}
        
        print(f"{Fore.GREEN}✓ Direct OpenAI Semantic Router initialized")
        print(f"  Model: text-embedding-3-small")
        print(f"  Routes: {len(self.routes)}")
        print(f"  API Key: {self.api_key[:7]}...{self.api_key[-4:]}{Style.RESET_ALL}")
    
    def _create_comprehensive_routes(self) -> List[Route]:
        """Create comprehensive routing definitions optimized for various use cases"""
        
        # Technical/Development route
        technical_route = Route(
            name="technical_development",
            utterances=[
                "Debug this Python code",
                "Optimize algorithm performance",
                "Review code architecture", 
                "Fix memory leak issue",
                "Implement REST API endpoint",
                "Design database schema",
                "Setup CI/CD pipeline",
                "Configure Docker container",
                "Write unit tests",
                "Refactor legacy code",
                "Analyze system bottlenecks",
                "Deploy to production",
                "Monitor application performance",
                "Handle API rate limiting",
                "Implement caching strategy",
                "Setup load balancing",
                "Configure SSL certificates",
                "Migrate database",
                "Scale microservices",
                "Troubleshoot network issues"
            ]
        )
        
        # Business Analytics route
        analytics_route = Route(
            name="business_analytics",
            utterances=[
                "Calculate quarterly revenue",
                "Analyze customer churn rate",
                "Show sales performance metrics",
                "Generate financial reports",
                "Track user engagement",
                "Measure conversion rates",
                "Compare year-over-year growth",
                "Segment customer demographics",
                "Forecast demand trends",
                "Analyze market share",
                "Calculate ROI metrics",
                "Monitor KPI dashboard",
                "Evaluate campaign performance",
                "Assess product profitability",
                "Track inventory turnover",
                "Measure customer lifetime value",
                "Analyze pricing strategies",
                "Review operational efficiency",
                "Study competitive analysis",
                "Evaluate market opportunities"
            ]
        )
        
        # Customer Service route
        customer_service_route = Route(
            name="customer_service",
            utterances=[
                "Help with order issue",
                "Process refund request",
                "Track shipping status",
                "Reset account password",
                "Cancel subscription",
                "Update billing information",
                "Report product defect",
                "Schedule service appointment",
                "Escalate to supervisor",
                "Change delivery address",
                "Apply discount code",
                "Extend warranty coverage",
                "Access order history",
                "Update contact information",
                "Request technical support",
                "File complaint",
                "Check account balance",
                "Modify subscription plan",
                "Download invoice",
                "Contact customer care"
            ]
        )
        
        # Content Creation route
        content_route = Route(
            name="content_creation",
            utterances=[
                "Write blog post",
                "Create marketing copy",
                "Generate product descriptions",
                "Draft email newsletter",
                "Compose social media posts",
                "Write technical documentation",
                "Create user manual",
                "Generate press release",
                "Write video script",
                "Create presentation slides",
                "Draft proposal document",
                "Write case study",
                "Create FAQ content",
                "Generate ad copy",
                "Write website content",
                "Create training materials",
                "Draft legal terms",
                "Write product reviews",
                "Create survey questions",
                "Generate report summary"
            ]
        )
        
        # Research & Analysis route
        research_route = Route(
            name="research_analysis",
            utterances=[
                "Research market trends",
                "Analyze competitor strategies",
                "Study industry reports",
                "Investigate user behavior",
                "Examine data patterns",
                "Review academic papers",
                "Survey customer feedback",
                "Analyze survey results",
                "Study best practices",
                "Research new technologies",
                "Investigate security threats",
                "Analyze performance data",
                "Study user demographics",
                "Research regulatory changes",
                "Examine cost structures",
                "Investigate failure causes",
                "Research optimization opportunities",
                "Study workflow efficiency",
                "Analyze risk factors",
                "Research implementation options"
            ]
        )
        
        # General Conversation route
        general_route = Route(
            name="general_conversation",
            utterances=[
                "Hello, how are you?",
                "What can you help me with?",
                "Tell me about your capabilities",
                "What's the weather today?",
                "Thank you for your help",
                "How does this work?",
                "Can you explain this to me?",
                "What are my options?",
                "I need some guidance",
                "Can you recommend something?",
                "What should I do next?",
                "How do I get started?",
                "Is this the right approach?",
                "Can you clarify this?",
                "What's the best way to proceed?",
                "I'm not sure what to choose",
                "Can you provide more details?",
                "What are the alternatives?",
                "How long will this take?",
                "Is there a better solution?"
            ]
        )
        
        return [
            technical_route, 
            analytics_route, 
            customer_service_route,
            content_route,
            research_route,
            general_route
        ]
    
    def route_query(self, query: str, include_confidence: bool = True) -> Dict[str, Any]:
        """
        Route a single query and return detailed results
        
        Args:
            query: The input query to route
            include_confidence: Whether to include confidence scores
            
        Returns:
            Dictionary with routing results and metadata
        """
        start_time = time.time()
        
        # Get route decision
        try:
            route_decision = self.router(query)
        except Exception as e:
            print(f"Error during routing: {e}")
            route_decision = None
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Update tracking
        self.total_queries += 1
        self.total_latency += latency_ms
        
        # Handle different response formats
        if route_decision:
            if hasattr(route_decision, 'name'):
                route_name = route_decision.name
            elif isinstance(route_decision, str):
                route_name = route_decision
            elif hasattr(route_decision, 'route') and route_decision.route:
                route_name = route_decision.route.name if hasattr(route_decision.route, 'name') else str(route_decision.route)
            else:
                route_name = str(route_decision)
        else:
            route_name = "no_match"
            
        self.route_usage[route_name] = self.route_usage.get(route_name, 0) + 1
        
        result = {
            "query": query,
            "route": route_name,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
            "success": route_decision is not None
        }
        
        # Add confidence score if available and requested
        if include_confidence and route_decision and hasattr(route_decision, 'score'):
            result["confidence"] = round(route_decision.score, 3)
        
        return result
    
    def batch_route(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Route multiple queries efficiently
        
        Args:
            queries: List of queries to route
            
        Returns:
            List of routing results
        """
        print(f"{Fore.YELLOW}Processing {len(queries)} queries in batch...{Style.RESET_ALL}")
        
        results = []
        batch_start = time.time()
        
        for i, query in enumerate(queries):
            result = self.route_query(query)
            results.append(result)
            
            # Progress indicator for large batches
            if len(queries) > 10 and (i + 1) % 5 == 0:
                progress = ((i + 1) / len(queries)) * 100
                print(f"  Progress: {progress:.1f}% ({i + 1}/{len(queries)})")
        
        batch_time = (time.time() - batch_start) * 1000
        avg_latency = batch_time / len(queries)
        
        print(f"{Fore.GREEN}✓ Batch completed in {batch_time:.2f}ms")
        print(f"  Average latency: {avg_latency:.2f}ms per query{Style.RESET_ALL}")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if self.total_queries == 0:
            return {"error": "No queries processed yet"}
        
        avg_latency = self.total_latency / self.total_queries
        
        return {
            "total_queries": self.total_queries,
            "total_latency_ms": round(self.total_latency, 2),
            "average_latency_ms": round(avg_latency, 2),
            "routes_used": len(self.route_usage),
            "route_distribution": dict(self.route_usage),
            "most_used_route": max(self.route_usage, key=self.route_usage.get) if self.route_usage else None,
            "success_rate": round((1 - self.route_usage.get("no_match", 0) / self.total_queries) * 100, 2)
        }
    
    def run_comprehensive_demo(self):
        """Run comprehensive demonstration with various query types"""
        print(f"\n{Fore.YELLOW}=== Direct OpenAI Semantic Router Demo ==={Style.RESET_ALL}\n")
        
        # Comprehensive test queries covering all routes
        test_queries = [
            # Technical queries
            "Debug this async function that's causing deadlocks",
            "Optimize SQL query performance for large datasets",
            "Setup Docker container for microservice deployment",
            
            # Analytics queries  
            "Calculate customer acquisition cost for Q4",
            "Analyze user retention rates by cohort",
            "Generate monthly revenue growth report",
            
            # Customer service queries
            "Help customer with failed payment issue",
            "Process return for damaged product",
            "Schedule technical support call",
            
            # Content creation queries
            "Write product description for new smartphone",
            "Create social media post for product launch",
            "Draft email newsletter for monthly update",
            
            # Research queries
            "Research competitor pricing strategies",
            "Analyze market trends in AI industry", 
            "Study user feedback for feature improvements",
            
            # General conversation
            "What are the best practices for this?",
            "Can you help me understand this concept?",
            "What would you recommend in this situation?"
        ]
        
        print(f"Testing {len(test_queries)} diverse queries:\n")
        print("-" * 80)
        
        # Process queries and display results
        results = []
        
        for query in test_queries:
            result = self.route_query(query)
            results.append(result)
            
            # Color coding by route
            route_colors = {
                "technical_development": Fore.BLUE,
                "business_analytics": Fore.GREEN,
                "customer_service": Fore.YELLOW,
                "content_creation": Fore.MAGENTA,
                "research_analysis": Fore.CYAN,
                "general_conversation": Fore.WHITE,
                "no_match": Fore.RED
            }
            
            color = route_colors.get(result['route'], Fore.WHITE)
            
            print(f"{color}Query: {query[:65]}...")
            print(f"  → Route: {result['route']}")
            print(f"  → Latency: {result['latency_ms']}ms")
            if 'confidence' in result:
                print(f"  → Confidence: {result['confidence']}")
            print(f"  → Success: {'✓' if result['success'] else '✗'}{Style.RESET_ALL}")
            print()
        
        # Display comprehensive statistics
        self._display_performance_summary()
        
        # Interactive mode
        self._run_interactive_mode(route_colors)
    
    def _display_performance_summary(self):
        """Display detailed performance summary"""
        stats = self.get_performance_stats()
        
        print(f"\n{Fore.CYAN}=== Performance Summary ==={Style.RESET_ALL}")
        print(f"Total queries processed: {stats['total_queries']}")
        print(f"Average latency: {stats['average_latency_ms']:.2f}ms")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Routes utilized: {stats['routes_used']}/{len(self.routes)}")
        
        if stats['most_used_route']:
            print(f"Most used route: {stats['most_used_route']}")
        
        print(f"\n{Fore.CYAN}=== Route Distribution ==={Style.RESET_ALL}")
        for route, count in sorted(stats['route_distribution'].items()):
            percentage = (count / stats['total_queries']) * 100
            print(f"{route}: {count} queries ({percentage:.1f}%)")
    
    def _run_interactive_mode(self, route_colors: Dict[str, str]):
        """Run interactive query testing mode"""
        print(f"\n{Fore.YELLOW}=== Interactive Mode ==={Style.RESET_ALL}")
        print("Enter queries to test routing (type 'quit', 'exit', or 'q' to stop):")
        print("Commands: 'stats' for statistics, 'routes' for route info\n")
        
        while True:
            try:
                user_query = input(f"{Fore.CYAN}Query> {Style.RESET_ALL}")
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_query.lower() == 'stats':
                    self._display_performance_summary()
                    continue
                elif user_query.lower() == 'routes':
                    self._display_route_info()
                    continue
                elif not user_query.strip():
                    continue
                
                result = self.route_query(user_query)
                color = route_colors.get(result['route'], Fore.WHITE)
                
                print(f"{color}  → Route: {result['route']}")
                print(f"  → Latency: {result['latency_ms']}ms")
                if 'confidence' in result:
                    print(f"  → Confidence: {result['confidence']}")
                print(f"  → Success: {'✓' if result['success'] else '✗'}{Style.RESET_ALL}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}\n")
        
        print(f"\n{Fore.GREEN}Demo completed! Final statistics:{Style.RESET_ALL}")
        self._display_performance_summary()
    
    def _display_route_info(self):
        """Display information about available routes"""
        print(f"\n{Fore.CYAN}=== Available Routes ==={Style.RESET_ALL}")
        route_descriptions = {
            "technical_development": "Code debugging, system architecture, deployment",
            "business_analytics": "Revenue analysis, KPIs, performance metrics", 
            "customer_service": "Support requests, refunds, account issues",
            "content_creation": "Writing, marketing copy, documentation",
            "research_analysis": "Market research, data analysis, investigations",
            "general_conversation": "General questions and conversations"
        }
        
        for route in self.routes:
            description = route_descriptions.get(route.name, "No description available")
            example_utterances = route.utterances[:3]  # Show first 3 examples
            
            print(f"\n{Fore.YELLOW}{route.name}{Style.RESET_ALL}")
            print(f"  Description: {description}")
            print(f"  Examples: {', '.join(example_utterances)}")
            print(f"  Total training utterances: {len(route.utterances)}")

def main():
    """Main execution function"""
    try:
        # Initialize router
        router = DirectOpenAISemanticRouter()
        
        # Run comprehensive demo
        router.run_comprehensive_demo()
        
    except Exception as e:
        print(f"{Fore.RED}Error initializing router: {str(e)}{Style.RESET_ALL}")
        print("Make sure OPENAI_API_KEY is set in your environment or .env file")

if __name__ == "__main__":
    main()