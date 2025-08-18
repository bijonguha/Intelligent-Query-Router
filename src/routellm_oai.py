#!/usr/bin/env python3
"""
RouteLLM with Direct OpenAI API Integration
Cost-optimized routing between GPT-4o-mini (strong) and GPT-4o-nano (weak) models 
based on query complexity with real OpenAI API calls
"""

import os
import time
import asyncio
import tiktoken
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from openai import OpenAI
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# Load environment variables
load_dotenv()

class DirectOpenAIRouteLLM:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize RouteLLM with direct OpenAI API integration
        
        Args:
            api_key: OpenAI API key. If None, loads from OPENAI_API_KEY environment variable
        """
        print(f"{Fore.CYAN}Initializing Direct OpenAI RouteLLM...{Style.RESET_ALL}")
        
        # Set up OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Model configuration - using latest models with updated pricing
        self.strong_model = os.getenv("ROUTELLM_STRONG_MODEL", "gpt-4o-mini")
        self.weak_model = os.getenv("ROUTELLM_WEAK_MODEL", "gpt-4o-nano")
        self.threshold = float(os.getenv("ROUTELLM_THRESHOLD", "0.7"))
        
        # Updated pricing (per 1K tokens) as of late 2024
        self.costs = {
            "gpt-4o": {"input": 0.005, "output": 0.015},           # GPT-4o
            "gpt-4": {"input": 0.03, "output": 0.06},              # GPT-4 
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},   # GPT-4o-mini
            "gpt-4o-nano": {"input": 0.000015, "output": 0.00006}, # GPT-4o-nano (estimated pricing)
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}   # GPT-3.5-turbo
        }
        
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Performance tracking
        self.total_cost = 0
        self.total_queries = 0
        self.total_latency = 0
        self.model_usage = {self.strong_model: 0, self.weak_model: 0}
        self.complexity_stats = {"high": 0, "medium": 0, "low": 0}
        
        print(f"{Fore.GREEN}✓ Direct OpenAI RouteLLM initialized")
        print(f"  Strong model: {self.strong_model}")
        print(f"  Weak model: {self.weak_model}")
        print(f"  Threshold: {self.threshold}")
        print(f"  API Key: {self.api_key[:7]}...{self.api_key[-4:]}{Style.RESET_ALL}")
    
    def classify_complexity(self, query: str) -> Dict[str, Any]:
        """
        Enhanced complexity classification with detailed analysis
        
        Args:
            query: Input query to analyze
            
        Returns:
            Dictionary with complexity score and analysis details
        """
        complexity_indicators = {
            "high": {
                "keywords": ["explain", "analyze", "compare", "evaluate", "design", 
                           "architect", "optimize", "debug", "refactor", "implement",
                           "develop", "build", "create complex", "algorithm", "system"],
                "weight": 0.8
            },
            "medium": {
                "keywords": ["how", "why", "describe", "list", "summarize", "create",
                           "generate", "write", "plan", "outline", "process"],
                "weight": 0.5
            },
            "low": {
                "keywords": ["what", "when", "where", "who", "define", "name", 
                           "count", "find", "show", "get", "is", "are"],
                "weight": 0.3
            }
        }
        
        query_lower = query.lower()
        
        # Keyword-based scoring
        scores = {}
        for level, config in complexity_indicators.items():
            score = sum(1 for word in config["keywords"] if word in query_lower)
            scores[level] = score * config["weight"]
        
        # Determine base complexity from highest scoring category
        if scores["high"] > 0:
            base_score = 0.8
            complexity_level = "high"
        elif scores["medium"] > 0:
            base_score = 0.5
            complexity_level = "medium"
        else:
            base_score = 0.3
            complexity_level = "low"
        
        # Length factor (longer queries tend to be more complex)
        length_factor = min(len(query) / 300, 0.2)
        
        # Technical terms boost
        technical_terms = ["API", "database", "algorithm", "architecture", "framework", 
                          "optimization", "performance", "scalability", "microservices"]
        tech_boost = min(sum(1 for term in technical_terms if term.lower() in query_lower) * 0.1, 0.2)
        
        # Question complexity (multiple questions = higher complexity)
        question_marks = query.count("?")
        question_factor = min(question_marks * 0.05, 0.1)
        
        # Final complexity score
        final_complexity = min(base_score + length_factor + tech_boost + question_factor, 1.0)
        
        return {
            "score": round(final_complexity, 3),
            "level": complexity_level,
            "factors": {
                "base_score": base_score,
                "length_factor": round(length_factor, 3),
                "tech_boost": round(tech_boost, 3),
                "question_factor": round(question_factor, 3)
            },
            "keyword_matches": {level: scores[level] for level in scores}
        }
    
    def count_tokens(self, text: str) -> int:
        """Accurate token counting using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    async def route_and_execute(self, query: str, max_tokens: int = 150) -> Dict[str, Any]:
        """
        Route query and execute with real OpenAI API call
        
        Args:
            query: Input query to process
            max_tokens: Maximum tokens for response
            
        Returns:
            Dictionary with routing results, response, and metrics
        """
        start_time = time.time()
        
        # Complexity analysis
        complexity_analysis = self.classify_complexity(query)
        complexity_score = complexity_analysis["score"]
        
        # Model selection based on complexity
        selected_model = self.strong_model if complexity_score >= self.threshold else self.weak_model
        
        # Update tracking
        self.total_queries += 1
        self.model_usage[selected_model] += 1
        self.complexity_stats[complexity_analysis["level"]] += 1
        
        try:
            # Make real OpenAI API call
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Provide clear, concise responses."},
                    {"role": "user", "content": query}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Calculate actual tokens used
            input_tokens = self.count_tokens(query + "You are a helpful AI assistant. Provide clear, concise responses.")
            output_tokens = self.count_tokens(response_content) if response_content else 0
            
            # Calculate actual cost
            cost = (input_tokens * self.costs[selected_model]["input"] / 1000 +
                   output_tokens * self.costs[selected_model]["output"] / 1000)
            
            self.total_cost += cost
            
            # Get usage from response if available
            if hasattr(response, 'usage') and response.usage:
                actual_input_tokens = response.usage.prompt_tokens
                actual_output_tokens = response.usage.completion_tokens
                # Use actual usage for more accurate cost calculation
                actual_cost = (actual_input_tokens * self.costs[selected_model]["input"] / 1000 +
                             actual_output_tokens * self.costs[selected_model]["output"] / 1000)
                cost = actual_cost
                input_tokens = actual_input_tokens
                output_tokens = actual_output_tokens
                self.total_cost = self.total_cost - cost + actual_cost  # Adjust total
            
            success = True
            
        except Exception as e:
            response_content = f"Error: {str(e)}"
            input_tokens = self.count_tokens(query)
            output_tokens = 0
            cost = 0
            success = False
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        self.total_latency += latency_ms
        
        return {
            "query": query,
            "response": response_content,
            "complexity": complexity_analysis,
            "model": selected_model,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "cost": round(cost, 6),
            "latency_ms": round(latency_ms, 2),
            "saved": selected_model == self.weak_model,
            "success": success,
            "timestamp": time.time()
        }
    
    async def batch_route_and_execute(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries efficiently"""
        print(f"{Fore.YELLOW}Processing {len(queries)} queries in batch...{Style.RESET_ALL}")
        
        results = []
        batch_start = time.time()
        
        for i, query in enumerate(queries):
            result = await self.route_and_execute(query)
            results.append(result)
            
            # Progress indicator for large batches
            if len(queries) > 5 and (i + 1) % 3 == 0:
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
        
        # Calculate hypothetical costs for comparison
        hypothetical_all_strong = self.total_queries * 0.002  # Approximate cost per query for strong model
        hypothetical_all_weak = self.total_queries * 0.0001   # Approximate cost per query for weak model
        
        savings_vs_all_strong = hypothetical_all_strong - self.total_cost
        savings_percent = (savings_vs_all_strong / hypothetical_all_strong * 100) if hypothetical_all_strong > 0 else 0
        
        return {
            "total_queries": self.total_queries,
            "total_cost": round(self.total_cost, 6),
            "average_cost_per_query": round(self.total_cost / self.total_queries, 6),
            "average_latency_ms": round(avg_latency, 2),
            "model_usage": dict(self.model_usage),
            "complexity_distribution": dict(self.complexity_stats),
            "cost_analysis": {
                "actual_cost": round(self.total_cost, 6),
                "hypothetical_all_strong": round(hypothetical_all_strong, 6),
                "hypothetical_all_weak": round(hypothetical_all_weak, 6),
                "savings_vs_all_strong": round(savings_vs_all_strong, 6),
                "savings_percent": round(savings_percent, 2)
            },
            "efficiency": {
                "weak_model_usage_percent": round((self.model_usage[self.weak_model] / self.total_queries * 100), 2),
                "strong_model_usage_percent": round((self.model_usage[self.strong_model] / self.total_queries * 100), 2)
            }
        }
    
    async def run_comprehensive_demo(self):
        """Run comprehensive RouteLLM demonstration"""
        print(f"\n{Fore.YELLOW}=== Direct OpenAI RouteLLM Demo ==={Style.RESET_ALL}\n")
        
        # Test queries with varying complexity levels
        test_queries = [
            # Low complexity queries
            "What is the capital of France?",
            "Who invented the telephone?",
            "Define machine learning",
            
            # Medium complexity queries  
            "How does a neural network work?",
            "List the benefits of cloud computing",
            "Describe the HTTP protocol",
            
            # High complexity queries
            "Analyze the architectural trade-offs between microservices and monolithic applications",
            "Design a scalable real-time data processing pipeline for IoT devices",
            "Compare and evaluate different database consistency models for distributed systems",
            "Implement an optimization algorithm for resource allocation in cloud environments"
        ]
        
        print(f"Processing {len(test_queries)} queries with complexity-based routing:\n")
        print("-" * 80)
        
        # Process queries
        results = await self.batch_route_and_execute(test_queries)
        
        # Display results
        for result in results:
            color = Fore.GREEN if result['saved'] else Fore.YELLOW
            
            print(f"\n{color}Query: {result['query'][:70]}...")
            print(f"  → Complexity: {result['complexity']['score']} ({result['complexity']['level']})")
            print(f"  → Model: {result['model']}")
            print(f"  → Tokens: {result['tokens']['input']} in, {result['tokens']['output']} out")
            print(f"  → Cost: ${result['cost']:.6f}")
            print(f"  → Latency: {result['latency_ms']}ms")
            print(f"  → Saved: {'✓' if result['saved'] else '✗'}")
            if result['response'] and len(result['response']) > 100:
                print(f"  → Response: {result['response'][:100]}...{Style.RESET_ALL}")
            else:
                print(f"  → Response: {result['response']}{Style.RESET_ALL}")
        
        # Display comprehensive statistics
        self._display_performance_summary()
        
        # Interactive mode
        await self._run_interactive_mode()
    
    def _display_performance_summary(self):
        """Display detailed performance summary"""
        stats = self.get_performance_stats()
        
        print(f"\n{Fore.CYAN}=== Performance Summary ==={Style.RESET_ALL}")
        print(f"Total queries processed: {stats['total_queries']}")
        print(f"Total cost: ${stats['total_cost']:.6f}")
        print(f"Average cost per query: ${stats['average_cost_per_query']:.6f}")
        print(f"Average latency: {stats['average_latency_ms']:.2f}ms")
        
        print(f"\n{Fore.CYAN}=== Model Usage Distribution ==={Style.RESET_ALL}")
        for model, count in stats['model_usage'].items():
            percentage = (count / stats['total_queries']) * 100
            model_type = "Strong" if model == self.strong_model else "Weak"
            print(f"{model_type} model ({model}): {count} queries ({percentage:.1f}%)")
        
        print(f"\n{Fore.CYAN}=== Complexity Distribution ==={Style.RESET_ALL}")
        for level, count in stats['complexity_distribution'].items():
            percentage = (count / stats['total_queries']) * 100
            print(f"{level.capitalize()} complexity: {count} queries ({percentage:.1f}%)")
        
        print(f"\n{Fore.CYAN}=== Cost Analysis ==={Style.RESET_ALL}")
        cost_analysis = stats['cost_analysis']
        print(f"Actual cost: ${cost_analysis['actual_cost']:.6f}")
        print(f"Cost if all {self.strong_model}: ${cost_analysis['hypothetical_all_strong']:.6f}")
        print(f"Cost if all {self.weak_model}: ${cost_analysis['hypothetical_all_weak']:.6f}")
        print(f"Savings vs all strong: ${cost_analysis['savings_vs_all_strong']:.6f} ({cost_analysis['savings_percent']:.1f}%)")
        
        print(f"\n{Fore.CYAN}=== Efficiency Metrics ==={Style.RESET_ALL}")
        efficiency = stats['efficiency']
        print(f"Weak model usage: {efficiency['weak_model_usage_percent']:.1f}%")
        print(f"Strong model usage: {efficiency['strong_model_usage_percent']:.1f}%")
    
    async def _run_interactive_mode(self):
        """Run interactive query testing mode"""
        print(f"\n{Fore.YELLOW}=== Interactive Mode ==={Style.RESET_ALL}")
        print("Enter queries to test routing (type 'quit', 'exit', or 'q' to stop):")
        print("Commands: 'stats' for statistics, 'models' for model info\n")
        
        while True:
            try:
                user_query = input(f"{Fore.CYAN}Query> {Style.RESET_ALL}")
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_query.lower() == 'stats':
                    self._display_performance_summary()
                    continue
                elif user_query.lower() == 'models':
                    print(f"\nAvailable models:")
                    print(f"  Strong: {self.strong_model} (${self.costs[self.strong_model]['input']:.4f} in, ${self.costs[self.strong_model]['output']:.4f} out per 1K tokens)")
                    print(f"  Weak: {self.weak_model} (${self.costs[self.weak_model]['input']:.4f} in, ${self.costs[self.weak_model]['output']:.4f} out per 1K tokens)")
                    print(f"  Threshold: {self.threshold}\n")
                    continue
                elif not user_query.strip():
                    continue
                
                result = await self.route_and_execute(user_query)
                color = Fore.GREEN if result['saved'] else Fore.YELLOW
                
                print(f"{color}  → Complexity: {result['complexity']['score']} ({result['complexity']['level']})")
                print(f"  → Model: {result['model']}")
                print(f"  → Cost: ${result['cost']:.6f}")
                print(f"  → Latency: {result['latency_ms']}ms")
                print(f"  → Response: {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}{Style.RESET_ALL}\n")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}\n")
        
        print(f"\n{Fore.GREEN}Demo completed! Final statistics:{Style.RESET_ALL}")
        self._display_performance_summary()

def main():
    """Main execution function"""
    try:
        # Initialize RouteLLM
        route_llm = DirectOpenAIRouteLLM()
        
        # Run comprehensive demo
        asyncio.run(route_llm.run_comprehensive_demo())
        
    except Exception as e:
        print(f"{Fore.RED}Error initializing RouteLLM: {str(e)}{Style.RESET_ALL}")
        print("Make sure OPENAI_API_KEY is set in your environment or .env file")

if __name__ == "__main__":
    main()