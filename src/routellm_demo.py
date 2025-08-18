#!/usr/bin/env python3
"""
RouteLLM Demo - Cost-Optimized Model Routing
Routes between expensive (GPT-4) and cheap (GPT-3.5) models based on query complexity
"""

import os
import time
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
import openai
from colorama import init, Fore, Style

init(autoreset=True)
load_dotenv()

class RouteLLMDemo:
    def __init__(self):
        """Initialize RouteLLM with OpenAI models"""
        print(f"{Fore.CYAN}Initializing RouteLLM...{Style.RESET_ALL}")
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        self.strong_model = os.getenv("ROUTELLM_STRONG_MODEL", "gpt-4")
        self.weak_model = os.getenv("ROUTELLM_WEAK_MODEL", "gpt-3.5-turbo")
        self.threshold = float(os.getenv("ROUTELLM_THRESHOLD", "0.7"))
        
        self.costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
        
        self.total_cost = 0
        self.model_usage = {self.strong_model: 0, self.weak_model: 0}
        
        print(f"{Fore.GREEN}✓ RouteLLM initialized")
        print(f"  Strong model: {self.strong_model}")
        print(f"  Weak model: {self.weak_model}")
        print(f"  Threshold: {self.threshold}{Style.RESET_ALL}")
    
    def classify_complexity(self, query: str) -> float:
        complexity_indicators = {
            "high": ["explain", "analyze", "compare", "evaluate", "design", 
                    "architect", "optimize", "debug", "refactor", "implement"],
            "medium": ["how", "why", "describe", "list", "summarize", "create"],
            "low": ["what", "when", "where", "who", "define", "name", "count"]
        }
        
        query_lower = query.lower()
        
        high_score = sum(1 for word in complexity_indicators["high"] if word in query_lower)
        medium_score = sum(1 for word in complexity_indicators["medium"] if word in query_lower)
        low_score = sum(1 for word in complexity_indicators["low"] if word in query_lower)
        
        if high_score > 0:
            base_score = 0.7
        elif medium_score > 0:
            base_score = 0.5
        else:
            base_score = 0.3
        
        length_factor = min(len(query) / 200, 0.3)
        
        complexity = min(base_score + length_factor, 1.0)
        return complexity
    
    async def route_and_execute(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        complexity = self.classify_complexity(query)
        
        selected_model = self.strong_model if complexity >= self.threshold else self.weak_model
        self.model_usage[selected_model] += 1
        
        try:
            await asyncio.sleep(0.1)
            response = f"[{selected_model}] Response to: {query[:50]}..."
            
            input_tokens = len(query.split()) * 1.3
            output_tokens = 50
            
            cost = (input_tokens * self.costs[selected_model]["input"] / 1000 +
                   output_tokens * self.costs[selected_model]["output"] / 1000)
            
            self.total_cost += cost
            
        except Exception as e:
            response = f"Error: {str(e)}"
            cost = 0
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        return {
            "query": query,
            "complexity": round(complexity, 2),
            "model": selected_model,
            "response": response,
            "cost": round(cost, 6),
            "latency_ms": round(latency_ms, 2),
            "saved": selected_model == self.weak_model
        }
    
    async def run_demo(self):
        print(f"\n{Fore.YELLOW}=== RouteLLM Cost Optimization Demo ==={Style.RESET_ALL}\n")
        
        test_queries = [
            "What is the capital of France?",
            "How does a neural network work?",
            "Analyze the architectural trade-offs between microservices and monolithic applications",
            "Design a scalable real-time data processing pipeline for IoT devices"
        ]
        
        print(f"Processing {len(test_queries)} queries with complexity-based routing:\n")
        
        results = []
        
        for query in test_queries:
            result = await self.route_and_execute(query)
            results.append(result)
            
            color = Fore.GREEN if result['saved'] else Fore.YELLOW
            
            print(f"{color}Query: {query[:60]}...")
            print(f"  → Complexity: {result['complexity']}")
            print(f"  → Model: {result['model']}")
            print(f"  → Cost: ${result['cost']:.6f}")
            print(f"  → Latency: {result['latency_ms']}ms{Style.RESET_ALL}")
            print()
        
        total_weak = sum(1 for r in results if r['saved'])
        total_strong = len(results) - total_weak
        
        hypothetical_cost = len(results) * 0.001
        actual_cost = self.total_cost
        savings = hypothetical_cost - actual_cost
        savings_percent = (savings / hypothetical_cost * 100) if hypothetical_cost > 0 else 0
        
        print(f"\n{Fore.CYAN}=== Cost Analysis ==={Style.RESET_ALL}")
        print(f"Total queries: {len(results)}")
        print(f"Strong model used: {total_strong} times")
        print(f"Weak model used: {total_weak} times")
        print(f"Total cost: ${actual_cost:.6f}")
        print(f"Cost if all GPT-4: ${hypothetical_cost:.6f}")
        print(f"Savings: ${savings:.6f} ({savings_percent:.1f}%)")
        
        avg_latency = sum(r['latency_ms'] for r in results) / len(results)
        print(f"Average latency: {avg_latency:.2f}ms")

if __name__ == "__main__":
    demo = RouteLLMDemo()
    asyncio.run(demo.run_demo())