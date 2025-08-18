#!/usr/bin/env python3
"""
Semantic Router Demo with Azure OpenAI
Demonstrates routing with Azure-hosted models
"""

import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from semantic_router import Route
from semantic_router.encoders import AzureOpenAIEncoder
from semantic_router.layer import RouteLayer
from colorama import init, Fore, Style
import json

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

class AzureSemanticRouterDemo:
    def __init__(self):
        """Initialize Semantic Router with Azure OpenAI"""
        print(f"{Fore.CYAN}Initializing Semantic Router with Azure OpenAI...{Style.RESET_ALL}")
        
        # Initialize Azure OpenAI encoder
        self.encoder = AzureOpenAIEncoder(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        )
        
        # Define routes for Azure-specific use cases
        self.routes = self._create_azure_routes()
        
        # Create route layer
        self.router = RouteLayer(encoder=self.encoder, routes=self.routes)
        
        print(f"{Fore.GREEN}✓ Azure Semantic Router initialized{Style.RESET_ALL}")
    
    def _create_azure_routes(self) -> List[Route]:
        """Create routes optimized for Azure services integration"""
        
        # Azure Cognitive Services route
        azure_services_route = Route(
            name="azure_cognitive_services",
            utterances=[
                "Analyze this image with computer vision",
                "Translate this text to Spanish",
                "Extract text from this document",
                "Detect faces in this photo",
                "What's the sentiment of this text?",
                "Convert speech to text",
                "Generate speech from text",
                "Identify objects in image",
                "OCR this document",
                "Detect language of text",
                "Extract key phrases",
                "Recognize handwriting",
                "Analyze video content",
                "Moderate content for safety",
                "Custom vision model prediction"
            ]
        )
        
        # Azure SQL Database route
        azure_sql_route = Route(
            name="azure_sql_database",
            utterances=[
                "Query Azure SQL database",
                "Show data from SQL server",
                "Execute stored procedure",
                "Database performance metrics",
                "SQL query optimization",
                "Database backup status",
                "Show table schema",
                "Database connection string",
                "Run SQL migration",
                "Database user permissions",
                "SQL server configuration",
                "Query execution plan",
                "Database size and usage",
                "Replicate database",
                "SQL elastic pool metrics"
            ]
        )
        
        # Azure DevOps route
        azure_devops_route = Route(
            name="azure_devops",
            utterances=[
                "Show my work items",
                "Create a new bug",
                "Pipeline status",
                "Latest build results",
                "Deploy to production",
                "Pull request reviews",
                "Sprint burndown chart",
                "Code coverage report",
                "Release pipeline",
                "Repository branches",
                "Merge conflicts",
                "Test results",
                "Project backlog",
                "Team velocity",
                "CI/CD pipeline logs"
            ]
        )
        
        # Azure Blob Storage route
        azure_storage_route = Route(
            name="azure_blob_storage",
            utterances=[
                "Upload file to blob storage",
                "List files in container",
                "Download blob file",
                "Generate SAS token",
                "Container access policy",
                "Storage account metrics",
                "Blob metadata",
                "Copy blob between containers",
                "Delete expired blobs",
                "Storage tier optimization",
                "Blob versioning",
                "Storage lifecycle policy",
                "Container lease",
                "Blob snapshots",
                "Archive storage costs"
            ]
        )
        
        return [azure_services_route, azure_sql_route, azure_devops_route, azure_storage_route]
    
    def route_query_with_metadata(self, query: str, metadata: Dict = None) -> Dict[str, Any]:
        """Route query with additional metadata"""
        start_time = time.time()
        
        # Get route decision
        route_decision = self.router(query)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        result = {
            "query": query,
            "route": route_decision.name if route_decision else "no_match",
            "latency_ms": round(latency_ms, 2),
            "metadata": metadata,
            "suggested_action": self._get_suggested_action(route_decision.name if route_decision else None)
        }
        
        return result
    
    def _get_suggested_action(self, route_name: str) -> str:
        """Get suggested action based on route"""
        actions = {
            "azure_cognitive_services": "Initialize Azure Cognitive Services client",
            "azure_sql_database": "Connect to Azure SQL Database",
            "azure_devops": "Authenticate with Azure DevOps API",
            "azure_blob_storage": "Create BlobServiceClient instance",
            None: "Unable to determine appropriate Azure service"
        }
        return actions.get(route_name, "Process with default handler")
    
    def run_azure_demo(self):
        """Run Azure-specific routing demo"""
        print(f"\n{Fore.YELLOW}=== Azure Semantic Router Demo ==={Style.RESET_ALL}\n")
        
        # Azure-specific test queries
        test_queries = [
            ("Extract text from this PDF document", {"file_type": "pdf"}),
            ("Show pipeline deployment status", {"project": "MyApp"}),
            ("Query customer data from SQL database", {"database": "CustomerDB"}),
            ("Upload images to blob container", {"container": "images"}),
            ("What's the sentiment of customer reviews?", {"source": "reviews"}),
            ("Get latest build artifacts", {"branch": "main"}),
            ("Database backup completed successfully?", {"backup_id": "12345"}),
            ("List all files in documents container", {"container": "documents"}),
            ("Translate product description to French", {"target_lang": "fr"}),
            ("Sprint velocity for current iteration", {"team": "Alpha"})
        ]
        
        print(f"Testing {len(test_queries)} Azure-specific queries:\n")
        
        results = []
        
        for query, metadata in test_queries:
            result = self.route_query_with_metadata(query, metadata)
            results.append(result)
            
            # Color code by route
            route_colors = {
                "azure_cognitive_services": Fore.CYAN,
                "azure_sql_database": Fore.GREEN,
                "azure_devops": Fore.YELLOW,
                "azure_blob_storage": Fore.MAGENTA,
                "no_match": Fore.RED
            }
            
            color = route_colors.get(result['route'], Fore.WHITE)
            print(f"{color}Query: {query[:60]}...")
            print(f"  → Route: {result['route']}")
            print(f"  → Action: {result['suggested_action']}")
            print(f"  → Metadata: {json.dumps(metadata, indent=2)}")
            print(f"  → Latency: {result['latency_ms']}ms{Style.RESET_ALL}")
            print()
        
        # Calculate statistics
        total_latency = sum(r['latency_ms'] for r in results)
        avg_latency = total_latency / len(results)
        
        print(f"\n{Fore.CYAN}=== Azure Performance Metrics ==={Style.RESET_ALL}")
        print(f"Total queries: {len(test_queries)}")
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Total processing time: {total_latency:.2f}ms")
        
        # Show route distribution
        print(f"\n{Fore.CYAN}=== Azure Service Distribution ==={Style.RESET_ALL}")
        route_counts = {}
        for result in results:
            route = result['route']
            route_counts[route] = route_counts.get(route, 0) + 1
        
        for route, count in sorted(route_counts.items()):
            percentage = (count / len(test_queries)) * 100
            print(f"{route}: {count} queries ({percentage:.1f}%)")

if __name__ == "__main__":
    demo = AzureSemanticRouterDemo()
    demo.run_azure_demo()