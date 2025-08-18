class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0
    
    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds"""
        return self.elapsed_ms / 1000

class MetricsCollector:
    """Collect and analyze routing metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.costs = defaultdict(float)
        self.routes = defaultdict(int)
        self.start_time = time.time()
    
    def record_routing(self, 
                      query: str,
                      route: str,
                      latency_ms: float,
                      cost: float = 0,
                      metadata: Dict = None):
        """Record a routing decision"""
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "route": route,
            "latency_ms": latency_ms,
            "cost": cost,
            "metadata": metadata or {}
        }
        
        self.metrics["all"].append(record)
        self.metrics[route].append(record)
        self.costs[route] += cost
        self.routes[route] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        
        all_latencies = [m["latency_ms"] for m in self.metrics["all"]]
        
        if not all_latencies:
            return {"error": "No metrics collected"}
        
        stats = {
            "total_queries": len(self.metrics["all"]),
            "runtime_seconds": time.time() - self.start_time,
            "latency": {
                "mean": statistics.mean(all_latencies),
                "median": statistics.median(all_latencies),
                "stdev": statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
                "min": min(all_latencies),
                "max": max(all_latencies),
                "p95": sorted(all_latencies)[int(len(all_latencies) * 0.95)] if all_latencies else 0,
                "p99": sorted(all_latencies)[int(len(all_latencies) * 0.99)] if all_latencies else 0
            },
            "costs": {
                "total": sum(self.costs.values()),
                "by_route": dict(self.costs)
            },
            "routes": {
                "distribution": dict(self.routes),
                "most_used": max(self.routes, key=self.routes.get) if self.routes else None
            },
            "throughput": {
                "queries_per_second": len(self.metrics["all"]) / (time.time() - self.start_time)
            }
        }
        
        return stats
    
    def export_metrics(self, filename: str = "metrics.json"):
        """Export metrics to JSON file"""
        
        data = {
            "summary": self.get_statistics(),
            "detailed_metrics": [m for m in self.metrics["all"]]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics exported to {filename}")
    
    def print_summary(self):
        """Print formatted summary"""
        
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("ROUTING METRICS SUMMARY")
        print("="*60)
        
        print(f"\nTotal Queries: {stats['total_queries']}")
        print(f"Runtime: {stats['runtime_seconds']:.2f} seconds")
        print(f"Throughput: {stats['throughput']['queries_per_second']:.2f} QPS")
        
        print(f"\nLatency Statistics (ms):")
        print(f"  Mean: {stats['latency']['mean']:.2f}")
        print(f"  Median: {stats['latency']['median']:.2f}")
        print(f"  Min: {stats['latency']['min']:.2f}")
        print(f"  Max: {stats['latency']['max']:.2f}")
        print(f"  P95: {stats['latency']['p95']:.2f}")
        print(f"  P99: {stats['latency']['p99']:.2f}")
        
        print(f"\nRoute Distribution:")
        for route, count in stats['routes']['distribution'].items():
            percentage = (count / stats['total_queries']) * 100
            print(f"  {route}: {count} ({percentage:.1f}%)")
        
        print(f"\nTotal Cost: ${stats['costs']['total']:.6f}")
        if stats['costs']['by_route']:
            print("Cost by Route:")
            for route, cost in stats['costs']['by_route'].items():
                print(f"  {route}: ${cost:.6f}")
        
        print("="*60)