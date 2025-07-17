#!/usr/bin/env python3
"""
KB Brain CLI - Main entry point
"""

import click
from rich.console import Console
from rich.panel import Panel

console = Console()

@click.group()
@click.version_option(version="1.0.0")
def main():
    """KB Brain - Intelligent Knowledge Management System"""
    pass

@main.command()
@click.option('--optimize-performance', is_flag=True, default=False, help='Enable performance optimizations (Intel extensions + JIT compilation)')
def status(optimize_performance):
    """Show KB Brain system status"""
    try:
        from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain
        
        brain = HybridGPUKBBrain(enable_performance_optimizations=optimize_performance)
        status = brain.get_system_status()
        
        # Performance optimization status
        perf_status = ""
        if 'performance_optimizations' in status:
            perf_info = status['performance_optimizations']
            perf_status = f"\nPerformance Optimizations: {'✅ Enabled' if optimize_performance else '❌ Disabled'}"
            if optimize_performance and perf_info:
                perf_status += f"\nIntel Extensions: {'✅' if perf_info.get('intel_extensions_available') else '❌'}"
                perf_status += f"\nJIT Compilation: {'✅' if perf_info.get('numba_available') else '❌'}"
                if 'performance_tests' in perf_info:
                    speedup = perf_info['performance_tests'].get('cpu_speedup_factor', 1.0)
                    perf_status += f"\nCPU Speedup: {speedup:.1f}x"
        
        console.print(Panel.fit(
            f"[bold green]KB Brain Status[/bold green]\n"
            f"GPU Available: {'✅' if status['hybrid_gpu_status']['gpu_available'] else '❌'}\n"
            f"Knowledge Embeddings: {status['hybrid_gpu_status']['knowledge_embeddings']}\n"
            f"Solution Texts: {status['hybrid_gpu_status']['solution_texts']}\n"
            f"Brain Type: {status['hybrid_gpu_status']['brain_type']}\n"
            f"Total Solutions: {status['total_solutions']}\n"
            f"KB Files: {status['kb_files_count']}\n"
            f"Last Updated: {status['last_updated']}{perf_status}"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@main.command()
@click.argument('query')
@click.option('--top-k', default=5, help='Number of results to return')
@click.option('--optimize-performance', is_flag=True, default=False, help='Enable performance optimizations (Intel extensions + JIT compilation)')
def search(query, top_k, optimize_performance):
    """Search the knowledge base"""
    try:
        from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain
        
        if optimize_performance:
            console.print("🚀 Performance optimizations enabled")
        
        brain = HybridGPUKBBrain(enable_performance_optimizations=optimize_performance)
        solutions = brain.find_best_solution_hybrid(query, top_k=top_k)
        
        console.print(f"[bold blue]Search Results for: {query}[/bold blue]")
        
        for i, solution in enumerate(solutions, 1):
            console.print(Panel(
                f"[bold]Solution {i}[/bold]\n"
                f"Similarity: {solution.similarity_score:.3f}\n"
                f"Confidence: {solution.confidence:.3f}\n"
                f"Source: {solution.source_kb}\n"
                f"Text: {solution.solution_text[:200]}..."
            ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@main.command()
@click.option('--optimize-performance', is_flag=True, default=False, help='Enable performance optimizations')
def rebuild_index(optimize_performance):
    """Rebuild the knowledge base index"""
    try:
        from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain
        
        if optimize_performance:
            console.print("🚀 Performance optimizations enabled for index rebuild")
        
        brain = HybridGPUKBBrain(enable_performance_optimizations=optimize_performance)
        console.print("🔄 Rebuilding knowledge index...")
        
        brain.rebuild_knowledge_index()
        
        console.print("✅ Knowledge index rebuilt successfully")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@main.command()
def benchmark():
    """Run performance benchmarks"""
    try:
        console.print("🧪 Running KB Brain performance benchmarks...")
        
        # Test without optimizations
        console.print("\n📊 Testing without performance optimizations...")
        from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain
        import time
        
        brain_standard = HybridGPUKBBrain(enable_performance_optimizations=False)
        start_time = time.time()
        solutions_standard = brain_standard.find_best_solution_hybrid("test query", top_k=3)
        standard_time = time.time() - start_time
        
        # Test with optimizations
        console.print("📊 Testing with performance optimizations...")
        brain_optimized = HybridGPUKBBrain(enable_performance_optimizations=True)
        start_time = time.time()
        solutions_optimized = brain_optimized.find_best_solution_hybrid("test query", top_k=3)
        optimized_time = time.time() - start_time
        
        # Calculate speedup
        speedup = standard_time / optimized_time if optimized_time > 0 else 1.0
        
        console.print(Panel.fit(
            f"[bold green]Performance Benchmark Results[/bold green]\n"
            f"Standard Mode: {standard_time:.3f}s\n"
            f"Optimized Mode: {optimized_time:.3f}s\n"
            f"Speedup Factor: {speedup:.1f}x\n"
            f"Results Found: {len(solutions_optimized)}"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@main.command()
@click.option('--optimize-performance', is_flag=True, default=True, help='Enable performance optimizations (default: enabled)')
def interactive(optimize_performance):
    """Start interactive KB Brain session"""
    try:
        from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain
        
        console.print("🧠 Starting KB Brain interactive session")
        console.print("Type 'quit' to exit, 'help' for commands")
        
        if optimize_performance:
            console.print("🚀 Performance optimizations enabled")
        
        brain = HybridGPUKBBrain(enable_performance_optimizations=optimize_performance)
        
        while True:
            try:
                query = input("\nKB Brain> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("👋 Goodbye!")
                    break
                elif query.lower() in ['help', '?']:
                    console.print(Panel(
                        "[bold]Available Commands:[/bold]\n"
                        "• Enter any query to search the knowledge base\n"
                        "• 'help' or '?' - Show this help\n"
                        "• 'status' - Show system status\n"
                        "• 'quit', 'exit', or 'q' - Exit interactive mode"
                    ))
                    continue
                elif query.lower() == 'status':
                    status = brain.get_system_status()
                    console.print(f"Knowledge embeddings: {status['hybrid_gpu_status']['knowledge_embeddings']}")
                    console.print(f"GPU available: {'✅' if status['hybrid_gpu_status']['gpu_available'] else '❌'}")
                    continue
                elif not query:
                    continue
                
                # Search query
                console.print(f"🔍 Searching for: {query}")
                solutions = brain.find_best_solution_hybrid(query, top_k=3)
                
                if solutions:
                    for i, solution in enumerate(solutions, 1):
                        console.print(Panel(
                            f"[bold]Result {i}[/bold]\n"
                            f"Similarity: {solution.similarity_score:.3f}\n"
                            f"Confidence: {solution.confidence:.3f}\n"
                            f"Text: {solution.solution_text[:150]}..."
                        ))
                else:
                    console.print("❌ No results found")
                    
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!")
                break
            except EOFError:
                console.print("\n👋 Goodbye!")
                break
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()
