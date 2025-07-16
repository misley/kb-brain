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
def status():
    """Show KB Brain system status"""
    try:
        from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain
        
        brain = HybridGPUKBBrain()
        status = brain.get_system_status()
        
        console.print(Panel.fit(
            f"[bold green]KB Brain Status[/bold green]\n"
            f"GPU Available: {'✅' if status['hybrid_gpu_status']['gpu_available'] else '❌'}\n"
            f"Knowledge Embeddings: {status['hybrid_gpu_status']['knowledge_embeddings']}\n"
            f"Solution Texts: {status['hybrid_gpu_status']['solution_texts']}\n"
            f"Brain Type: {status['hybrid_gpu_status']['brain_type']}\n"
            f"Total Solutions: {status['total_solutions']}\n"
            f"KB Files: {status['kb_files_count']}\n"
            f"Last Updated: {status['last_updated']}"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@main.command()
@click.argument('query')
@click.option('--top-k', default=5, help='Number of results to return')
def search(query, top_k):
    """Search the knowledge base"""
    try:
        from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain
        
        brain = HybridGPUKBBrain()
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

if __name__ == "__main__":
    main()
