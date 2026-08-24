from rich.console import Console
from rich.table import Table
from src.dataset import load_images_from_directory
from src.pipeline import run_batch_inference_pipeline

console = Console()


def print_colored_summary_table(df_summary):
    """Renders terminal table using Rich library with class-specific styling."""
    if df_summary.empty:
        console.print("[yellow]No objects detected above confidence threshold.[/yellow]")
        return

    table = Table(title="--- Summary Detection Report ---", show_header=True, header_style="bold magenta")
    table.add_column("Image Name", style="cyan")
    table.add_column("Class Name", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Bounding Box Color", justify="center")

    color_styles = {
        "bus": "[bold blue]Blue[/bold blue]",
        "person": "[bold green]Green[/bold green]",
        "car": "[bold yellow]Yellow[/bold yellow]",
        "frisbee": "[bold magenta]Magenta[/bold magenta]",
        "kite": "[bold red]Orange[/bold red]",
    }

    for _, row in df_summary.iterrows():
        cls_name = row["class_name"]
        color_tag = color_styles.get(cls_name, "[white]Default Green[/white]")
        
        table.add_row(
            str(row["image_name"]),
            f"[bold]{cls_name}[/bold]",
            f"{row['confidence']:.4f}",
            color_tag
        )

    console.print(table)


def main():
    image_paths = load_images_from_directory("sample_images")

    if not image_paths:
        console.print("[red]⚠️ No images found in 'sample_images/'. Run sample generators first.[/red]")
        return

    df_summary = run_batch_inference_pipeline(image_paths, conf_threshold=0.25)
    print_colored_summary_table(df_summary)


if __name__ == "__main__":
    main()