# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich import box

import config

console = Console()

def get_system_stats():
    """Sistem kaynaklarını ve internet performansını simüle eder (Psutil eklenebilir)."""
    # Gerçek uygulamada psutil kullanılmalı
    return {
        "cpu": "24.5%",
        "ram": "12.2 GB / 32 GB",
        "gpu_vram": "4.8 GB / 24 GB",
        "net_up": "12.4 Mbps",
        "net_down": "45.1 Mbps",
        "api_latency": "124ms"
    }

def get_user_stats():
    """Kullanıcı verilerini data/users altından sayar."""
    users = []
    base_path = Path(config.BASE_PATH)
    if base_path.exists():
        for user_dir in base_path.iterdir():
            if user_dir.is_dir():
                # Kayıtlı kelime sayısını sayalım
                words_dir = user_dir / "words"
                count = 0
                if words_dir.exists():
                    count = sum(len(list(d.glob("*.wav"))) for d in words_dir.iterdir() if d.is_dir())
                
                # Model var mı?
                model_exists = (Path("data/models/personalized_models") / user_dir.name / "adapter_model.safetensors").exists()
                
                users.append({
                    "name": user_dir.name,
                    "recordings": count,
                    "status": "Ready" if model_exists else "Training Needed",
                    "last_active": "Today"
                })
    return users

def create_layout():
    """Dashboard yerleşimi: Üst (Sistem), Orta (Kullanıcılar), Alt (Loglar)."""
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=10)
    )
    layout["body"].split_row(
        Layout(name="users", ratio=2),
        Layout(name="system", ratio=1)
    )
    return layout

def generate_header():
    return Panel(
        f"[bold cyan]PRONOUNS AI - ADMIN CONTROL CENTER[/] | [white]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]",
        box=box.ROUNDED,
        style="on blue"
    )

def generate_user_table():
    table = Table(title="Aktif Kullanıcılar ve Veri Durumu", expand=True)
    table.add_column("Kullanıcı", style="magenta")
    table.add_column("Kayıt Sayısı", justify="right")
    table.add_column("Model Durumu", justify="center")
    table.add_column("Son Aktivite", justify="right")
    
    for u in get_user_stats():
        status_style = "green" if u["status"] == "Ready" else "yellow"
        table.add_row(
            u["name"], 
            str(u["recordings"]), 
            f"[{status_style}]{u['status']}[/]", 
            u["last_active"]
        )
    return Panel(table, border_style="cyan")

def generate_system_panel():
    s = get_system_stats()
    content = (
        f"💻 [bold]CPU Usage:[/] {s['cpu']}
"
        f"🧠 [bold]RAM Usage:[/] {s['ram']}
"
        f"🎮 [bold]GPU VRAM:[/] {s['gpu_vram']}
"
        f"-----------------
"
        f"🌐 [bold]Upload:[/] {s['net_up']}
"
        f"🌐 [bold]Download:[/] {s['net_down']}
"
        f"⚡ [bold]API Latency:[/] {s['api_latency']}"
    )
    return Panel(content, title="Sistem Sağlığı", border_style="green")

def run_dashboard():
    layout = create_layout()
    with Live(layout, refresh_per_second=2, screen=True):
        while True:
            layout["header"].update(generate_header())
            layout["users"].update(generate_user_table())
            layout["system"].update(generate_system_panel())
            layout["footer"].update(Panel("[dim white]Loglar: API Request received from user 'FurkanV1' at 11:24:10 | Model Training started for user 'Hasan'...", title="Sistem Mesajları"))
            time.sleep(0.5)

if __name__ == "__main__":
    try:
        run_dashboard()
    except KeyboardInterrupt:
        console.print("
[bold red]Dashboard kapatılıyor...[/]")
