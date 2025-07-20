#!/usr/bin/env python3
"""
Real-time Collection Session Monitor
Monitor and display live statistics for data collection sessions
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class CollectionSessionMonitor:
    """Real-time monitor for data collection sessions"""
    
    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)
        self.refresh_interval = 30  # seconds
        self.last_update = None
        
        if not self.session_dir.exists():
            raise ValueError(f"Session directory not found: {session_dir}")
    
    def get_latest_checkpoint(self) -> Optional[Dict]:
        """Get the most recent checkpoint data"""
        checkpoint_files = list(self.session_dir.glob("checkpoint_*.json"))
        
        if not checkpoint_files:
            return None
        
        # Get the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error reading checkpoint: {str(e)}")
            return None
    
    def get_session_logs(self, lines: int = 20) -> List[str]:
        """Get recent log entries"""
        log_files = list(self.session_dir.glob("*.log"))
        
        if not log_files:
            return ["No log files found"]
        
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        
        try:
            # Use tail command to get last N lines
            result = subprocess.run(['tail', '-n', str(lines), str(latest_log)], 
                                  capture_output=True, text=True)
            return result.stdout.strip().split('\n') if result.stdout else []
        except Exception:
            # Fallback to Python implementation
            try:
                with open(latest_log, 'r') as f:
                    lines_list = f.readlines()
                    return [line.strip() for line in lines_list[-lines:]]
            except Exception as e:
                return [f"Error reading logs: {str(e)}"]
    
    def calculate_progress_metrics(self, checkpoint_data: Dict) -> Dict:
        """Calculate progress and performance metrics"""
        session = checkpoint_data.get('session', {})
        stats = checkpoint_data.get('collection_stats', {})
        
        start_time = datetime.fromisoformat(session.get('start_time', datetime.now().isoformat()))
        current_time = datetime.now()
        elapsed_time = (current_time - start_time).total_seconds()
        
        duration_hours = session.get('duration_hours', 8)
        total_session_time = duration_hours * 3600
        
        progress_percentage = min(100, (elapsed_time / total_session_time) * 100)
        remaining_time = max(0, total_session_time - elapsed_time)
        
        # Performance metrics
        total_records = session.get('total_records_collected', 0)
        total_api_calls = session.get('total_api_calls', 0)
        
        records_per_hour = (total_records / (elapsed_time / 3600)) if elapsed_time > 0 else 0
        api_calls_per_hour = (total_api_calls / (elapsed_time / 3600)) if elapsed_time > 0 else 0
        
        # Projected totals
        projected_records = records_per_hour * duration_hours if records_per_hour > 0 else 0
        projected_api_calls = api_calls_per_hour * duration_hours if api_calls_per_hour > 0 else 0
        
        return {
            'progress_percentage': progress_percentage,
            'elapsed_hours': elapsed_time / 3600,
            'remaining_hours': remaining_time / 3600,
            'records_per_hour': records_per_hour,
            'api_calls_per_hour': api_calls_per_hour,
            'projected_records': projected_records,
            'projected_api_calls': projected_api_calls,
            'efficiency': (total_records / max(total_api_calls, 1))
        }
    
    def display_dashboard(self, checkpoint_data: Dict):
        """Display the monitoring dashboard"""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        session = checkpoint_data.get('session', {})
        stats = checkpoint_data.get('collection_stats', {})
        metrics = self.calculate_progress_metrics(checkpoint_data)
        
        # Header
        print("🚀 DATA COLLECTION SESSION MONITOR")
        print("=" * 80)
        
        # Session info
        print(f"📊 Session ID: {session.get('session_id', 'Unknown')}")
        print(f"📅 Started: {session.get('start_time', 'Unknown')}")
        print(f"📊 Status: {session.get('status', 'Unknown').upper()}")
        print(f"⏱️ Duration: {session.get('duration_hours', 0)} hours")
        
        # Progress bar
        progress = metrics['progress_percentage']
        bar_length = 50
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\n📈 Progress: [{bar}] {progress:.1f}%")
        print(f"⏱️ Elapsed: {metrics['elapsed_hours']:.1f}h | Remaining: {metrics['remaining_hours']:.1f}h")
        
        # Current statistics
        print(f"\n📊 CURRENT STATISTICS")
        print("-" * 40)
        print(f"📈 Records collected: {session.get('total_records_collected', 0):,}")
        print(f"📞 API calls made: {session.get('total_api_calls', 0):,}")
        print(f"❌ Errors encountered: {session.get('errors_encountered', 0)}")
        print(f"🔧 API efficiency: {metrics['efficiency']:.2f} records/call")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS")
        print("-" * 40)
        print(f"📊 Records/hour: {metrics['records_per_hour']:.0f}")
        print(f"📞 API calls/hour: {metrics['api_calls_per_hour']:.0f}")
        print(f"🎯 Projected records: {metrics['projected_records']:.0f}")
        print(f"🎯 Projected API calls: {metrics['projected_api_calls']:.0f}")
        
        # Source statistics
        source_stats = stats.get('sources_stats', {})
        if source_stats:
            print(f"\n🔧 SOURCE PERFORMANCE")
            print("-" * 40)
            for source, source_data in source_stats.items():
                records = source_data.get('total_records', 0)
                success_rate = source_data.get('success_rate', 0) * 100
                print(f"{source:20} {records:>8,} records  {success_rate:>5.1f}% success")
        
        # Hourly breakdown
        hourly_progress = stats.get('hourly_progress', [])
        if hourly_progress:
            print(f"\n📅 HOURLY BREAKDOWN")
            print("-" * 40)
            for hour_data in hourly_progress[-8:]:  # Show last 8 hours
                hour = hour_data.get('hour', 0)
                records = hour_data.get('records', 0)
                api_calls = hour_data.get('api_calls', 0)
                errors = hour_data.get('errors', 0)
                print(f"Hour {hour:2d}: {records:>6,} records, {api_calls:>6,} calls, {errors:>3} errors")
        
        # Recent logs
        print(f"\n📝 RECENT ACTIVITY")
        print("-" * 40)
        recent_logs = self.get_session_logs(8)
        for log_line in recent_logs[-8:]:
            if log_line.strip():
                # Extract timestamp and message
                parts = log_line.split(' - ', 3)
                if len(parts) >= 4:
                    timestamp = parts[0]
                    level = parts[2]
                    message = parts[3]
                    
                    # Color code by level
                    if 'ERROR' in level:
                        color = '\033[91m'  # Red
                    elif 'WARNING' in level:
                        color = '\033[93m'  # Yellow
                    elif 'INFO' in level:
                        color = '\033[92m'  # Green
                    else:
                        color = '\033[0m'   # Default
                    
                    # Truncate long messages
                    if len(message) > 60:
                        message = message[:57] + "..."
                    
                    print(f"{color}{timestamp[-8:]} {message}\033[0m")
        
        # Footer
        print("\n" + "=" * 80)
        print(f"🔄 Last updated: {datetime.now().strftime('%H:%M:%S')} | "
              f"Refresh interval: {self.refresh_interval}s | Press Ctrl+C to exit")
    
    def monitor_session(self):
        """Start monitoring the session"""
        print(f"🔍 Monitoring session: {self.session_dir.name}")
        print(f"📁 Directory: {self.session_dir}")
        print(f"🔄 Refresh interval: {self.refresh_interval} seconds")
        print("\nStarting monitor... Press Ctrl+C to exit")
        
        try:
            while True:
                checkpoint_data = self.get_latest_checkpoint()
                
                if checkpoint_data:
                    self.display_dashboard(checkpoint_data)
                    self.last_update = datetime.now()
                else:
                    print("⚠️ No checkpoint data found. Waiting for session to start...")
                
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped by user")
        except Exception as e:
            print(f"\n❌ Monitor error: {str(e)}")

def find_active_sessions() -> List[Path]:
    """Find active session directories"""
    sessions_root = project_root / 'data' / 'orchestrator_sessions'
    
    if not sessions_root.exists():
        return []
    
    active_sessions = []
    for session_dir in sessions_root.iterdir():
        if session_dir.is_dir():
            # Check if session has recent activity (checkpoint within last hour)
            checkpoints = list(session_dir.glob("checkpoint_*.json"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                last_modified = datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)
                
                if (datetime.now() - last_modified).total_seconds() < 3600:  # Within last hour
                    active_sessions.append(session_dir)
    
    return active_sessions

def main():
    """Main monitor function"""
    parser = argparse.ArgumentParser(description="Monitor data collection session")
    parser.add_argument('--session', type=str, help='Session directory to monitor')
    parser.add_argument('--list', action='store_true', help='List active sessions')
    parser.add_argument('--refresh', type=int, default=30, help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    if args.list:
        print("🔍 ACTIVE COLLECTION SESSIONS")
        print("=" * 50)
        
        active_sessions = find_active_sessions()
        
        if not active_sessions:
            print("No active sessions found")
            return
        
        for i, session_dir in enumerate(active_sessions, 1):
            print(f"{i}. {session_dir.name}")
            
            # Get basic info from latest checkpoint
            checkpoints = list(session_dir.glob("checkpoint_*.json"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_checkpoint, 'r') as f:
                        data = json.load(f)
                        session_info = data.get('session', {})
                        print(f"   Status: {session_info.get('status', 'unknown')}")
                        print(f"   Records: {session_info.get('total_records_collected', 0):,}")
                        print(f"   Started: {session_info.get('start_time', 'unknown')}")
                except Exception:
                    print("   (Unable to read session data)")
            print()
        
        return
    
    # Monitor specific session
    if args.session:
        session_dir = args.session
    else:
        # Auto-detect active session
        active_sessions = find_active_sessions()
        
        if not active_sessions:
            print("❌ No active sessions found")
            print("💡 Use --list to see all sessions or --session to specify one")
            return
        elif len(active_sessions) == 1:
            session_dir = str(active_sessions[0])
            print(f"🎯 Auto-detected session: {active_sessions[0].name}")
        else:
            print("🔍 Multiple active sessions found:")
            for i, session in enumerate(active_sessions, 1):
                print(f"   {i}. {session.name}")
            
            try:
                choice = int(input("Select session to monitor (number): ")) - 1
                if 0 <= choice < len(active_sessions):
                    session_dir = str(active_sessions[choice])
                else:
                    print("❌ Invalid selection")
                    return
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid selection")
                return
    
    # Start monitoring
    try:
        monitor = CollectionSessionMonitor(session_dir)
        monitor.refresh_interval = args.refresh
        monitor.monitor_session()
    except Exception as e:
        print(f"❌ Failed to start monitor: {str(e)}")

if __name__ == "__main__":
    main()
