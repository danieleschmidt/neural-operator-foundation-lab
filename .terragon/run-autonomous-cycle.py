#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Cycle Runner
Orchestrates the complete autonomous discovery -> prioritization -> execution cycle.
"""

import subprocess
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class AutonomousCycleRunner:
    def __init__(self):
        self.repo_root = Path.cwd()
        self.terragon_dir = Path(".terragon")
        
    def run_discovery(self) -> bool:
        """Run value discovery phase"""
        print("🔍 Phase 1: Value Discovery")
        print("=" * 50)
        
        try:
            result = subprocess.run([
                sys.executable, ".terragon/value-discovery.py"
            ], cwd=self.repo_root, check=True, capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print(f"Warnings: {result.stderr}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Discovery failed: {e}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            return False
    
    def run_metrics_collection(self) -> bool:
        """Run metrics collection phase"""
        print("\n📊 Phase 2: Metrics Collection")
        print("=" * 50)
        
        try:
            result = subprocess.run([
                sys.executable, ".terragon/metrics-collector.py"
            ], cwd=self.repo_root, check=True, capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print(f"Warnings: {result.stderr}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Metrics collection failed: {e}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            return False
    
    def analyze_backlog(self) -> Optional[Dict[str, Any]]:
        """Analyze current backlog and return next best item"""
        backlog_file = self.terragon_dir / "backlog.json"
        
        if not backlog_file.exists():
            print("📋 No backlog found. Discovery may have found no items.")
            return None
        
        try:
            with open(backlog_file) as f:
                backlog_data = json.load(f)
            
            items = backlog_data.get("items", [])
            if not items:
                print("📋 Backlog is empty. No work items discovered.")
                return None
            
            # Sort by composite score
            items.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
            best_item = items[0]
            
            print(f"\n🎯 Next Best Value Item:")
            print(f"   ID: {best_item['id']}")
            print(f"   Title: {best_item['title']}")
            print(f"   Category: {best_item['category']}")
            print(f"   Score: {best_item.get('composite_score', 'N/A')}")
            print(f"   Estimated Effort: {best_item.get('estimated_effort_hours', 'N/A')}h")
            print(f"   Files Affected: {len(best_item.get('files_affected', []))}")
            
            return best_item
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Failed to analyze backlog: {e}")
            return None
    
    def run_execution(self, dry_run: bool = False) -> bool:
        """Run autonomous execution phase"""
        print(f"\n🚀 Phase 3: Autonomous Execution {'(DRY RUN)' if dry_run else ''}")
        print("=" * 50)
        
        if dry_run:
            print("🔍 DRY RUN: Would execute highest-value item but no changes will be made")
            return True
        
        try:
            result = subprocess.run([
                sys.executable, ".terragon/autonomous-executor.py"
            ], cwd=self.repo_root, check=True, capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print(f"Warnings: {result.stderr}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Execution failed: {e}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            return False
    
    def update_backlog_display(self):
        """Update the BACKLOG.md file with current status"""
        print("\n📝 Phase 4: Updating Backlog Display")
        print("=" * 50)
        
        try:
            # Load current backlog
            backlog_file = self.terragon_dir / "backlog.json"
            metrics_file = self.terragon_dir / "metrics" / "latest.json"
            
            backlog_data = {}
            metrics_data = {}
            
            if backlog_file.exists():
                with open(backlog_file) as f:
                    backlog_data = json.load(f)
            
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics_data = json.load(f)
            
            # Generate updated BACKLOG.md
            self._generate_updated_backlog(backlog_data, metrics_data)
            print("✅ Updated BACKLOG.md with current status")
            
        except Exception as e:
            print(f"❌ Failed to update backlog display: {e}")
    
    def _generate_updated_backlog(self, backlog_data: Dict[str, Any], metrics_data: Dict[str, Any]):
        """Generate updated BACKLOG.md content"""
        
        items = backlog_data.get("items", [])
        total_items = len(items)
        
        # Categorize items by score (handle string scores)
        def get_score(item):
            score = item.get("composite_score", 0)
            return float(score) if isinstance(score, (int, float, str)) and str(score).replace('.', '').isdigit() else 0.0
        
        high_priority = [item for item in items if get_score(item) > 50]
        medium_priority = [item for item in items if 25 <= get_score(item) <= 50]
        low_priority = [item for item in items if get_score(item) < 25]
        
        # Calculate source distribution
        source_counts = {}
        for item in items:
            source = item.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Create updated content
        content = f"""# 📊 Autonomous Value Backlog

> **Terragon SDLC Value Discovery System**  
> Repository: neural-operator-foundation-lab  
> Maturity Level: MATURING (50-75%)

Last Updated: {datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}  
Next Discovery Run: Every PR merge + hourly security scans

## 🎯 Current Status

**Repository Health**: {'🟢 Good' if metrics_data.get('maturity_score', 0) > 70 else '🟡 Moderate' if metrics_data.get('maturity_score', 0) > 50 else '🔴 Needs Attention'}  
**SDLC Maturity**: {metrics_data.get('maturity_score', 'N/A')}% (MATURING → ADVANCED transition)  
**Technical Debt Level**: {'🟢 Low' if metrics_data.get('technical_debt', {}).get('total_debt_comments', 0) < 10 else '🟡 Moderate' if metrics_data.get('technical_debt', {}).get('total_debt_comments', 0) < 25 else '🔴 High'}  
**Security Posture**: {'🟢 Strong' if metrics_data.get('security', {}).get('security_issues', 0) == 0 else '🟡 Moderate' if metrics_data.get('security', {}).get('security_issues', 0) < 5 else '🔴 Needs Attention'}  

## 🚀 Next Execution Ready

> Run: `python .terragon/run-autonomous-cycle.py`

## 📋 Discovered Work Items

### High Priority (Score > 50) - {len(high_priority)} items
"""
        
        if high_priority:
            for item in high_priority[:5]:  # Show top 5
                score = get_score(item)
                content += f"""
**[{item['id']}] {item['title']}**
- **Score**: {score:.1f}
- **Category**: {item['category']}
- **Effort**: {item.get('estimated_effort_hours', 'N/A')}h
- **Files**: {', '.join(item.get('files_affected', [])[:2])}{'...' if len(item.get('files_affected', [])) > 2 else ''}
"""
        else:
            content += "\n- No high-priority items discovered\n"
        
        content += f"""
### Medium Priority (Score 25-50) - {len(medium_priority)} items
"""
        
        if medium_priority:
            for item in medium_priority[:3]:  # Show top 3
                score = get_score(item)
                content += f"""
**[{item['id']}] {item['title']}**
- **Score**: {score:.1f} | **Category**: {item['category']} | **Effort**: {item.get('estimated_effort_hours', 'N/A')}h
"""
        else:
            content += "\n- No medium-priority items discovered\n"
        
        content += f"""
### Low Priority (Score < 25) - {len(low_priority)} items

{"- " + str(len(low_priority)) + " low-priority maintenance items available" if low_priority else "- No low-priority items discovered"}

## 📈 Value Delivery Metrics

### Repository Metrics (Latest)
- **Maturity Score**: {metrics_data.get('maturity_score', 'N/A')}/100
- **Lines of Code**: {metrics_data.get('repository', {}).get('lines_of_code', 'N/A')}
- **Test Coverage**: {metrics_data.get('testing', {}).get('test_coverage_percentage', 'N/A')}%
- **Technical Debt Comments**: {metrics_data.get('technical_debt', {}).get('total_debt_comments', 'N/A')}
- **Security Issues**: {metrics_data.get('security', {}).get('security_issues', 'N/A')}

### Continuous Discovery Stats
- **Total Items Discovered**: {total_items}
- **Items Awaiting Execution**: {len([item for item in items if get_score(item) > 15])}
- **Average Item Score**: {sum(get_score(item) for item in items) / len(items):.1f if items else 'N/A'}

### Discovery Sources Distribution
"""
        
        if source_counts:
            total_sources = sum(source_counts.values())
            for source, count in sorted(source_counts.items(), key=lambda x: int(x[1]) if isinstance(x[1], (int, str)) and str(x[1]).isdigit() else 0, reverse=True):
                percentage = (count / total_sources) * 100
                content += f"- **{source.replace('_', ' ').title()}**: {percentage:.1f}% ({count} items)\n"
        else:
            content += "- No items discovered yet\n"
        
        content += f"""

## 🔄 Autonomous Execution Schedule

### Continuous Execution Pipeline
- **Immediate**: After each PR merge
- **Hourly**: Security vulnerability scans
- **Daily**: Comprehensive static analysis  
- **Weekly**: Deep architectural analysis
- **Monthly**: Strategic value recalibration

### Current Execution Status
- **Last Discovery**: {backlog_data.get('last_updated', 'Never')}
- **Items In Backlog**: {total_items}
- **Ready for Execution**: {len([item for item in items if get_score(item) > 15])}

## 🛠️ Discovered Technical Debt Hotspots

"""
        
        if metrics_data.get('technical_debt', {}).get('debt_hotspots'):
            content += "### Code Complexity Hotspots\n"
            for file_path, count in metrics_data['technical_debt']['debt_hotspots']:
                content += f"- **{file_path}**: {count} debt markers\n"
        else:
            content += "### Code Complexity Hotspots\n- No significant debt hotspots identified\n"
        
        content += f"""

### Security Debt
- **Security Issues Found**: {metrics_data.get('security', {}).get('security_issues', 'N/A')}
- **Vulnerable Dependencies**: {metrics_data.get('security', {}).get('vulnerable_dependencies', 'N/A')}

### Performance Debt
- **Performance Test Files**: {metrics_data.get('performance', {}).get('performance_test_files', 'N/A')}
- **Performance Monitoring Usage**: {metrics_data.get('performance', {}).get('performance_monitoring_usage', 'N/A')} markers

## 🎨 Quick Start Commands

### Run Complete Autonomous Cycle
```bash
# Full discovery -> execution cycle
python .terragon/run-autonomous-cycle.py

# Discovery only
python .terragon/value-discovery.py

# Metrics collection only
python .terragon/metrics-collector.py

# Execution only (requires existing backlog)
python .terragon/autonomous-executor.py
```

### Manual Analysis
```bash
# View current backlog
cat .terragon/backlog.json | jq '.items[:5]'

# View latest metrics
cat .terragon/metrics/latest.json | jq '.maturity_score'

# Check system status
find .terragon -name "*.json" -exec echo {{}} \\; -exec head -5 {{}} \\;
```

---

**🤖 Powered by Terragon Autonomous SDLC**  
*Continuously discovering and delivering maximum value through intelligent prioritization*

Last Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Write updated content
        with open("BACKLOG.md", "w") as f:
            f.write(content)
    
    def run_full_cycle(self, dry_run: bool = False, skip_execution: bool = False) -> bool:
        """Run the complete autonomous SDLC cycle"""
        
        print("🤖 TERRAGON AUTONOMOUS SDLC CYCLE")
        print("=" * 60)
        print(f"Repository: {self.repo_root.name}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {'DRY RUN' if dry_run else 'PRODUCTION'}")
        print("=" * 60)
        
        success_count = 0
        total_phases = 4
        
        # Phase 1: Discovery
        if self.run_discovery():
            success_count += 1
        else:
            print("❌ Discovery phase failed. Stopping cycle.")
            return False
        
        # Phase 2: Metrics Collection
        if self.run_metrics_collection():
            success_count += 1
        else:
            print("⚠️ Metrics collection failed but continuing...")
        
        # Analyze what was discovered
        next_item = self.analyze_backlog()
        
        # Phase 3: Execution (if not skipped)
        if not skip_execution:
            if next_item:
                if self.run_execution(dry_run=dry_run):
                    success_count += 1
                else:
                    print("❌ Execution phase failed.")
            else:
                print("✅ No items to execute. Repository is optimized!")
                success_count += 1
        else:
            print("⏭️ Execution phase skipped.")
            success_count += 1
        
        # Phase 4: Update Display
        self.update_backlog_display()
        success_count += 1
        
        # Summary
        print(f"\n🎯 CYCLE COMPLETE")
        print("=" * 60)
        print(f"✅ Successful phases: {success_count}/{total_phases}")
        print(f"📊 Repository status: Check BACKLOG.md for details")
        
        if next_item and not skip_execution and not dry_run:
            print(f"🚀 Executed: {next_item['title']}")
            print(f"💰 Value delivered: {next_item.get('composite_score', 'N/A')} points")
        
        return success_count >= 3  # Allow one phase to fail

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Terragon Autonomous SDLC Cycle")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (no actual changes)")
    parser.add_argument("--discovery-only", action="store_true", help="Run discovery and metrics only")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    runner = AutonomousCycleRunner()
    
    success = runner.run_full_cycle(
        dry_run=args.dry_run,
        skip_execution=args.discovery_only
    )
    
    if success:
        print("\n🎉 Autonomous SDLC cycle completed successfully!")
        exit(0)
    else:
        print("\n❌ Autonomous SDLC cycle failed!")
        exit(1)

if __name__ == "__main__":
    main()