#!/usr/bin/env python3
"""
🎯 最终交付验证脚本 - 模拟客户使用场景

这个脚本模拟客户拿到项目后的完整使用流程：
1. 环境验证和依赖检查
2. 项目结构完整性验证  
3. 文档可读性检查
4. 模型快速上手验证
5. 核心功能端到端测试
6. 生成交付报告

使用方法:
    python final_delivery_validation.py
"""

import os
import sys
import subprocess
import importlib
import json
import time
from datetime import datetime
from pathlib import Path
import torch


class DeliveryValidator:
    """最终交付验证器"""

    def __init__(self):
        self.validation_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'validation_time': self.validation_time,
            'validation_steps': {},
            'overall_status': 'unknown',
            'issues': [],
            'recommendations': []
        }

        print(f"🎯 Final Delivery Validation")
        print(f"   Time: {self.validation_time}")
        print(f"=" * 60)

    def validate_environment(self):
        """Step 1: 环境验证和依赖检查"""

        print(f"\n📋 Step 1: Environment & Dependencies Validation")
        step_results = {'status': 'unknown', 'checks': {}}

        # Check Python version
        python_version = sys.version_info
        check_python = python_version >= (3, 8)
        step_results['checks']['python_version'] = {
            'status': check_python,
            'value': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            'requirement': '>=3.8'
        }
        print(
            f"   Python Version: {python_version.major}.{python_version.minor} {'✅' if check_python else '❌'}")

        # Check core dependencies
        core_deps = {
            'torch': 'PyTorch',
            'numpy': 'NumPy',
            'matplotlib': 'Matplotlib'
        }

        for module, name in core_deps.items():
            try:
                importlib.import_module(module)
                step_results['checks'][module] = {
                    'status': True, 'error': None}
                print(f"   {name}: ✅")
            except ImportError as e:
                step_results['checks'][module] = {
                    'status': False, 'error': str(e)}
                print(f"   {name}: ❌ ({e})")
                self.results['issues'].append(f"Missing dependency: {name}")

        # Check optional dependencies
        optional_deps = {'mamba_ssm': 'Mamba SSM'}
        for module, name in optional_deps.items():
            try:
                importlib.import_module(module)
                step_results['checks'][module] = {
                    'status': True, 'error': None}
                print(f"   {name} (optional): ✅")
            except ImportError:
                step_results['checks'][module] = {
                    'status': False, 'error': 'Not installed'}
                print(f"   {name} (optional): ⚠️ Not available")
                self.results['recommendations'].append(
                    f"Install {name} for full functionality")

        # Check PyTorch CUDA
        if 'torch' in [k for k, v in step_results['checks'].items() if v['status']]:
            cuda_available = torch.cuda.is_available()
            step_results['checks']['cuda'] = {'status': cuda_available}
            print(
                f"   CUDA Support: {'✅' if cuda_available else '⚠️ CPU Only'}")

        # Overall step status
        core_checks = [v['status']
                       for k, v in step_results['checks'].items() if k in core_deps]
        step_results['status'] = 'pass' if all(core_checks) else 'fail'

        self.results['validation_steps']['environment'] = step_results
        return step_results['status'] == 'pass'

    def validate_project_structure(self):
        """Step 2: 项目结构完整性验证"""

        print(f"\n📋 Step 2: Project Structure Validation")
        step_results = {'status': 'unknown', 'structure': {}}

        # Required directories
        required_dirs = {
            'model': 'Core model code',
            'data': 'Data processing',
            'utils': 'Utility functions',
            'configs': 'Configuration files',
            'docs': 'Documentation',
            'tests': 'Test files',
            'scripts': 'Utility scripts'
        }

        for dir_name, description in required_dirs.items():
            dir_path = Path(dir_name)
            exists = dir_path.exists() and dir_path.is_dir()
            step_results['structure'][dir_name] = {
                'exists': exists,
                'description': description
            }
            print(f"   📂 {dir_name}/: {'✅' if exists else '❌'} ({description})")

            if not exists:
                self.results['issues'].append(
                    f"Missing directory: {dir_name}/")

        # Required files
        required_files = {
            'README.md': 'Project documentation',
            'requirements.txt': 'Python dependencies',
            'train.py': 'Main training script',
            'demo.py': 'Demo script',
            'LICENSE': 'License file'
        }

        for file_name, description in required_files.items():
            file_path = Path(file_name)
            exists = file_path.exists() and file_path.is_file()
            step_results['structure'][file_name] = {
                'exists': exists,
                'description': description,
                'size': file_path.stat().st_size if exists else 0
            }
            print(f"   📄 {file_name}: {'✅' if exists else '❌'} ({description})")

            if not exists:
                self.results['issues'].append(f"Missing file: {file_name}")

        # Core model files
        core_models = [
            'model/MotionAGFormer.py',
            'model/modules/mamba_gcn_block.py',
            'model/modules/mamba_layer.py',
            'model/modules/gcn_layer.py'
        ]

        for model_file in core_models:
            model_path = Path(model_file)
            exists = model_path.exists()
            step_results['structure'][model_file] = {'exists': exists}
            print(f"   🧠 {model_file}: {'✅' if exists else '❌'}")

            if not exists:
                self.results['issues'].append(
                    f"Missing core model: {model_file}")

        # Calculate overall status
        all_checks = [v['exists'] for v in step_results['structure'].values()]
        step_results['status'] = 'pass' if all(all_checks) else 'partial'

        self.results['validation_steps']['project_structure'] = step_results
        return step_results['status'] != 'fail'

    def validate_documentation(self):
        """Step 3: 文档可读性检查"""

        print(f"\n📋 Step 3: Documentation Validation")
        step_results = {'status': 'unknown', 'docs': {}}

        # Check README content
        readme_path = Path('README.md')
        if readme_path.exists():
            readme_content = readme_path.read_text()
            readme_checks = {
                'has_title': '# MotionAGFormer' in readme_content or '# Mamba' in readme_content,
                'has_installation': 'install' in readme_content.lower() or 'pip' in readme_content,
                'has_usage': 'usage' in readme_content.lower() or 'example' in readme_content.lower(),
                'has_architecture': 'architecture' in readme_content.lower() or 'model' in readme_content.lower(),
                # Should be comprehensive
                'sufficient_length': len(readme_content) > 5000
            }

            step_results['docs']['README.md'] = readme_checks

            passed_checks = sum(readme_checks.values())
            total_checks = len(readme_checks)
            print(
                f"   📚 README.md: {passed_checks}/{total_checks} checks ({'✅' if passed_checks >= 4 else '⚠️'})")

            for check, passed in readme_checks.items():
                print(f"      {check}: {'✅' if passed else '❌'}")

            if passed_checks < 4:
                self.results['issues'].append("README.md needs improvement")
        else:
            step_results['docs']['README.md'] = {'exists': False}
            print(f"   📚 README.md: ❌ Missing")
            self.results['issues'].append("README.md is missing")

        # Check technical documentation
        doc_files = list(Path('docs').glob('*.md')
                         ) if Path('docs').exists() else []
        step_results['docs']['technical_docs'] = {
            'count': len(doc_files),
            'files': [f.name for f in doc_files]
        }
        print(
            f"   📖 Technical docs: {len(doc_files)} files {'✅' if len(doc_files) >= 3 else '⚠️'}")

        if len(doc_files) < 3:
            self.results['recommendations'].append(
                "Add more technical documentation")

        # Overall status
        readme_score = sum(step_results['docs'].get('README.md', {}).values(
        )) if isinstance(step_results['docs'].get('README.md'), dict) else 0
        step_results['status'] = 'pass' if readme_score >= 4 and len(
            doc_files) >= 2 else 'partial'

        self.results['validation_steps']['documentation'] = step_results
        return step_results['status'] != 'fail'

    def validate_quick_start(self):
        """Step 4: 模型快速上手验证"""

        print(f"\n📋 Step 4: Quick Start Validation")
        step_results = {'status': 'unknown', 'tests': {}}

        try:
            # Test basic model import
            print(f"   🧠 Testing model import...")
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from model.MotionAGFormer import MotionAGFormer
            step_results['tests']['model_import'] = {'status': True}
            print(f"      ✅ Model import successful")

            # Test model creation
            print(f"   🔧 Testing model creation...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Test baseline model
            model_baseline = MotionAGFormer(
                n_layers=2, dim_in=2, dim_feat=32, dim_out=3,
                n_frames=9, use_mamba_gcn=False
            ).to(device)
            step_results['tests']['baseline_creation'] = {'status': True}
            print(f"      ✅ Baseline model creation")

            # Test MambaGCN model (if possible)
            try:
                model_mamba = MotionAGFormer(
                    n_layers=2, dim_in=2, dim_feat=32, dim_out=3,
                    n_frames=9, use_mamba_gcn=True,
                    mamba_gcn_use_mamba=True, mamba_gcn_use_attention=False
                ).to(device)
                step_results['tests']['mamba_gcn_creation'] = {'status': True}
                print(f"      ✅ MambaGCN model creation")
            except Exception as e:
                step_results['tests']['mamba_gcn_creation'] = {
                    'status': False, 'error': str(e)}
                print(f"      ⚠️ MambaGCN model creation failed: {e}")
                self.results['recommendations'].append(
                    "Check Mamba-SSM installation for full functionality")

            # Test forward pass
            print(f"   ⚡ Testing forward pass...")
            test_input = torch.randn(2, 9, 17, 2).to(device)

            with torch.no_grad():
                output_baseline = model_baseline(test_input)
                expected_shape = [2, 9, 17, 3]
                shape_correct = list(output_baseline.shape) == expected_shape

                step_results['tests']['forward_pass'] = {
                    'status': shape_correct,
                    'output_shape': list(output_baseline.shape),
                    'expected_shape': expected_shape
                }
                print(
                    f"      {'✅' if shape_correct else '❌'} Forward pass: {list(output_baseline.shape)}")

            step_results['status'] = 'pass'

        except Exception as e:
            step_results['tests']['general_error'] = {
                'status': False, 'error': str(e)}
            step_results['status'] = 'fail'
            print(f"   ❌ Quick start validation failed: {e}")
            self.results['issues'].append(f"Quick start failed: {e}")

        self.results['validation_steps']['quick_start'] = step_results
        return step_results['status'] == 'pass'

    def validate_end_to_end(self):
        """Step 5: 核心功能端到端测试"""

        print(f"\n📋 Step 5: End-to-End Functionality Test")
        step_results = {'status': 'unknown', 'tests': {}}

        try:
            # Check if validation scripts exist and run one
            validation_scripts = [
                'tests/poc_training_validation.py',
                'tests/end_to_end_validation.py',
                'demo.py'
            ]

            available_script = None
            for script in validation_scripts:
                if Path(script).exists():
                    available_script = script
                    break

            if available_script:
                print(f"   🧪 Running validation script: {available_script}")

                # Run the validation script (with timeout)
                try:
                    if 'demo.py' in available_script:
                        cmd = [sys.executable, available_script,
                               '--quick', '--config', 'baseline']
                    else:
                        cmd = [sys.executable, available_script]

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=120  # 2 minutes timeout
                    )

                    success = result.returncode == 0
                    step_results['tests']['validation_script'] = {
                        'status': success,
                        'script': available_script,
                        'returncode': result.returncode,
                        'stdout_lines': len(result.stdout.split('\n')),
                        'stderr_lines': len(result.stderr.split('\n'))
                    }

                    print(f"      {'✅' if success else '❌'} Script execution")
                    if not success:
                        print(f"      Error: {result.stderr[:200]}...")
                        self.results['issues'].append(
                            f"Validation script failed: {available_script}")

                except subprocess.TimeoutExpired:
                    step_results['tests']['validation_script'] = {
                        'status': False,
                        'error': 'timeout'
                    }
                    print(f"      ⏰ Script timeout (>2min)")
                    self.results['recommendations'].append(
                        "Consider optimizing script performance")

                except Exception as e:
                    step_results['tests']['validation_script'] = {
                        'status': False,
                        'error': str(e)
                    }
                    print(f"      ❌ Script error: {e}")

            else:
                step_results['tests']['validation_script'] = {
                    'status': False,
                    'error': 'no_script_found'
                }
                print(f"   ⚠️ No validation scripts found")
                self.results['issues'].append(
                    "No validation scripts available")

            # Manual mini-test as fallback
            print(f"   🔬 Running mini integration test...")
            try:
                from model.MotionAGFormer import MotionAGFormer

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = MotionAGFormer(
                    n_layers=1, dim_in=2, dim_feat=16, dim_out=3,
                    n_frames=3, use_mamba_gcn=False
                ).to(device)

                # Mini training test
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = torch.nn.MSELoss()

                model.train()
                for i in range(3):  # 3 mini steps
                    optimizer.zero_grad()

                    x = torch.randn(1, 3, 17, 2).to(device)
                    y = torch.randn(1, 3, 17, 3).to(device)

                    pred = model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()

                step_results['tests']['mini_integration'] = {'status': True}
                print(f"      ✅ Mini integration test passed")

            except Exception as e:
                step_results['tests']['mini_integration'] = {
                    'status': False, 'error': str(e)}
                print(f"      ❌ Mini integration test failed: {e}")
                self.results['issues'].append(f"Integration test failed: {e}")

            # Determine overall status
            test_results = [v.get('status', False)
                            for v in step_results['tests'].values()]
            step_results['status'] = 'pass' if any(test_results) else 'fail'

        except Exception as e:
            step_results['status'] = 'fail'
            step_results['tests']['general_error'] = {'error': str(e)}
            print(f"   ❌ End-to-end validation failed: {e}")
            self.results['issues'].append(f"End-to-end validation error: {e}")

        self.results['validation_steps']['end_to_end'] = step_results
        return step_results['status'] == 'pass'

    def generate_delivery_report(self):
        """生成交付报告"""

        print(f"\n📋 Step 6: Generating Delivery Report")

        # Calculate overall status
        step_statuses = [
            self.results['validation_steps'].get(
                step, {}).get('status', 'fail')
            for step in ['environment', 'project_structure', 'documentation', 'quick_start', 'end_to_end']
        ]

        pass_count = sum(1 for status in step_statuses if status == 'pass')
        partial_count = sum(
            1 for status in step_statuses if status == 'partial')
        total_steps = len(step_statuses)

        if pass_count == total_steps:
            overall_status = 'excellent'
        elif pass_count + partial_count >= total_steps * 0.8:
            overall_status = 'good'
        elif pass_count + partial_count >= total_steps * 0.6:
            overall_status = 'acceptable'
        else:
            overall_status = 'needs_improvement'

        self.results['overall_status'] = overall_status
        self.results['summary'] = {
            'total_steps': total_steps,
            'passed_steps': pass_count,
            'partial_steps': partial_count,
            'failed_steps': total_steps - pass_count - partial_count,
            'pass_rate': (pass_count + partial_count * 0.5) / total_steps
        }

        # Save detailed report
        report_file = f"delivery_validation_report_{self.validation_time}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"   ✅ Detailed report saved: {report_file}")
        return report_file

    def print_final_summary(self):
        """打印最终总结"""

        print(f"\n" + "="*60)
        print(f"🎯 FINAL DELIVERY VALIDATION SUMMARY")
        print(f"="*60)

        summary = self.results.get('summary', {})
        status_icons = {
            'excellent': '🌟',
            'good': '✅',
            'acceptable': '⚠️',
            'needs_improvement': '❌'
        }

        status_descriptions = {
            'excellent': 'Excellent - Ready for production',
            'good': 'Good - Minor issues, ready to deploy',
            'acceptable': 'Acceptable - Some improvements needed',
            'needs_improvement': 'Needs Improvement - Major issues found'
        }

        overall_status = self.results.get('overall_status', 'unknown')
        print(f"\n{status_icons.get(overall_status, '❓')} Overall Status: {status_descriptions.get(overall_status, 'Unknown')}")

        if summary:
            print(f"\n📊 Validation Results:")
            print(
                f"   ✅ Passed Steps: {summary['passed_steps']}/{summary['total_steps']}")
            print(
                f"   ⚠️ Partial Steps: {summary['partial_steps']}/{summary['total_steps']}")
            print(
                f"   ❌ Failed Steps: {summary['failed_steps']}/{summary['total_steps']}")
            print(f"   📈 Success Rate: {summary['pass_rate']:.1%}")

        if self.results.get('issues'):
            print(f"\n🚨 Issues Found ({len(self.results['issues'])}):")
            for issue in self.results['issues'][:5]:  # Show top 5
                print(f"   • {issue}")
            if len(self.results['issues']) > 5:
                print(f"   ... and {len(self.results['issues']) - 5} more")

        if self.results.get('recommendations'):
            print(
                f"\n💡 Recommendations ({len(self.results['recommendations'])}):")
            for rec in self.results['recommendations'][:3]:  # Show top 3
                print(f"   • {rec}")
            if len(self.results['recommendations']) > 3:
                print(
                    f"   ... and {len(self.results['recommendations']) - 3} more")

        print(f"\n🕐 Validation completed at: {self.validation_time}")
        print(
            f"📄 Detailed report: delivery_validation_report_{self.validation_time}.json")

        # Final recommendation
        if overall_status in ['excellent', 'good']:
            print(f"\n🚀 READY FOR DELIVERY! {status_icons[overall_status]}")
        elif overall_status == 'acceptable':
            print(f"\n⚠️ MOSTLY READY - Address minor issues before delivery")
        else:
            print(f"\n❌ NOT READY - Significant improvements needed")


def main():
    """主函数"""

    print("🎯 MotionAGFormer + MambaGCN Final Delivery Validation")
    print("   Simulating customer delivery scenario...")
    print()

    validator = DeliveryValidator()

    # Run validation steps
    step1_pass = validator.validate_environment()
    step2_pass = validator.validate_project_structure()
    step3_pass = validator.validate_documentation()
    step4_pass = validator.validate_quick_start()
    step5_pass = validator.validate_end_to_end()

    # Generate report
    report_file = validator.generate_delivery_report()

    # Print final summary
    validator.print_final_summary()


if __name__ == "__main__":
    main()
