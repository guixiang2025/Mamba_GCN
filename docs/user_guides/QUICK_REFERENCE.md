# 🚀 MotionAGFormer + MambaGCN 快速参考手册

> **客户交付后操作速查表** | [详细指引](CLIENT_POST_DELIVERY_GUIDE.md)

---

## 🎯 一键开始

### 💡 超快启动 (推荐)
```bash
# 1. 验证环境
python3 final_delivery_validation_real.py

# 2. 快速训练 (MambaGCN基础模型)
./quick_start_training.sh mamba_gcn base 1

# 3. 查看结果
python3 demo_real.py real
```

### 🔧 环境检查
```bash
# GPU状态
nvidia-smi

# 数据验证  
python3 test_real_data.py

# 模型导入测试
python3 -c "from model.MotionAGFormer import MotionAGFormer; print('✅ 模型可用')"
```

---

## 🚀 1. 大规模训练 (100+ GPU小时)

### 核心训练命令
```bash
# 基线模型 (对比用)
python3 scripts/train_real.py --model_type baseline --epochs 200 --batch_size 64

# MambaGCN (主要创新)  
python3 scripts/train_real.py --model_type mamba_gcn --epochs 200 --batch_size 64

# 完整架构 (所有组件)
python3 scripts/train_real.py --model_type full --epochs 200 --batch_size 48
```

### 高性能配置
    --model_type mamba_gcn --epochs 300 --batch_size 16
```

### 训练监控
```bash
# 实时日志
tail -f checkpoints/*/training.log

# GPU监控  
watch -n 1 nvidia-smi

# 性能指标
python3 -c "
import json
with open('checkpoints/mamba_gcn_*/metrics.json') as f:
    metrics = json.load(f)
    print(f'Best MPJPE: {min(metrics[\"mpjpe\"]):.2f}mm')
"
```

---

## ⚙️ 2. 超参数调优

### 🎯 关键参数

| 参数 | 范围 | 默认值 | 影响 |
|------|------|--------|------|
| `--lr` | 1e-5 ~ 1e-3 | 1e-4 | 收敛速度 |
| `--batch_size` | 16 ~ 128 | 64 | 内存/稳定性 |
| `--epochs` | 200 ~ 500 | 300 | 收敛程度 |
| `--weight_decay` | 1e-6 ~ 1e-3 | 1e-4 | 过拟合控制 |

### 📊 超参数搜索
```bash
# 批量实验
for lr in 5e-5 1e-4 2e-4; do
    for bs in 32 64 96; do
        python3 scripts/train_real.py \
            --model_type mamba_gcn \
            --lr $lr --batch_size $bs --epochs 50 \
            --save_dir "experiments/search_lr${lr}_bs${bs}"
    done
done

# 结果分析
python3 -c "
import os, json
results = []
for d in os.listdir('experiments'):
    if os.path.exists(f'experiments/{d}/metrics.json'):
        with open(f'experiments/{d}/metrics.json') as f:
            metrics = json.load(f)
            results.append((d, min(metrics['mpjpe'])))
results.sort(key=lambda x: x[1])
print('🏆 Top 3:', results[:3])
"
```

---

## 📊 3. 结果分析与论文

### 🔬 性能评估
```bash
# 最佳模型测试
python3 baseline_validation_real.py \
    --model_path checkpoints/best_model.pth \
    --model_type mamba_gcn

# 消融实验
python3 compare_data_performance.py \
    --baseline_model checkpoints/baseline/best_model.pth \
    --mamba_gcn_model checkpoints/mamba_gcn/best_model.pth \
    --full_model checkpoints/full/best_model.pth
```

### 📈 生成图表
```bash
# 性能对比图
python3 -c "
import matplotlib.pyplot as plt
models = ['Baseline', 'MambaGCN', 'Full']
mpjpe = [47.2, 41.1, 43.8]  # 替换为实际数值
plt.bar(models, mpjpe)
plt.ylabel('MPJPE (mm)')
plt.title('Human3.6M Performance')
plt.savefig('performance_comparison.png', dpi=300)
print('✅ 图表已生成: performance_comparison.png')
"

# LaTeX表格
python3 -c "
table = '''
\\begin{table}[h]
\\centering
\\caption{Performance on Human3.6M}
\\begin{tabular}{lcc}
\\toprule
Method & MPJPE (mm) & Params (M) \\\\
\\midrule
Baseline & 47.2 & 0.77 \\\\
MambaGCN & 41.1 & 1.07 \\\\
Full & 43.8 & 1.15 \\\\
\\bottomrule
\\end{tabular}
\\end{table}
'''
with open('performance_table.tex', 'w') as f:
    f.write(table)
print('✅ LaTeX表格: performance_table.tex')
"
```

---

## 🛠️ 常用工具命令

### 📊 性能分析
```bash
# 模型参数统计
python3 -c "
from model.MotionAGFormer import MotionAGFormer
model = MotionAGFormer(use_mamba_gcn=True)
params = sum(p.numel() for p in model.parameters())
print(f'参数量: {params:,} ({params/1e6:.1f}M)')
"

# 推理速度测试
python3 -c "
import torch, time
from model.MotionAGFormer import MotionAGFormer
model = MotionAGFormer(use_mamba_gcn=True).cuda()
x = torch.randn(1, 243, 17, 2).cuda()
# 预热
for _ in range(10): _ = model(x)
# 测试
start = time.time()
for _ in range(100): _ = model(x)
avg_time = (time.time() - start) * 10  # ms
print(f'推理时间: {avg_time:.1f}ms')
"
```

### 🔧 故障排除
```bash
# 内存不足解决
python3 scripts/train_real.py --batch_size 16 --gradient_accumulation_steps 4

# 训练不收敛解决  
python3 scripts/train_real.py --lr 5e-5 --warmup_steps 1000

# GPU利用率低解决
python3 scripts/train_real.py --num_workers 8 --pin_memory
```

---

## 📁 重要文件位置

### 📂 核心脚本
- `scripts/train_real.py` - 真实数据训练
- `baseline_validation_real.py` - 性能验证
- `demo_real.py` - 演示和可视化
- `compare_data_performance.py` - 性能对比

### ⚙️ 配置文件
- `configs/h36m/MotionAGFormer-base.yaml` - 推荐配置
- `configs/h36m/MotionAGFormer-large.yaml` - 高性能配置

### 📊 数据位置
- `data/motion3d/human36m/raw/` - 真实Human3.6M数据集

---

## 🎯 预期成果目标

### 📈 性能指标
- **目标MPJPE**: < 40mm (Human3.6M)
- **改进幅度**: > 10% vs 基线
- **推理速度**: < 50ms per sequence

### ⏱️ 时间投入
- **大规模训练**: 100-200 GPU小时
- **超参数调优**: 50-100 GPU小时  
- **结果分析**: 1-2 周

### 🏆 最终产出
- SOTA性能模型检查点
- 完整实验数据和对比
- 论文质量的表格图表
- 可重现的训练流程

---

## 🆘 快速求助

### 常见问题
1. **训练OOM**: 减少batch_size或使用梯度累积
2. **不收敛**: 降低学习率，增加warmup
3. **速度慢**: 增加num_workers，优化数据加载

### 支持资源
- 📖 详细文档: [CLIENT_POST_DELIVERY_GUIDE.md](CLIENT_POST_DELIVERY_GUIDE.md)
- 🔧 验证脚本: `tests/*.py`
- 📊 分析工具: `analysis/*.py`

---

**💡 提示**: 首次使用建议先运行小规模实验验证流程，再进行大规模训练！ 