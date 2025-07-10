#!/bin/bash
# 🚀 MotionAGFormer + MambaGCN 快速开始训练脚本
# 
# 使用方法:
#   ./quick_start_training.sh [baseline|mamba_gcn|full] [small|base|large] [gpu_count]
#
# 示例:
#   ./quick_start_training.sh mamba_gcn base 1    # 单GPU训练MambaGCN基础模型
#   ./quick_start_training.sh full large 4       # 4GPU训练完整架构大模型

set -e  # 遇到错误立即退出

# 默认参数
MODEL_TYPE=${1:-"mamba_gcn"}  # baseline, mamba_gcn, full
MODEL_SIZE=${2:-"base"}       # xsmall, small, base, large
GPU_COUNT=${3:-1}             # GPU数量

# 配置文件路径
CONFIG_FILE="configs/h36m/MotionAGFormer-${MODEL_SIZE}.yaml"

# 创建时间戳目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_DIR="checkpoints/training_${MODEL_TYPE}_${MODEL_SIZE}_${TIMESTAMP}"

# 根据模型大小调整批次大小
case $MODEL_SIZE in
    "xsmall")
        BATCH_SIZE=128
        EPOCHS=150
        ;;
    "small")
        BATCH_SIZE=96
        EPOCHS=200
        ;;
    "base")
        BATCH_SIZE=64
        EPOCHS=200
        ;;
    "large")
        BATCH_SIZE=32
        EPOCHS=300
        ;;
    *)
        echo "❌ 无效的模型大小: $MODEL_SIZE"
        echo "   支持的选项: xsmall, small, base, large"
        exit 1
        ;;
esac

# 验证模型类型
case $MODEL_TYPE in
    "baseline"|"mamba_gcn"|"full")
        ;;
    *)
        echo "❌ 无效的模型类型: $MODEL_TYPE"
        echo "   支持的选项: baseline, mamba_gcn, full"
        exit 1
        ;;
esac

# 打印配置信息
echo "🚀 MotionAGFormer + MambaGCN 快速训练启动器"
echo "=============================================="
echo "📊 训练配置:"
echo "   🎯 模型类型: $MODEL_TYPE"
echo "   📏 模型大小: $MODEL_SIZE"
echo "   🖥️  GPU数量: $GPU_COUNT"
echo "   📦 批次大小: $BATCH_SIZE"
echo "   🔄 训练轮数: $EPOCHS"
echo "   📁 保存目录: $SAVE_DIR"
echo "   ⚙️  配置文件: $CONFIG_FILE"
echo "=============================================="

# 预检查
echo "🔍 环境预检查..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3"
    exit 1
fi

# 检查PyTorch和CUDA
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'✅ CUDA可用: {torch.cuda.is_available()}')"
if [ $GPU_COUNT -gt 1 ]; then
    python3 -c "import torch; print(f'✅ GPU数量: {torch.cuda.device_count()}')"
fi

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查数据
echo "📊 验证数据集..."
python3 -c "
from data.reader.real_h36m import DataReaderRealH36M
try:
    reader = DataReaderRealH36M(n_frames=243)
    print('✅ 真实Human3.6M数据可用')
except Exception as e:
    print(f'❌ 数据加载失败: {e}')
    exit(1)
"

# 创建保存目录
mkdir -p "$SAVE_DIR"
echo "📁 创建保存目录: $SAVE_DIR"

# 保存训练配置
cat > "$SAVE_DIR/training_config.json" << EOF
{
    "model_type": "$MODEL_TYPE",
    "model_size": "$MODEL_SIZE", 
    "gpu_count": $GPU_COUNT,
    "batch_size": $BATCH_SIZE,
    "epochs": $EPOCHS,
    "config_file": "$CONFIG_FILE",
    "save_dir": "$SAVE_DIR",
    "timestamp": "$TIMESTAMP"
}
EOF

# 构建训练命令
if [ $GPU_COUNT -gt 1 ]; then
    # 多GPU训练
    TRAIN_CMD="python3 -m torch.distributed.launch --nproc_per_node=$GPU_COUNT scripts/train_real.py"
    ADJUSTED_BATCH_SIZE=$((BATCH_SIZE / GPU_COUNT))
else
    # 单GPU训练
    TRAIN_CMD="python3 scripts/train_real.py"
    ADJUSTED_BATCH_SIZE=$BATCH_SIZE
fi

# 添加训练参数
TRAIN_CMD="$TRAIN_CMD \
    --config $CONFIG_FILE \
    --model_type $MODEL_TYPE \
    --epochs $EPOCHS \
    --batch_size $ADJUSTED_BATCH_SIZE \
    --device cuda \
    --save_dir $SAVE_DIR"

echo "🎯 训练命令预览:"
echo "   $TRAIN_CMD"
echo ""

# 确认开始训练
read -p "🚀 开始训练? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "✅ 开始训练..."
    echo "📝 训练日志将保存到: $SAVE_DIR/training.log"
    echo "📊 实时监控命令: tail -f $SAVE_DIR/training.log"
    echo ""
    
    # 启动训练并保存日志
    $TRAIN_CMD 2>&1 | tee "$SAVE_DIR/training.log"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 训练完成！"
        echo "📁 模型保存在: $SAVE_DIR"
        echo "📊 查看结果: ls -la $SAVE_DIR"
    else
        echo ""
        echo "❌ 训练失败，请检查日志: $SAVE_DIR/training.log"
        exit 1
    fi
else
    echo "❌ 训练已取消"
    echo "📁 已创建的目录: $SAVE_DIR"
    exit 0
fi

# 训练后分析
echo ""
echo "📊 训练后分析..."
if [ -f "$SAVE_DIR/metrics.json" ]; then
    python3 -c "
import json
try:
    with open('$SAVE_DIR/metrics.json', 'r') as f:
        metrics = json.load(f)
        if 'mpjpe' in metrics and len(metrics['mpjpe']) > 0:
            best_mpjpe = min(metrics['mpjpe'])
            final_mpjpe = metrics['mpjpe'][-1]
            print(f'🏆 最佳MPJPE: {best_mpjpe:.2f}mm')
            print(f'📈 最终MPJPE: {final_mpjpe:.2f}mm')
            print(f'📊 训练轮数: {len(metrics[\"mpjpe\"])}')
        else:
            print('⚠️  指标文件为空或格式错误')
except Exception as e:
    print(f'❌ 无法读取指标文件: {e}')
"
else
    echo "⚠️  未找到指标文件"
fi

echo ""
echo "✅ 训练流程完成！"
echo "📋 后续步骤:"
echo "   1. 查看训练日志: cat $SAVE_DIR/training.log"
echo "   2. 验证模型性能: python3 baseline_validation_real.py --model_path $SAVE_DIR/best_model.pth"
echo "   3. 运行演示: python3 demo_real.py real --model_path $SAVE_DIR/best_model.pth"
echo "   4. 继续超参数调优: 参考 CLIENT_POST_DELIVERY_GUIDE.md" 