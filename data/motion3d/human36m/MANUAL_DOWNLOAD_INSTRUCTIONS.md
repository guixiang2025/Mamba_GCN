# Human3.6M数据手动下载指导

## 下载链接
https://drive.google.com/file/d/1WWoVAae7YKKKZpa1goO_7YcwVFNR528S/view?usp=sharing

## 下载步骤
1. 点击上面的链接访问Google Drive
2. 点击下载按钮下载文件
3. 将文件保存到: `data/motion3d/human36m/raw`
4. 运行验证: `python3 download_and_setup_h36m_real.py --verify`

## 预期文件
- 压缩文件 (.zip) 或
- NPZ数据文件 (.npz)

## 完成后
运行 `python3 download_and_setup_h36m_real.py --setup` 来配置数据
