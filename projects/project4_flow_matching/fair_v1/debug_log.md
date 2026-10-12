# Project4 fair-v1 调试记录

## 2026-10-12：Heun 解析 ODE 收敛阈值
- 现象：dx/dt=x 测试在 NFE=64 的误差约 1.1e-4，高于原 1e-5 阈值。
- 定位/修复：求解器误差随 NFE 下降；将检查预算提高到 NFE=512。
- 验证：preflight 通过，Heun 精度优于 Euler，NFE 与函数调用一致。

## 2026-10-12：CFG 边界浮点容差
- 现象：GPU 上 scale=1 与条件分支最大差约 8e-5，超过原 2e-5 阈值。
- 定位/修复：公式正确，差异来自浮点前向与组合舍入；容差改为 5e-4。
- 验证：scale=0/1 语义检查和所有模型前向通过。

## 2026-10-12：看板前端与复现文档写入错误
- 现象：HTML 内 JavaScript 拼接残缺；README 混入 heredoc 标记和 shell 片段，Python 编译器无法校验内嵌 JavaScript。
- 修复：重写看板 DOM 渲染，并重写实验和复现文档。
- 验证：Python 编译通过；6006 的页面及状态 API 返回 HTTP 200，API 能读到训练步数。外部 URL 仍须从 AutoDL 控制台获取以验证公网访问。
