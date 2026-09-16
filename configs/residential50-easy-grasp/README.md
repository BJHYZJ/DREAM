# 便于抓放的 50 房屋任务

此配置使用便于抓取、放置的物体，保留原多物体任务的全部 50 个房屋、布局、移动路径和随机种子 42，替换其中 33 场的目标模型。它与早期 diverse-object 配置分别评测，成功率各自统计。

6 类、21 个模型均使用原始尺寸；每类安排 8–9 场。候选从已有 541 个模型记录中筛选，排除细薄物体、过大物体、细长物体及放置不稳定模型。22 个候选中 21 个通过短物理检查；Apple_19 因偏移落点放下后速度过高被排除。候选模型不要求每个场景各不相同。

短检查包含托盘支撑和盘上中心、左右各 2 cm 三个落点。它不是机器人完整抓取试验，场景成功由独立完整执行判定。

- `task_manifest.json`：50 个任务与房间地图的校验值。
- `object_changes.csv`：每个场景替换前后的物体。
- `object_selection.json`：接受和剔除模型及其短检查依据。

运行 `./scripts/run_residential50_fast.sh --workers-per-gpu 4 --output /新的结果目录 --execute`。每场 900 秒机器人动作上限、2700 秒执行看门狗，GPU 0、1 各 4 场、合计最多 8 场并行；不加 `--execute` 只查看计划。

DREAM Fetch 控制器在完整 50 场实验中完成 36 场任务，全部通过独立物理、观测和收臂复核，严格成功率为 **36/50（72%）**。14 场失败均计入分母。[逐场结果](../../reports/residential50-20260915/v55-full50-final.json) · [公共源码一致性](../../reports/residential50-20260915/v55-public-promotion.json) · [完整执行记录](../../reproducibility/evidence/residential50-easy-v55/README.md)。视频资格单独记录。
