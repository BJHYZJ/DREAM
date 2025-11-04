# DREAM: Dynamic Resilient Spatio-Semantic Memory with Hybrid Localization for Mobile Manipulation
![log](docs/dream.jpg)


rerun-sdk==0.26.1
numpy==1.24.3


网络端口被占用时

```bash
sudo lsof -i:4403
```

暂停rtabmap
```bash
ros2 service call /rtabmap/rtabmap/pause std_srvs/srv/Empty {}
```

恢复rtabmap
```bash
ros2 service call /rtabmap/rtabmap/resume std_srvs/srv/Empty {}
```