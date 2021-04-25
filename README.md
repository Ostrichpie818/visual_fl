# 联邦学习的可视化基本实现

基于streamlit的简单web应用，联邦学习单机实现部分代码及结构借鉴于github项目https://github.com/WHDY/FedAvg

### 环境

1.python3.8.5

2.pytorch1.8.1 cpu版本及以上

可以直接打开https://share.streamlit.io/ostrichpie818/visual_fl/main/server.py 进行操作，但是因为streamlit的app部署依托于github，训练成果不能保存，就只能训练和复现本机push上去的结果

### 不足和操作注意事项

1.在每次调参数时都会刷新界面重新执行，所以容易出现没有改变完参数就开始训练的情况
  操作方法：回到主页来调整侧边栏参数

2.源代码server.py 117行偶尔会出现越界导致程序崩掉
