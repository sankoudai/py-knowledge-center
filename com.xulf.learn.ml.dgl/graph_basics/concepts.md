DGLGraph:  node, edges, features on them
- node, edge Ids always start from 0


- 图计算基础
    - 图的构造
    - 图的属性/信息
    - 获取子图
    - 图上的特征计算
  
（两个基本任务： 构建图， 计算edge_softmax）

- 图模型基础
    - gnn module: 一个图g上的计算器，输入g，输出节点/边上的更新特征
        为了可复用，一般计算相关的边/节点特征会以参数形式传入
      (
        基本任务
        1) 一般图上的gnn module
        2) 二部图上的gnn module
     )
    - 图模型的训练
      - 无邻居采样的训练任务
      - 邻居采样与训练任务
      - 基本任务：
        - 节点分类任务： 子图采样， 
        - 边预测任务： 
      

  

- 常见图模型的实现
