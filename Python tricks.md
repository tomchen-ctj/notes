# Python编程笔记

1. ##  python中*与**的用法

   ### 序列解包

   一般用于前缀时，起到“打包传值，拆包传值”的作用

   前缀用法中，*用于位置传值，**用于字典传值

   对于python原生数据结构：list/tuple/dict等都可适用

   ```python
   #*前缀的用法
   # 1.按序合并
   arr1 = (12, 13)
   arr2 = (*arr1, 100) # (100, 12, 13)
   # 2.有序打印所有元素
   arr1 = [2, 3]
   arr2 = [4, 5]
   print(*arr1, *arr2) # [2,3,4,5]
   # 3.元组拆包
   arr1 = [2, 3, 4, 5]
   e1, *e2, e3 = arr1
   e1 #[2]
   e2 #[3,4]
   e3 #[5]
   
   #**前缀的用法
   # 1.合并两个字典
   dic1 = {1:2}
   dic2 = {**dic1, 2:4} # {1:2, 2:4}
   # 2.初始化一个以lis为键的字典
   lis = [1,2]
   dic = dict(zip(lis, [[]]*len(lis))) # {1:[], 2:[]}
   # 3.字典传值
   queryfeature_kwargs = {
               'input_dim': self.hidden_size,
               'hidden_dims': (1024,),
               'output_dim' : self.key_query_dim,
               'use_batchnorm': True,
               'use_relu': True,
               'dropout': 0,
           }
   image_queryfeature_mlp = networks.build_mlp(**queryfeature_kwargs)#将对应位置的键值对作为函数参数传入
   ```

   

