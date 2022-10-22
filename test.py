#读写文件
'''of=open('E:/B/pycharm_files/text.txt','a+') #a+表示不存在文件就创建，存在就在文件里面追加内容
print('hellow',file=of)
print('hellow','李明春',file=of)
of.close()
'''
#转义字符
'''print('hellow\nLMC')#换行
print('hellow\tLMC')#制表符4个空格
print('你说：\'大家好\'')#要是用某一个符号就要用到\
'''

#二、八十六进制
'''print(0b01101)#二进制
print(0o1764)#八进制
print(0xe28a)#十六进制'''

#浮点类型
'''print(1.1+2.2)#可以引入库函数decimal
from decimal import Decimal
print(Decimal('1.1')+Decimal('2.2'))
print(2.2+3.3)'''

#字符型，单引号，双引号，三引号都可以，三引号可以换行写
'''strs=\'''我要发
100篇SCI\'''
print(strs)'''

#数据类型转换--在数据拼接的时候用得上
'''name='张三'  #python不需要在前面定义，或有默认类型C\c++则不行
age=20
#print('我叫'+name+'今年'+age+'岁') #用int() str() float()进行；类型转换
print('我叫'+name+'今年'+str(age)+'岁')'''

#函数input
'''
a=input('请输入：')
b=input('请输入：')
print(a+b,type(a))#默认str类型，计算则需要将输入的转换成整型'''
#python运算跟c、c++不一样/表示出发而c整除，python//才是整除

#print(9//-4)#向下取整
#and   or   not   in   not in逻辑运算符
#&按位与：二进制与，都是1才为真


#顺序、选择、循环
#列表表示：[]、list(),元组表示：()、tuple(),字典表示{}、diet(),集合表示：set()
#常用的if语句和嵌套就没写
'''a=int(10)
if a>2:
    pass
else:
    pass  #pass只是当做一个占位符，不让计算机报错'''
#创建列表
'''lst=['a','b','l','g','i']
lst1=list(['c','k'])
print(lst.index('a'))#打印这个列表中第一个该字母
print(lst.index('a',0,4))#指定从哪一位开始检索
print(lst[0:3:2]) #切片操作起始位置:终点位置:步长
print(lst[::-1])#起点终点都默认步长为-1'''

#判断某一个对象是否在某一个对象当中可以用in   not in
'''lst=['lmc','lk',20,30]
print('lmc' in lst)
for i in lst:
    print(i)#遍历'''
#列表添加删减
'''lst=[10,20,30,80]
lst2=['lmc','hk']
lst.append(22)
print(lst)
print(id(lst))#位置
#lst.append(lst2)  #直接添加

lst.extend(lst2)  #分开添加
lst.insert(2,88)
lst[2:]=lst2  #切片替换
print(lst)'''

#列表移除
'''lst=[20,90,66,34]
lst.remove(20)  #不赋值给新变量说明是在原来的地址原来的空间上的变量上进行更改所以不需要重新定义新的变量
print(lst)
#lst[1:3]=0#就是删除某一指定的元素
lst.pop(1)#指定索引移除元素,不指定索引则删除最后一个元素
print(lst)'''

#列表排序
'''lst=[20,90,66,34]
lst.sort()
print(lst)
lst.sort(reverse=True)
print(lst)
new_lst=sorted(lst)#使用内置函数排序生成一个新的列表
llst=sorted(lst,reverse=True)
print('new_lst:',new_lst)
print('llst:',llst)'''

#生成列表
'''lst=[i for i in range(1,10)]
print(lst)
llst=[i*2 for i in range (1,5)]
print(llst)'''

#字典的创建方式1.{}  2.关键字dict()
abb={'张三':100,'李四':98}
print(abb)
ddic=dict(name='lmc',age=22)
print(ddic,type(ddic))
#获取字典中的元素
print(abb['张三'])
print(ddic.get('name'))
print('啦啦啦默认值为：',ddic.get('啦啦啦',100)) #当不存在这个对象的时候就设置默认值100
#字典元素删除
del ddic['name']
print('删除后的ddic为：',ddic)
#clear.ddic()   删除字典所有元素
#新增
ddic['权值']=20
ddic['王四']='厉害'
print('新增后的ddic为：',ddic)
print(ddic.keys())  #获取字典当中所有的建
print(ddic.values()) #获取字典所有的值
print(ddic.items())  #获取字典的所有组（返回的是元组）
#字典的遍历
for i in ddic:
    print('遍历字典的键',i)#遍历字典的键
    print('遍历字典的值',ddic[i],ddic.get(i))

#字典生成式
person={'A','B','C'}
grade={98,85,70}
dd={i:j  for i,j in zip(person,grade)}
print(dd)

#元组---元组跟列表从表面上看只是括号不一样
pp=('lmc','lkq',66)
print(pp,type(pp))
po=tuple(('lmc','lkq',66))
print(po)
print(('hds',))#如果元组只有一个元素那么后面必须加上逗号
t=(30,[20,50],66)
t[1][1]=10  #元组内的元素不能变，如果元组内的元素有可变类型那么只能更改这个元素内部的内容而不能将这个元素直接修改掉
print(t)

#集合----也是{}但是只是单个元素不是成对存在，且元素不能重复
uo={'km',20}
print(type(uo))
print(set([10,20,6,10,6]))  #直接将列表中的元素变成集合当中的元素同时除去相同的元素--且元素是无序的
print(set('python'))
print([10,20]==[20,10])  #列表有序而元组只要里面元素一样就是相等的
#判断集合的关系
s1={10,20,30,40,50}
s2={10,40}
s3={10,60,20}
print(s2.issubset(s2))  #s2是s1的子集
print(s2.isdisjoint(s3))#有无交集

