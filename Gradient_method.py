import sympy as sp
import numpy as np


# 梯度
def grad(f,symbol_list):
    grad_list = []
    for i in symbol_list:
        grad_list.append(sp.diff(f,i))
    return sp.Matrix(grad_list)

# {变量:数值},绑定
def symbol_value(symbol_list,value_array):
    subs_dict = {}
    for i in range(len(symbol_list)):
        subs_dict[symbol_list[i]] = value_array[i][0]
    return subs_dict

# 梯度上升
def grad_up(goal_func,symbol_list,initial_value_list,
            step_value=5e-2,max_num=1e5,abs_diff=1e-6,step_decay=False):
    print('start gradient decent')
    grad_func_mat = grad(goal_func,symbol_list)
    print('grad func ok')
    # 起始点
    x = np.array([initial_value_list]).T
    subs_dict = symbol_value(symbol_list,x)
    goal_func_old = -1
    goal_func_new = goal_func.evalf(subs=subs_dict)
    print('x0 ok')
    # 变量更新
    count = 0
    while abs(goal_func_new-goal_func_old)>abs_diff and count<max_num:
        goal_func_old = goal_func_new

        x = x + step_value*np.array(grad_func_mat.evalf(subs=subs_dict))
        subs_dict = symbol_value(symbol_list,x)
        goal_func_new = goal_func.evalf(subs=subs_dict)
        count += 1
        if step_decay:
            if count >= 20:
                step_value = step_value * 0.5
            if count >= 50:
                step_value = step_value* 0.5

        print(count)
        print(x)
        print(goal_func_new)
        
        print('\n')
    
    if count >= max_num:
        print('超出最大设定迭代次数，收敛失败')
        return -1
    
    return x,goal_func_new

