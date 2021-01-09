"""
以1230的为基准来进行调试
"""

import sys
import pandas as pd
import numpy as np
import math
import pyqtgraph as pg
import cgitb
from PyQt5.Qt import *
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QMainWindow
import LNG_demo_1229

def sampling(Rf, lmda, k_wbl):
    """
    蒙特卡洛逆变换抽样原理，类似于一个反函数求解
    Params：
        Rf:  01之间的随机数
        lmda: 尺度参数
        k_wbl:  形状参数
    return:
        tf 满足威布尔分布的随机数
    """
    tf = (-math.log(1 - Rf)) ** (1 / k_wbl) / lmda
    return tf


def mean_life(t, k_wbl, flag):
    """
    设备服从威布尔分布，平均可用度为0.99时确定的设计寿命t
    """
    if flag == 1:
        T = t / ((k_wbl + 1) / 100) ** (1 / k_wbl)

    else:
        """
        平均可用度为0.95时确定的设计寿命t
        """
        T = t / (5 * (k_wbl + 1) / 100) ** (1 / k_wbl)
    return T


def cost_pm(idx_pm, tmp_count):
    """
    一次PM的费用是固定的
    params：
        idx_pm：预防维修的设备的编号
        tmp_count:用以记录常数的字典
    return：
        不同设备一次PM对应的费用
    """
    return tmp_count[idx_pm]['c_pm']


def cost_cm(idx_fault, tmp_count):
    """
    一次CM的费用是固定的
    params：
        idx_fault：预防维修的设备的编号
        tmp_count:用以记录常数的字典
    return：
        不同设备一次CM对应的费用
    """
    return tmp_count[idx_fault]['c_cm']


def cost_stop(idx, tmp_count):
    """
    一次产能损失的费用是固定的
    params：
        idx：预防维修的设备的编号
        tmp_count:用以记录常数的字典
    return：
        不同设备一次产能损失对应的费用
    """
    return tmp_count[idx]['cs']


class pump_thread(QThread):
    sinout = pyqtSignal(dict) #定义一个信号，调用run，即可发射该信号
    def __init__(self):
        super().__init__()
    def run(self):
        global pump_df
        life1_value = pump_df['life1_value']
        life2_value = pump_df['life2_value']
        k_wbl = 3
        flag = 1
        mean_home = mean_life(life1_value, k_wbl, flag)
        mean_import = mean_life(life2_value, k_wbl, flag)
        self.num_home = pump_df['num_home']
        self.num_import = pump_df['num_import']
        self.num_low = pump_df['num_low']
        np.random.seed(1)
        rand_list = np.random.rand(int((pump_df['high']-pump_df['low'])/pump_df['step']*50*pump_df['times']))
        # rand_list = np.random.rand(10000)
        #要先读入设备的数目，以及仿真的次数，再来确定随机数的多少
        life1 = [sampling(i, 1 / mean_home, k_wbl) for i in rand_list]
        life2 = [sampling(i, 1 / mean_import, k_wbl) for i in rand_list]
        dic_home = {
            "c_pm": pump_df['home_pm'],
            'c_cm': pump_df['home_cm'],
            'cs': pump_df['home_cs'],
            'mean_life': mean_home,
            'life': life1
        }
        dic_import = {
            "c_pm": pump_df['import_pm'],
            'c_cm': pump_df['import_cm'],
            'cs': pump_df['import_cs'],
            'mean_life': mean_import,
            'life': life2
        }

        d_year = 360
        self.h_time = int(30 * pump_df['h_time'])
        pump_tmp = {}
        for idx in range(self.num_home):
            pump_tmp[idx] = dic_home.copy()
            pump_tmp[idx]['life'] = life1.copy()[:-(idx + 1) * 10]

        for idx in range(self.num_home, self.num_home + self.num_import):
            pump_tmp[idx] = dic_import.copy()
            pump_tmp[idx]['life'] = life2.copy()[:-(idx + 1) * 10]
        pump_tmp['h_time'] = self.h_time
        pump_tmp['num_home'] = pump_df['num_home']
        pump_tmp['num_import'] = pump_df['num_import']
        pump_tmp['num_low'] = pump_df['num_low']

        self.pump_tmp = pump_tmp
        initial_time = d_year - self.h_time
        year = 25
        th_start_list = [initial_time + i * d_year for i in range(year)]
        th_end_list = [(i + 1) * d_year for i in range(year)]
        t_table = pd.DataFrame()
        t_table['th_start'] = th_start_list
        t_table['th_end'] = th_end_list
        self.t_table = t_table #记录高负荷期期间的dataframe
        cols = ['n_pm', 'n_cm', 'n_stop', 'c_pm', 'c_cm', 'c_stop', 'sum_cost'] #需要记录的各项参数
        self.times = int(pump_df['times'])
        low = int(pump_df['low'])
        high = int(pump_df['high']) + 1
        step = int(pump_df['step'])
        range1 = range(low, high, step) #要仿真的时间区间
        self.range1 = range1
        idxs = [str(tp) for tp in range1]
        df = pd.DataFrame(np.zeros((len(idxs), len(cols))), index=idxs, columns=cols)
        all_record = {}
        for idx in range(self.num_home + self.num_import):
            all_record[idx] = df.copy()
        self.all_record = all_record



        self.col_list = ['t1', 't2', 'state', 'life', 'n_pm', 'n_cm', 'n_stop', 'c_pm', 'c_cm', 'c_stop', 'now',
                         'sum_cost']

        self.pump_count()
        self.sinout.emit(self.all_record)

    def pump_count(self):
        for tp in tqdm(self.range1):
            for _ in range(self.times):
                df_ = pd.DataFrame([[0]*len(self.col_list)], columns=self.col_list)
                dic = {'t_now_list': [0]}
                for idx_machine in range(self.num_home+self.num_import):
                    dic[idx_machine] = df_.copy()
                    dic[idx_machine].loc[dic[idx_machine].index[-1], 'life'] = self.pump_tmp[idx_machine]['life'].pop()
                for idx in range(self.num_low):
                    dic[idx]['state'] = 1
                for idx_t in range(len(self.t_table)):
                    t_next_start = self.t_table.iloc[idx_t, 0]
                    flag, dt = self.initial_process(dic, tp)
                    while dic['t_now_list'][-1] + dt < t_next_start:
                        if flag == 1:
                            dic = self.low_fault(dic, self.pump_tmp)
                        else:
                            dic = self.low_pm(dic, tp, self.pump_tmp)
                        flag, dt = self.initial_process(dic, tp)
                    dic = self.low2high(dic, tp, self.pump_tmp, t_next_start)
                    t_next_end = self.t_table.iloc[idx_t, 1]
                    dt = self.initial_high(dic)
                    while dic['t_now_list'][-1] + dt < t_next_end:
                        dic = self.high_fault(dic, self.pump_tmp)
                        dt = self.initial_high(dic)
                    num = self.num_home + self.num_import - self.num_low
                    dic = self.high2low(dic, t_next_end, num)

                for idx_machine in range(len(dic) -1):
                    self.all_record[idx_machine].loc[str(tp), 'n_pm'] += np.nansum(dic[idx_machine]['n_pm'])
                    self.all_record[idx_machine].loc[str(tp), 'n_cm'] += np.nansum(dic[idx_machine]['n_cm'])
                    self.all_record[idx_machine].loc[str(tp), 'c_pm'] += np.nansum(dic[idx_machine]['c_pm'])
                    self.all_record[idx_machine].loc[str(tp), 'c_cm'] += np.nansum(dic[idx_machine]['c_cm'])
                    self.all_record[idx_machine].loc[str(tp), 'c_stop'] += np.nansum(dic[idx_machine]['c_stop'])
                    self.all_record[idx_machine].loc[str(tp), 'sum_cost'] += np.nansum(dic[idx_machine]['sum_cost'])

    def initial_process(self, dic, tp):
        """
        低负荷期判断下一次维护是什么时候
        params:
            dic:记录所有的运行信息
            tp： 维修周期
        return
            flag， dt
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_p = [tp - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dt = min(t_p)
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        if dt < dtf:
            flag = 2
        else:
            flag = 1
            dt = dtf
        return flag, dt

    def initial_high(self, dic):
        """
        高负荷期内判定下一次操作的时间
        params:
            dic:记录所有的运行信息
            tp： 维修周期
        return
            dt 下一次故障间隔
        """
        idx_using = [i for i in range(len(dic)-1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dt = min(t_f)
        return  dt

    def find_min_idx(self, dic, idx_using):
        """
        找到进口压缩机中未运行的设备中，运行时间最低的idx
        只有高负荷期，进口压缩机发生故障，才会调用这个函数
        """
        range_tmp = range(len(dic) - 1)
        idx_no_use = [i for i in range_tmp if i not in idx_using]
        t_f = [dic[i].loc[dic[i].index[-1], 't1'] for i in idx_no_use]
        idx = idx_no_use[t_f.index(min(t_f))]
        return idx

    def low_fault(self, dic, tmp_count):

        """
        低负荷期发生故障，拟考虑运行时间较短的备件来接替运行
        params:
            dic:记录信息的字典,改变
            tmp_count:设备固有数据，不会变
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
        idx_fault = []
        for idx_ in range(len(idx_using)):
            if t_f[idx_] == dtf:
                idx_fault.append(idx_using[idx_])
        # 不管怎么样，以下这些操作，两种方案应该都一样的
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0
        for idx_f in idx_fault:
            x = dic[idx_f].index[-1]  # 这个时候x已经自加了,df自动更新的
            # 只考虑fault只有一台机器损坏
            dic[idx_f].loc[x, 't1'] = 0
            dic[idx_f].loc[x, 'state'] = 3
            dic[idx_f].loc[x, 'life'] = tmp_count[idx_f]['life'].pop()
            dic[idx_f].loc[x - 1, 'n_cm'] = 1
            dic[idx_f].loc[x - 1, 'c_cm'] = cost_cm(idx_f, tmp_count)
            dic[idx_f].loc[x - 1, 'sum_cost'] = dic[idx_f].loc[x - 1, 'c_cm']

        idx = self.find_min_idx(dic, idx_using)
        dic[idx].loc[dic[idx].index[-1], 'state'] = 1
        return dic

    def low_pm(self, dic, tp, tmp_count):

        """
        低负荷期发生预防性维修的函数，拟采用运行时间较少的备件来接替运行
        同时考虑多台设备一起维修的情况
        params:
            dic:记录信息的字典
            tp:维修周期
            tmp_count:设备固有数据，不会变
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_p = [tp - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtp = min(t_p)
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtp)
        idx_pm = []
        for idx_ in range(len(idx_using)):
            if t_p[idx_] == dtp:
                idx_pm.append(idx_using[idx_])
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtp
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0

        for idx_ in idx_pm:
            x = dic[idx_].index[-1]  # 这个时候x已经自加了
            dic[idx_].loc[x, 't1'] = 0
            dic[idx_].loc[x, 'state'] = 3
            dic[idx_].loc[x, 'life'] = tmp_count[idx_]['life'].pop()
            dic[idx_].loc[x - 1, 'n_pm'] = 1
            dic[idx_].loc[x - 1, 'c_pm'] = cost_pm(idx_, tmp_count)
            dic[idx_].loc[x - 1, 'sum_cost'] += dic[idx_].loc[x - 1, 'c_pm']

            # 这个得分情况讨论吧，如果同时维修数大于2，则直接原地维修
        # 如果小于2，则可以考虑接替
        n_pm = len(idx_pm)
        if n_pm > 2:
            for idx_u in idx_pm:
                dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
        elif n_pm == 2:
            idx_using = [i for i in range(len(dic) - 1) if i not in idx_pm]
            for idx_u in idx_using:
                dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
        else:
            idx_u = self.find_min_idx(dic, idx_using)
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
        return dic

    def low2high(self, dic, tp, tmp_count, t_next_start):
        """
        设备从低负荷期到高负荷期的转换过程，所有设备都处于工作状态
        params:
            dic:记录信息的字典,改变
            tmp_count:设备固有数据，不会变
            t_next_start: 高负荷期开始的时间
            tp:维修周期
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        dt = t_next_start - dic['t_now_list'][-1]
        dic['t_now_list'].append(t_next_start)
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1  # 先进入停机状态，然后再确定哪些机器需要继续运行
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0
        idx_using = range(len(dic) - 1)
        # 判定是否会在高负荷期内达到预防周期，是则提前进行预防维修，否则不用操作
        for idx_u in idx_using:
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
            if dic[idx_u].loc[dic[idx_u].index[-1], 't1'] + tmp_count['h_time'] >= tp:
                x = dic[idx_u].index[-1]
                dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1']
                dic[idx_u].loc[x + 1, 't1'] = 0  # 因为进行了pm，所以下一阶段的t1一定是0
                dic[idx_u].loc[x + 1, 'life'] = tmp_count[idx_u]['life'].pop()
                dic[idx_u].loc[x + 1, 'state'] = 1
                dic[idx_u].loc[x, 'n_pm'] = 1
                dic[idx_u].loc[x, 'n_cm'] = 0
                dic[idx_u].loc[x, 'c_pm'] = cost_pm(idx_u, tmp_count)
                dic[idx_u].loc[x, 'c_cm'] = 0
                dic[idx_u].loc[x, 'c_stop'] = 0
                dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
                dic[idx_u].loc[x, 'sum_cost'] = dic[idx_u].loc[x, 'c_pm']

        return dic

    def high_fault(self, dic, tmp_count):
        """
        设备在高负荷期运行的模式，设备发生故障，产生产能损失，修复完成后，立即投入使用
        params：
            dic:记录信息的字典
            tmp_count:设备固有数据，不会变
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        idx_fault = []
        for idx_ in range(len(idx_using)):
            if t_f[idx_] == dtf:
                idx_fault.append(idx_using[idx_])
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
        # 不管怎么样，一下这些操作，两种方案应该都一样的
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1  # 先进入停机状态，然后再令0号机开始运行
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0

        for idx_f in idx_fault:
            x = dic[idx_f].index[-1]  # 这个时候x已经自加了
            dic[idx_f].loc[x, 't1'] = 0
            dic[idx_f].loc[x, 'life'] = tmp_count[idx_f]['life'].pop()
            dic[idx_f].loc[x - 1, 'n_cm'] = 1
            dic[idx_f].loc[x - 1, 'c_cm'] = cost_cm(idx_f, tmp_count)
            dic[idx_f].loc[x - 1, 'c_stop'] = cost_stop(idx_f, tmp_count)
            dic[idx_f].loc[x - 1, 'sum_cost'] = dic[idx_f].loc[x - 1, 'c_stop'] + dic[idx_f].loc[x - 1, 'c_cm']
        return dic

    def high_change1(self, dic, num):
        """
        策略1：停掉距离最近的num台泵,正常组
        params:
            dic:记录设备运行信息的字典，可变
            num：需要停掉的设备数目
        return:
            idx：num个需要停机的设备的编号
        """
        life_list = [dic[i].loc[dic[i].index[-1], 't1'] for i in range(len(dic) - 1)]
        sorted_nums = sorted(enumerate(life_list), key=lambda x: x[1])
        df = pd.DataFrame(sorted_nums, columns=['idx', 'num'])
        df['diff'] = df['num'].diff()  # 累减函数，从小往上1个步长
        df['diff'] = df['diff'].fillna(1000000)  # 这就保证了运行时间最小的设备一定不会停机
        df = df.sort_values(by="diff", axis=0, ascending=True)  # axis=0就是表示的一列
        range1 = df['idx'][:num]
        return range1

    def high2low(self, dic, t_next_end,  num):
        """
        设备从高负荷期向低负荷期切换的策略，这个策略比较多
        params
            dic:记录设备运行信息
            tmp_count:设备不变的信息
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        dt = t_next_end - dic['t_now_list'][-1]
        dic['t_now_list'].append(t_next_end)
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0

        idx_list = self.high_change1(dic, num)
        for idx in idx_list:
            dic[idx].loc[dic[idx].index[-1], 'state'] = 3
        return dic


class press_thread(QThread):
    sinout = pyqtSignal(dict)
    def __init__(self):
        super(press_thread, self).__init__()

    def run(self):
        global press_df
        life1_value = press_df['life1_value']
        life2_value = press_df['life2_value']
        k_wbl = 3
        flag = 1
        mean_home = mean_life(life1_value, k_wbl, flag)
        mean_import = mean_life(life2_value, k_wbl, flag)
        np.random.seed(1)
        self.num_home = press_df['num_home']
        self.num_import = press_df['num_import']
        self.num_high = press_df['num_high']
        rand_list = np.random.rand(1000)  # 要先读入设备的数目，以及仿真的次数，再来确定随机数的多少
        life1 = [sampling(i, 1 / mean_home, k_wbl) for i in rand_list]
        life2 = [sampling(i, 1 / mean_import, k_wbl) for i in rand_list]
        dic_home = {
            "c_pm": press_df['home_pm'],
            'c_cm': press_df['home_cm'],
            'cs': press_df['home_cs'],
            'mean_life': mean_home,
            'life': life1
        }
        dic_import = {
            "c_pm": press_df['import_pm'],
            'c_cm': press_df['import_cm'],
            'cs': press_df['import_cs'],
            'mean_life': mean_import,
            'life': life2
        }

        d_year = 360
        self.h_time = 30 * press_df['h_time']
        press_tmp = {}
        for idx in range(self.num_home):
            press_tmp[idx] = dic_home.copy()
            press_tmp[idx]['life'] = life1.copy()[:-(idx + 1) * 10]

        for idx in range(self.num_home, self.num_home + self.num_import):
            press_tmp[idx] = dic_import.copy()
            press_tmp[idx]['life'] = life2.copy()[:-(idx + 1) * 10]
        press_tmp['h_time'] = self.h_time
        press_tmp['num_home'] = press_df['num_home']
        press_tmp['num_import'] = press_df['num_import']
        press_tmp['num_high'] = press_df['num_high']


        self.press_tmp = press_tmp
        initial_time = d_year - self.h_time
        year = 25
        th_start_list = [initial_time + i * d_year for i in range(year)]
        th_end_list = [(i + 1) * d_year for i in range(year)]
        t_table = pd.DataFrame()
        t_table['th_start'] = th_start_list
        t_table['th_end'] = th_end_list
        self.t_table = t_table  # 记录高负荷期期间的dataframe
        cols = ['n_pm', 'n_cm', 'n_stop', 'c_pm', 'c_cm', 'c_stop', 'sum_cost']  # 需要记录的各项参数
        self.times = int(press_df['times'])
        low = int(press_df['low'])
        high = int(press_df['high']) + 1
        step = int(press_df['step'])
        range1 = range(low, high, step)  # 要仿真的时间区间
        self.range1 = range1
        idxs = [str(tp) for tp in range1]
        df = pd.DataFrame(np.zeros((len(idxs), len(cols))), index=idxs, columns=cols)
        all_record = {}
        for idx in range(self.num_home + self.num_import):
            all_record[idx] = df.copy()
        self.all_record = all_record
        self.col_list = ['t1', 't2', 'state', 'life', 'n_pm', 'n_cm', 'n_stop', 'c_pm', 'c_cm', 'c_stop', 'now',
                         'sum_cost']
        self.press_count()
        self.sinout.emit(self.all_record)

    def press_count(self):
        for tp in tqdm(self.range1):
            for _ in range(self.times):
                df_ = pd.DataFrame([[0]*len(self.col_list)], columns=self.col_list)
                dic = {'t_now_list': [0]}

                for idx_machine in range(self.num_home + self.num_import):
                    dic[idx_machine] = df_.copy()
                    dic[idx_machine].loc[dic[idx_machine].index[-1], 'life'] = self.press_tmp[idx_machine]['life'].pop()

                for idx in range(self.num_home):
                    dic[idx]['state'] = 1

                for idx_t in range(len(self.t_table)):
                    t_next_start = self.t_table.iloc[idx_t, 0]
                    flag, dt = self.initial_process(dic, tp)
                    while dic['t_now_list'][-1] + dt < t_next_start:
                        if flag == 1:
                            dic = self.low_fault(dic, self.press_tmp)
                        else:
                            dic = self.low_pm(dic, tp, self.press_tmp)

                        flag, dt = self.initial_process(dic, tp)

                    dic = self.low2high(dic, tp, self.press_tmp, t_next_start)
                    t_next_end = self.t_table.iloc[idx_t, 1]
                    dt = self.initial_high(dic)
                    while dic['t_now_list'][-1] + dt < t_next_end:
                        dic = self.high_fault(dic, self.press_tmp)
                        dt = self.initial_high(dic)
                    dic = self.high2low(dic, self.press_tmp, t_next_end)

                for idx_machine in range(len(dic) -1):
                    self.all_record[idx_machine].loc[str(tp), 'n_pm'] += np.nansum(dic[idx_machine]['n_pm'])
                    self.all_record[idx_machine].loc[str(tp), 'n_cm'] += np.nansum(dic[idx_machine]['n_cm'])
                    self.all_record[idx_machine].loc[str(tp), 'c_pm'] += np.nansum(dic[idx_machine]['c_pm'])
                    self.all_record[idx_machine].loc[str(tp), 'c_cm'] += np.nansum(dic[idx_machine]['n_cm'])
                    self.all_record[idx_machine].loc[str(tp), 'c_stop'] += np.nansum(dic[idx_machine]['c_stop'])
                    self.all_record[idx_machine].loc[str(tp), 'sum_cost'] += np.nansum(dic[idx_machine]['sum_cost'])


    def initial_process(self, dic, tp):
        """
        低负荷期判断下一次维护是什么时候
        params:
            dic:记录所有的运行信息
            tp： 维修周期
        return
            flag， dt
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_p = [tp - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dt = min(t_p)
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        if dt < dtf:
            flag = 2
        else:
            flag = 1
            dt = dtf
        return flag, dt

    def initial_high(self, dic):
        """
        高负荷期内发生故障的维护程序
        params：
            dic:记录所有的运行信息
            tp： 维修周期
        return
            dt 下一次故障间隔
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dt = min(t_f)
        return dt

    def low_fault(self, dic, tmp_count):
        """
        低负荷期发生故障，拟考虑运行时间较短的备件来接替运行
        params:
            dic:记录信息的字典,改变
            tmp_count:设备固有数据，不会变
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
        idx_fault = []
        for idx_ in range(len(idx_using)):
            if t_f[idx_] == dtf:
                idx_fault.append(idx_using[idx_])
        # 不管怎么样，以下这些操作，两种方案应该都一样的
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0
        for idx_f in idx_fault:
            x = dic[idx_f].index[-1]  # 这个时候x已经自加了,df自动更新的
            dic[idx_f].loc[x, 't1'] = 0
            dic[idx_f].loc[x, 'life'] = tmp_count[idx_f]['life'].pop()
            dic[idx_f].loc[x - 1, 'n_cm'] = 1
            dic[idx_f].loc[x - 1, 'c_cm'] = cost_cm(idx_f, tmp_count)
            dic[idx_f].loc[x - 1, 'sum_cost'] = dic[idx_f].loc[x - 1, 'c_cm']
        return dic

    def low_pm(self, dic, tp, tmp_count):
        """
        低负荷期发生预防性维修的函数，拟采用运行时间较少的备件来接替运行
        同时考虑多台设备一起维修的情况
        params:
            dic:记录信息的字典
            tp:维修周期
            tmp_count:设备固有数据，不会变
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_p = [tp - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtp = min(t_p)
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtp)
        idx_pm = []
        for idx_ in range(len(idx_using)):
            if t_p[idx_] == dtp:
                idx_pm.append(idx_using[idx_])
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtp
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0
        for idx_ in idx_pm:
            x = dic[idx_].index[-1]  # 这个时候x已经自加了
            dic[idx_].loc[x, 't1'] = 0
            dic[idx_].loc[x, 'life'] = tmp_count[idx_]['life'].pop()
            dic[idx_].loc[x - 1, 'n_pm'] = 1
            dic[idx_].loc[x - 1, 'c_pm'] = cost_pm(idx_, tmp_count)
            dic[idx_].loc[x - 1, 'sum_cost'] = dic[idx_].loc[x - 1, 'c_pm']
        return dic

    def low2high(self, dic, tp, tmp_count, t_next_start):
        """
        设备从低负荷期到高负荷期的转换过程，所有设备都处于工作状态
        params:
            dic:记录信息的字典,改变
            tmp_count:设备固有数据，不会变
            t_next_start: 高负荷期开始的时间
            tp:维修周期
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        dt = t_next_start - dic['t_now_list'][-1]
        dic['t_now_list'].append(t_next_start)
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 3  # 先进入停机状态，然后再确定哪些机器需要继续运行
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0

        idx_using = list(range(tmp_count['num_home']))  # 表示国产压缩机都得运行
        range_tmp = range(tmp_count['num_home'], tmp_count['num_home'] + tmp_count['num_import'])  # 进口压缩机的idx
        idx_list = list(range_tmp)
        using_time = [dic[i].loc[dic[i].index[-1], 't1'] for i in range_tmp]
        using_time = list(enumerate(using_time))  # 形成元组，第一个参数是idx，第二个参数是元素
        sort_using = sorted(using_time, key=lambda x: x[1])  # sorted默认reverse=false，升序，从小到大
        for idx_ in range(tmp_count['num_high']):
            # num_high是高负荷期进口压缩机需要运行的设备数目
            idx_using.append(idx_list[sort_using[idx_][0]])
            # idx_using表示在高负荷期需要运行的设备的编号

        for idx_u in idx_using:
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
            if dic[idx_u].loc[dic[idx_u].index[-1], 't1'] + tmp_count['h_time'] >= tp:
                x = dic[idx_u].index[-1]
                dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1']
                dic[idx_u].loc[x + 1, 't1'] = 0  # 因为进行了pm，所以下一阶段的t1一定是0
                dic[idx_u].loc[x + 1, 'life'] = tmp_count[idx_u]['life'].pop()
                dic[idx_u].loc[x + 1, 'state'] = 1
                dic[idx_u].loc[x, 'n_pm'] = 1
                dic[idx_u].loc[x, 'n_cm'] = 0
                dic[idx_u].loc[x, 'c_pm'] = cost_pm(idx_u, tmp_count)
                dic[idx_u].loc[x, 'c_cm'] = 0
                dic[idx_u].loc[x, 'c_stop'] = 0
                dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
                dic[idx_u].loc[x, 'sum_cost'] = dic[idx_u].loc[x, 'c_pm']
        return dic

    def find_min_idx(self, dic, tmp_count, idx_using):
        """
        找到进口压缩机中未运行的设备中，运行时间最低的idx
        只有高负荷期，进口压缩机发生故障，才会调用这个函数
        """
        range_tmp = range(tmp_count['num_home'], tmp_count['num_home'] + tmp_count['num_import'])
        idx_no_use = [i for i in range_tmp if i not in idx_using]
        t_f = [dic[i].loc[dic[i].index[-1], 't1'] for i in idx_no_use]
        idx = idx_no_use[t_f.index(min(t_f))]
        return idx

    def high_fault(self, dic, tmp_count):
        """
        设备在高负荷期运行的模式，国产设备发生故障，产生产能损失，修复完成后，立即投入使用
        进口设备发生故障，由备件接替运行
        params：
            dic:记录信息的字典
            tmp_count:设备固有数据，不会变
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        idx_fault = []
        for idx_ in range(len(idx_using)):
            if t_f[idx_] == dtf:
                idx_fault.append(idx_using[idx_])
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
        # 不管怎么样，一下这些操作，两种方案应该都一样的
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1  # 先进入停机状态，然后再令0号机开始运行
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0

        for idx_f in idx_fault:
            x = dic[idx_f].index[-1]  # 这个时候x已经自加了
            dic[idx_f].loc[x, 't1'] = 0
            dic[idx_f].loc[x, 'life'] = tmp_count[idx_f]['life'].pop()
            dic[idx_f].loc[x, 'state'] = 1
            dic[idx_f].loc[x - 1, 'n_cm'] = 1
            dic[idx_f].loc[x - 1, 'c_cm'] = cost_cm(idx_f, tmp_count)
            dic[idx_f].loc[x - 1, 'c_stop'] = 0
            if idx_f in list(range(tmp_count['num_home'])):
                dic[idx_f].loc[x - 1, 'c_stop'] = cost_stop(idx_f, tmp_count)
            else:
                dic[idx_f].loc[x, 'state'] = 3
                idx = self.find_min_idx(dic, tmp_count, idx_using)
                x_ = dic[idx].index[-1]
                dic[idx].loc[x_, 'state'] = 1
            dic[idx_f].loc[x - 1, 'sum_cost'] = dic[idx_f].loc[x - 1, 'c_stop'] + dic[idx_f].loc[x - 1, 'c_cm']

        return dic

    def high2low(self, dic, tmp_count, t_next_end):
        """
        设备从高负荷期向低负荷期切换的策略，仅考虑国产压缩机继续运行的情况
        params
            dic:记录设备运行信息
            tmp_count:设备不变的信息
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        dt = t_next_end - dic['t_now_list'][-1]
        dic['t_now_list'].append(t_next_end)
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 3
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0

        for idx_ in range(tmp_count['num_home']):
            dic[idx_].loc[dic[idx_].index[-1], 'state'] = 1
        return dic

class MainCode(QMainWindow, LNG_demo_1229.Ui_mainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        LNG_demo_1229.Ui_mainWindow.__init__(self)
        self.setupUi(self) #导入界面
        self.interface_plus() #将自己设计好的图像添加进去
        self.pushButton_1.clicked.connect(self.pump_work)
        self.pushButton_2.clicked.connect(self.press_work)



    def pump_work(self):
        self.pump_data()
        self.pushButton_1.setEnabled(False)
        self.work = pump_thread()
        self.work.start()
        self.work.sinout.connect(self.pump_plot)

    def pump_data(self):
        """
        读入所有的高压泵相关数据
        :return:
        """
        global pump_df
        pump_df = {}
        pump_df['life1_value'] = self.doubleSpinBox_1.value()/24
        pump_df['life2_value'] = self.doubleSpinBox_2.value()/24
        pump_df['num_home'] = self.spinBox_1.value()
        pump_df['num_import'] = self.spinBox_2.value()
        pump_df['num_low'] = self.spinBox_3.value()
        pump_df['home_pm'] = self.doubleSpinBox_3.value()
        pump_df['home_cm'] = self.doubleSpinBox_4.value()
        pump_df['home_cs'] = self.doubleSpinBox_5.value()
        pump_df['import_pm'] = self.doubleSpinBox_6.value()
        pump_df['import_cm'] = self.doubleSpinBox_7.value()
        pump_df['import_cs'] = self.doubleSpinBox_8.value()
        pump_df['h_time'] = self.spinBox_4.value()
        pump_df['times'] = self.spinBox_9.value()
        pump_df['low'] = self.spinBox_10.value()
        pump_df['high'] = self.spinBox_11.value()
        pump_df['step'] = self.spinBox_12.value()

    def pump_plot(self, all_record):
        """
        通过信号，将all_record传入
        并绘制相应的图形
        """
        global pump_df
        self.verticalLayout.removeWidget(self.plt)
        pg.setConfigOption("background", "w")
        self.plt = pg.PlotWidget()
        self.plt.addLegend(size=(150, 50))
        self.plt.showGrid(x=True, y=True, alpha=0.5)
        x_num = [int(x) for x in all_record[0].index]
        y_num = 0
        for i in range(len(all_record)):
            y_num += all_record[i]['sum_cost'] / pump_df['times']

        self.plt.plot(x=x_num, y=y_num, name='高压泵预防维修成本曲线', pen='r')
        min_fare = min(y_num)
        y_num = y_num.tolist()
        min_d = x_num[y_num.index(min_fare)]
        self.lineEdit_1.setText(str(min_d) + '天')
        self.lineEdit_2.setText(str(min_fare) + '万元')
        self.verticalLayout.addWidget(self.plt)
        self.pushButton_1.setEnabled(True)

    def press_work(self):
        self.press_data()
        self.pushButton_2.setEnabled(False)
        self.work2 = press_thread()
        self.work2.start()
        self.work2.sinout.connect(self.press_plot)

    def press_data(self):
        """
        读取所有的压缩机相关的数据
        """
        global press_df
        press_df = {}
        press_df['life1_value'] = self.doubleSpinBox_15.value() / 24
        press_df['life2_value'] = self.doubleSpinBox_16.value() / 24
        press_df['num_home'] = self.spinBox_7.value()
        press_df['num_import'] = self.spinBox_8.value()
        press_df['num_high'] = self.spinBox_17.value()
        press_df['num_low'] = self.spinBox_5.value()
        press_df['home_pm'] = self.doubleSpinBox_9.value()
        press_df['home_cm'] = self.doubleSpinBox_10.value()
        press_df['home_cs'] = self.doubleSpinBox_11.value()
        press_df['import_pm'] = self.doubleSpinBox_12.value()
        press_df['import_cm'] = self.doubleSpinBox_13.value()
        press_df['import_cs'] = self.doubleSpinBox_14.value()
        press_df['h_time'] = self.spinBox_6.value()
        press_df['times'] = self.spinBox_13.value()
        press_df['low'] = self.spinBox_14.value()
        press_df['high'] = self.spinBox_15.value()
        press_df['step'] = self.spinBox_16.value()

    def press_plot(self, all_record):
        """
        通过信号，将all_record传入
        并绘制相应的图形
        """
        global press_df
        self.verticalLayout_2.removeWidget(self.plt2)
        pg.setConfigOption("background", "w")
        self.plt2 = pg.PlotWidget()
        self.plt2.addLegend(size=(150, 50))
        self.plt2.showGrid(x=True, y=True, alpha=0.5)
        x_num = [int(x) for x in all_record[0].index]
        y_num = 0
        for i in range(len(all_record)):
            y_num += all_record[i]['sum_cost'] / press_df['times']

        self.plt2.plot(x=x_num, y=y_num, name='压缩机预防维修成本曲线', pen='g')
        min_fare = min(y_num)
        y_num = y_num.tolist()
        min_d = x_num[y_num.index(min_fare)]
        self.lineEdit_3.setText(str(min_d) + '天')
        self.lineEdit_4.setText(str(min_fare) + '万元')
        self.verticalLayout_2.addWidget(self.plt2)
        self.pushButton_2.setEnabled(True)

    def interface_plus(self):
        """
        用绘图先占据空白的部分，使图像不至于乱码
        :return:
        """
        pg.setConfigOption("background", "w")
        self.plt = pg.PlotWidget()
        self.plt.addLegend(size=(150, 50))
        self.plt.showGrid(x=True, y=True, alpha=0.5)
        self.verticalLayout.addWidget(self.plt)

        self.plt2 = pg.PlotWidget()
        self.plt2.addLegend(size=(150, 50))
        self.plt2.showGrid(x=True, y=True, alpha=0.5)
        self.verticalLayout_2.addWidget(self.plt2)


if __name__ == '__main__':
    cgitb.enable(format="text")
    app = QApplication(sys.argv)
    md = MainCode()
    md.show()
    sys.exit(app.exec_())