# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:26:04 2020

@author: alex1
"""
import math
import numpy as np
import pandas as pd


# # debug
# mp = MpFunctions(data=df, freq=2, style='tpo',  avglen=8, ticksize=24, session_hr=24)
# mplist = mp.get_context()
# #mplist[1]
# meandict = mp.get_mean()
# #meandict['volume_mean']


class MpFunctions():
    def __init__(self, data, freq=30, style='tpo', avglen=8, ticksize=8, session_hr=8):
        self.data = data
        self.freq = freq
        self.style = style
        self.avglen = avglen
        self.ticksize = ticksize
        self.session_hr = session_hr

    def get_ticksize(self):
        # data = df
        numlen = int(len(self.data) / 2)
        # sample size for calculating ticksize = 50% of most recent data
        tztail = self.data.tail(numlen).copy()
        tztail['tz'] = tztail.Close.rolling(self.freq).std()  # std. dev of 30 period rolling
        tztail = tztail.dropna()
        ticksize = np.ceil(tztail['tz'].mean() * 0.25)  # 1/4 th of mean std. dev is our ticksize

        if ticksize < 0.2:
            ticksize = 0.2  # minimum ticksize limit

        return int(ticksize)

    def abc(self):
        caps = [' A', ' B', ' C', ' D', ' E', ' F', ' G', ' H', ' I', ' J', ' K', ' L', ' M',
                ' N', ' O', ' P', ' Q', ' R', ' S', ' T', ' U', ' V', ' W', ' X', ' Y', ' Z']
        abc_lw = [x.lower() for x in caps]
        Aa = caps + abc_lw
        alimit = math.ceil(self.session_hr * (60 / self.freq)) + 3
        if alimit > 52:
            alphabets = Aa * int(
                (np.ceil((alimit - 52) / 52)) + 1)  # if bar frequency is less than 30 minutes then multiply list
        else:
            alphabets = Aa[0:alimit]
        bk = [28, 31, 35, 40, 33, 34, 41, 44, 35, 52, 41, 40, 46, 27, 38]
        ti = []
        for s1 in bk:
            ti.append(Aa[s1 - 1])
        tt = (''.join(ti))

        return alphabets, tt

    def get_rf(self):
        self.data['cup'] = np.where(self.data['Close'] >= self.data['Close'].shift(), 1, -1)
        self.data['hup'] = np.where(self.data['High'] >= self.data['High'].shift(), 1, -1)
        self.data['lup'] = np.where(self.data['Low'] >= self.data['Low'].shift(), 1, -1)

        self.data['rf'] = self.data['cup'] + self.data['hup'] + self.data['lup']
        dataf = self.data.drop(['cup', 'lup', 'hup'], axis=1)
        return dataf

    def get_mean(self):
        """
        dfhist: pandas dataframe 1 min frequency
        avglen: Length for mean values
        freq: timeframe for the candlestick & TPOs

        return: a) daily mean for volume, rotational factor (absolute value), IB volume, IB RF b) session length
        dfhist = df.copy()
        """
        dfhist = self.get_rf()
        # dfhist = get_rf(dfhist.copy())
        dfhistd = dfhist.resample("D").agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'volume': 'sum',
             'rf': 'sum', })
        dfhistd = dfhistd.dropna()
        comp_days = len(dfhistd)

        vm30 = dfhistd['volume'].rolling(self.avglen).mean()
        volume_mean = vm30[len(vm30) - 1]
        rf30 = abs((dfhistd['rf'])).rolling(
            self.avglen).mean()  # it is abs mean to get meaningful value to compare daily values
        rf_mean = rf30[len(rf30) - 1]

        date2 = dfhistd.index[1].date()
        mask = dfhist.index.date < date2
        dfsession = dfhist.loc[mask]
        session_hr = math.ceil(len(dfsession) / 60)

        all_val = dict(volume_mean=volume_mean, rf_mean=rf_mean, session_hr=session_hr)

        return all_val

    def tpo(self, dft_rs):
        # dft_rs = dfc1.copy()
        # if len(dft_rs) > int(60 / freq):
        if len(dft_rs) > int(0):
            dft_rs = dft_rs.drop_duplicates('datetime')
            dft_rs = dft_rs.reset_index(inplace=False, drop=True)
            dft_rs['rol_mx'] = dft_rs['High'].cummax()
            dft_rs['rol_mn'] = dft_rs['Low'].cummin()
            dft_rs['ext_up'] = dft_rs['rol_mn'] > dft_rs['rol_mx'].shift(2)
            dft_rs['ext_dn'] = dft_rs['rol_mx'] < dft_rs['rol_mn'].shift(2)
            alphabets = self.abc()[0]
            # alphabets = abc(session_hr, freq)[0]
            alphabets = alphabets[0:len(dft_rs)]
            hh = dft_rs['High'].max()
            ll = dft_rs['Low'].min()
            day_range = hh - ll
            dft_rs['abc'] = alphabets
            # place represents total number of steps to take to compare the TPO count
            place = int(np.ceil((hh - ll) / self.ticksize))
            # kk = 0
            abl_bg = []
            tpo_countbg = []
            pricel = []
            volcountbg = []
            # datel = []
            for u in range(place):
                abl = []
                tpoc = []
                volcount = []
                p = ll + (u * self.ticksize)
                for lenrs in range(len(dft_rs)):
                    if p >= dft_rs['Low'][lenrs] and p < dft_rs['High'][lenrs]:
                        abl.append(dft_rs['abc'][lenrs])
                        tpoc.append(1)
                        volcount.append((dft_rs['volume'][lenrs]) / self.freq)
                abl_bg.append(''.join(abl))
                tpo_countbg.append(sum(tpoc))
                volcountbg.append(sum(volcount))
                pricel.append(p)

            dftpo = pd.DataFrame({'close': pricel, 'alphabets': abl_bg,
                                  'tpocount': tpo_countbg, 'volsum': volcountbg})
            # drop empty rows
            dftpo['alphabets'].replace('', np.nan, inplace=True)
            dftpo = dftpo.dropna()
            dftpo = dftpo.reset_index(inplace=False, drop=True)
            dftpo = dftpo.sort_index(ascending=False)
            dftpo = dftpo.reset_index(inplace=False, drop=True)

            if self.style == 'tpo':
                column = 'tpocount'
            else:
                column = 'volsum'

            dfmx = dftpo[dftpo[column] == dftpo[column].max()]

            mid = ll + ((hh - ll) / 2)
            dfmax = dfmx.copy()
            dfmax['poc-mid'] = abs(dfmax['close'] - mid)
            pocidx = dfmax['poc-mid'].idxmin()
            poc = dfmax['close'][pocidx]
            poctpo = dftpo[column].max()
            tpo_updf = dftpo[dftpo['close'] > poc]
            tpo_updf = tpo_updf.sort_index(ascending=False)
            tpo_updf = tpo_updf.reset_index(inplace=False, drop=True)

            tpo_dndf = dftpo[dftpo['close'] < poc]
            tpo_dndf = tpo_dndf.reset_index(inplace=False, drop=True)

            valtpo = (dftpo[column].sum()) * 0.70

            abovepoc = tpo_updf[column].to_list()
            belowpoc = tpo_dndf[column].to_list()

            if (len(abovepoc) / 2).is_integer() is False:
                abovepoc = abovepoc + [0]

            if (len(belowpoc) / 2).is_integer() is False:
                belowpoc = belowpoc + [0]

            bel2 = np.array(belowpoc).reshape(-1, 2)
            bel3 = bel2.sum(axis=1)
            bel4 = list(bel3)
            abv2 = np.array(abovepoc).reshape(-1, 2)
            abv3 = abv2.sum(axis=1)
            abv4 = list(abv3)
            # cum = poctpo
            # up_i = 0
            # dn_i = 0
            df_va = pd.DataFrame({'abv': pd.Series(abv4), 'bel': pd.Series(bel4)})
            df_va = df_va.fillna(0)
            df_va['abv_idx'] = np.where(df_va.abv > df_va.bel, 1, 0)
            df_va['bel_idx'] = np.where(df_va.bel > df_va.abv, 1, 0)
            df_va['cum_tpo'] = np.where(df_va.abv > df_va.bel, df_va.abv, 0)
            df_va['cum_tpo'] = np.where(df_va.bel > df_va.abv, df_va.bel, df_va.cum_tpo)

            df_va['cum_tpo'] = np.where(df_va.abv == df_va.bel, df_va.abv + df_va.bel, df_va.cum_tpo)
            df_va['abv_idx'] = np.where(df_va.abv == df_va.bel, 1, df_va.abv_idx)
            df_va['bel_idx'] = np.where(df_va.abv == df_va.bel, 1, df_va.bel_idx)
            df_va['cum_tpo_cumsum'] = df_va.cum_tpo.cumsum()
            # haven't add poc tpo because loop cuts off way before 70% so it gives same effect
            df_va_cut = df_va[df_va.cum_tpo_cumsum + poctpo <= valtpo]
            vah_idx = (df_va_cut.abv_idx.sum()) * 2
            val_idx = (df_va_cut.bel_idx.sum()) * 2

            if vah_idx >= len(tpo_updf) and vah_idx != 0:
                vah_idx = vah_idx - 2

            if val_idx >= len(tpo_dndf) and val_idx != 0:
                val_idx = val_idx - 2

            vah = tpo_updf.close[vah_idx]
            val = tpo_dndf.close[val_idx]

            tpoval = dftpo[self.ticksize:-(self.ticksize)]['tpocount']  # take mid section
            exhandle_index = np.where(tpoval <= 2, tpoval.index, None)  # get index where TPOs are 2
            exhandle_index = list(filter(None, exhandle_index))
            distance = self.ticksize * 3  # distance b/w two ex handles / lvn
            lvn_list = []
            for ex in exhandle_index[0:-1:distance]:
                lvn_list.append(dftpo['close'][ex])

            area_above_poc = dft_rs.High.max() - poc
            area_below_poc = poc - dft_rs.Low.min()
            if area_above_poc == 0:
                area_above_poc = 1
            if area_below_poc == 0:
                area_below_poc = 1
            balance = area_above_poc / area_below_poc

            if balance >= 0:
                bal_target = poc - area_above_poc
            else:
                bal_target = poc + area_below_poc

            mp = {'df': dftpo, 'vah': round(vah, 2), 'poc': round(poc, 2), 'val': round(val, 2), 'lvn': lvn_list,
                  'bal_target': round(bal_target, 2)}

        else:
            print('not enough bars for date {}'.format(dft_rs['datetime'][0]))
            mp = {}

        return mp

    # !!! fetch all MP derived results here with date and do extra context analysis

    def get_context(self):
        df_hi = self.get_rf()
        try:
            # df_hi = dflive30.copy() # testing
            DFcontext = [group[1] for group in df_hi.groupby(df_hi.index.date)]
            dfmp_l = []
            i_poctpo_l = []
            i_tposum = []
            vah_l = []
            poc_l = []
            val_l = []
            bt_l = []
            lvn_l = []
            # excess_l = []
            date_l = []
            volume_l = []
            rf_l = []
            # ibv_l = []
            # ibrf_l = []
            # ibh_l = []
            # ib_l = []
            close_l = []
            hh_l = []
            ll_l = []
            range_l = []

            for c in range(len(DFcontext)):  # c=0 for testing
                dfc1 = DFcontext[c].copy()
                # dfc1.iloc[:, 2:6] = dfc1.iloc[:, 2:6].apply(pd.to_numeric)

                dfc1 = dfc1.reset_index(inplace=False, drop=True)
                mpc = self.tpo(dfc1)
                dftmp = mpc['df']
                dfmp_l.append(dftmp)
                # for day types
                i_poctpo_l.append(dftmp['tpocount'].max())
                i_tposum.append(dftmp['tpocount'].sum())
                # !!! get value areas
                vah_l.append(mpc['vah'])
                poc_l.append(mpc['poc'])
                val_l.append(mpc['val'])

                bt_l.append(mpc['bal_target'])
                lvn_l.append(mpc['lvn'])
                # excess_l.append(mpc['excess'])

                # !!! operatio of non profile stats
                date_l.append(dfc1.datetime[0])
                close_l.append(dfc1.iloc[-1]['Close'])
                hh_l.append(dfc1.High.max())
                ll_l.append(dfc1.Low.min())
                range_l.append(dfc1.High.max() - dfc1.Low.min())

                volume_l.append(dfc1.volume.sum())
                rf_l.append(dfc1.rf.sum())
                # !!! get IB
                dfc1['cumsumvol'] = dfc1.volume.cumsum()
                dfc1['cumsumrf'] = dfc1.rf.cumsum()
                dfc1['cumsumhigh'] = dfc1.High.cummax()
                dfc1['cumsummin'] = dfc1.Low.cummin()

            dist_df = pd.DataFrame({'date': date_l, 'maxtpo': i_poctpo_l, 'tpocount': i_tposum, 'vahlist': vah_l,
                                    'poclist': poc_l, 'vallist': val_l, 'btlist': bt_l, 'lvnlist': lvn_l,
                                    'volumed': volume_l, 'rfd': rf_l, 'highd': hh_l, 'lowd': ll_l, 'ranged': range_l,
                                    'close': close_l})

        except Exception as e:
            print(str(e))
            ranking_df = []
            dfmp_l = []
            dist_df = []

        return (dfmp_l, dist_df)

    def get_dayrank(self):
        # dist_df = df_distribution_concat.copy()
        # LVNs
        dist_df = self.get_context()[1]
        lvnlist = dist_df['lvnlist'].to_list()
        cllist = dist_df['close'].to_list()
        lvn_powerlist = []
        total_lvns = 0
        for c, llist in zip(cllist, lvnlist):
            if len(llist) == 0:
                delta_lvn = 0
                total_lvns = 0
                lvn_powerlist.append(total_lvns)
            else:
                for l in llist:
                    delta_lvn = c - l
                    if delta_lvn >= 0:
                        lvn_i = 1
                    else:
                        lvn_i = -1
                    total_lvns = total_lvns + lvn_i
                lvn_powerlist.append(total_lvns)
            total_lvns = 0

        dist_df['Single_Prints'] = lvn_powerlist

        dist_df['distr'] = dist_df.tpocount / dist_df.maxtpo
        dismean = math.floor(dist_df.distr.mean())
        dissig = math.floor(dist_df.distr.std())

        # Assign day types based on TPO distribution and give numerical value for each day types for calculating total strength at the end

        dist_df['daytype'] = np.where(np.logical_and(dist_df.distr >= dismean,
                                                     dist_df.distr < dismean + dissig), 'Trend Distribution Day', '')

        dist_df['daytype_num'] = np.where(np.logical_and(dist_df.distr >= dismean,
                                                         dist_df.distr < dismean + dissig), 3, 0)

        dist_df['daytype'] = np.where(np.logical_and(dist_df.distr < dismean,
                                                     dist_df.distr >= dismean - dissig), 'Normal Variation Day',
                                      dist_df['daytype'])

        dist_df['daytype_num'] = np.where(np.logical_and(dist_df.distr < dismean,
                                                         dist_df.distr >= dismean - dissig), 2, dist_df['daytype_num'])

        dist_df['daytype'] = np.where(dist_df.distr < dismean - dissig,
                                      'Neutral Day', dist_df['daytype'])

        dist_df['daytype_num'] = np.where(dist_df.distr < dismean - dissig,
                                          1, dist_df['daytype_num'])

        dist_df['daytype'] = np.where(dist_df.distr > dismean + dissig,
                                      'Trend Day', dist_df['daytype'])
        dist_df['daytype_num'] = np.where(dist_df.distr > dismean + dissig,
                                          4, dist_df['daytype_num'])
        dist_df['daytype_num'] = np.where(dist_df.close >= dist_df.poclist, dist_df.daytype_num * 1,
                                          dist_df.daytype_num * -1)  # assign signs as per bias

        daytypes = dist_df['daytype'].to_list()

        # volume comparison with mean
        mean_val = self.get_mean()
        rf_mean = mean_val['rf_mean']
        vol_mean = mean_val['volume_mean']

        dist_df['vold_zscore'] = (dist_df.volumed - vol_mean) / dist_df.volumed.std(ddof=0)
        dist_df['rfd_zscore'] = (abs(dist_df.rfd) - rf_mean) / abs(dist_df.rfd).std(ddof=0)
        a, b = 1, 4
        x, y = dist_df.rfd_zscore.min(), dist_df.rfd_zscore.max()
        dist_df['norm_rf'] = (dist_df.rfd_zscore - x) / (y - x) * (b - a) + a

        p, q = dist_df.vold_zscore.min(), dist_df.vold_zscore.max()
        dist_df['norm_volume'] = (dist_df.vold_zscore - p) / (q - p) * (b - a) + a

        dist_df['volume_Factor'] = np.where(dist_df.close >= dist_df.poclist, dist_df.norm_volume * 1,
                                            dist_df.norm_volume * -1)
        dist_df['Rotation_Factor'] = np.where(dist_df.rfd >= 0, dist_df.norm_rf * 1, dist_df.norm_rf * -1)

        # !!! get ranking based on distribution data frame aka dist_df
        ranking_df = dist_df.copy()
        ranking_df['VAH_vs_yVAH'] = np.where(ranking_df.vahlist >= ranking_df.vahlist.shift(), 1, -1)
        ranking_df['VAL_vs_yVAL'] = np.where(ranking_df.vallist >= ranking_df.vallist.shift(), 1, -1)
        ranking_df['POC_vs_yPOC'] = np.where(ranking_df.poclist >= ranking_df.poclist.shift(), 1, -1)
        ranking_df['H_vs_yH'] = np.where(ranking_df.highd >= ranking_df.highd.shift(), 1, -1)
        ranking_df['L_vs_yL'] = np.where(ranking_df.lowd >= ranking_df.lowd.shift(), 1, -1)
        ranking_df['Close_vs_yCL'] = np.where(ranking_df.close >= ranking_df.close.shift(), 1, -1)
        ranking_df['CL>POC<VAH'] = np.where(
            np.logical_and(ranking_df.close >= ranking_df.poclist, ranking_df.close < ranking_df.vahlist), 1, 0)
        ranking_df['CL<poc>val'] = np.where(
            np.logical_and(ranking_df.close < ranking_df.poclist, ranking_df.close >= ranking_df.vallist), -1,
            0)  # Max is 2
        ranking_df['CL<VAL'] = np.where(ranking_df.close < ranking_df.vallist, -2, 0)
        ranking_df['CL>=VAH'] = np.where(ranking_df.close >= ranking_df.vahlist, 2, 0)

        ranking_df['power1'] = 100 * (
                (ranking_df.VAH_vs_yVAH + ranking_df.VAL_vs_yVAL + ranking_df.POC_vs_yPOC + ranking_df.H_vs_yH +
                 ranking_df.L_vs_yL + ranking_df['Close_vs_yCL'] + ranking_df['CL>POC<VAH'] + ranking_df[
                     'CL<poc>val'] + ranking_df.Single_Prints +
                 ranking_df['CL<VAL'] + ranking_df[
                     'CL>=VAH'] + ranking_df.volume_Factor + ranking_df.Rotation_Factor + ranking_df.daytype_num) / 14)

        c, d = 25, 100
        r, s = abs(ranking_df.power1).min(), abs(ranking_df.power1).max()
        ranking_df['power'] = (abs(ranking_df.power1) - r) / (s - r) * (d - c) + c
        ranking_df = ranking_df.round(2)
        # ranking_df['power'] = abs(ranking_df['power1'])

        breakdown_df = ranking_df[
            ['Single_Prints', 'daytype_num', 'volume_Factor', 'Rotation_Factor', 'VAH_vs_yVAH', 'VAL_vs_yVAL',
             'POC_vs_yPOC', 'H_vs_yH',
             'L_vs_yL', 'Close_vs_yCL', 'CL>POC<VAH', 'CL<poc>val', 'CL<VAL', 'CL>=VAH']].transpose()

        breakdown_df = breakdown_df.round(2)

        return (ranking_df, breakdown_df)
