'''
	Raul Valenzuela
	raul.valenzuela@colorado.edu

	Example:

	from tta_analysis import tta_analysis

	tta = tta_analysis(1998)

	tta.start(wdir_surf=125,wdir_wprof=170,
					rain_bby=0.25,nhours=5)
	tta.print_stats()

	--- or ---

	# creates pandas dataframe

	tta.start_df(wdir_surf=125,wdir_wprof=170,
					rain_bby=0.25,nhours=5)

	df = tta.df

'''

import numpy as np
import parse_data
import ctext
from datetime import timedelta

class tta_analysis:

	def __init__(self, year=None):
		self.year = year

	def start(self, wdir_surf=None, wdir_wprof=None, 
			  rain_bby=None,rain_czd=None,nhours=None):

		bby = parse_data.surface('bby', self.year)
		czd = parse_data.surface('czd', self.year)
		wprof = parse_data.windprof(self.year)

		beg_bby, end_bby = bby.check_beg_end()
		beg_czd, end_czd = czd.check_beg_end()
		beg_wpr, end_wpr = wprof.check_beg_end()

		''' the latest of the beg '''
		time_beg = max(beg_bby, beg_czd, beg_wpr)

		''' the earliest of the end '''
		time_end = min(end_bby, end_czd, end_wpr)

		''' rainfall before all obs start '''
		rbby_before = np.nansum(bby.dframe.loc[:time_beg].precip)
		rczd_before = np.nansum(czd.dframe.loc[:time_beg].precip)

		''' rainfall after all obs end '''
		rbby_after = np.nansum(bby.dframe.loc[time_end:].precip)
		rczd_after = np.nansum(czd.dframe.loc[time_end:].precip)

		''' number of windprofiles before (after)
			all obs start (end) '''
		nwprof_before = len(wprof.dframe.loc[:time_beg].wdir)
		nwprof_after = len(wprof.dframe.loc[time_end:].wdir)

		onehr = timedelta(hours=1)
		time = time_beg
		bool_buffer = np.array([False] * nhours)
		tta_bool = np.array([])
		rainfall_czd = np.array([])
		rainfall_bby = np.array([])
		wpr_wd_inc = []
		wpr_ws_inc = []
		count = 0
		count_while = 0
		count_exclude = 0

		while (time <= time_end):
				
			surf_wd = bby.dframe.loc[time].wdir
			wpr_wd0 = wprof.dframe.loc[time].wdir[0]  # first gate
			pbby = bby.dframe.loc[time].precip
			pczd = czd.dframe.loc[time].precip



			''' exclude data when there is nan in 
				surf obs or windprof first gate '''
			if surf_wd is None or np.isnan(surf_wd) or np.isnan(wpr_wd0):
				# tta_bool = np.append(tta_bool, [False])
				count_exclude += 1
				time += onehr
				continue



			''' these are obs included in the analysis, then we
				determine if they are tta or no-tta '''
			rainfall_bby=np.append(rainfall_bby,pbby)
			rainfall_czd=np.append(rainfall_czd,pczd)


			''' check conditions '''
			cond1 = (surf_wd <= wdir_surf)
			cond2 = (wpr_wd0 <= wdir_wprof)
			if rain_bby and rain_czd:
				cond3 = (pbby >= rain_bby)
				cond4 = (pczd >= rain_czd)
				tta_condition = cond1 and cond2 and \
								cond3 and cond4
			elif rain_czd:
				cond3 = (pczd >= rain_czd)
				tta_condition = cond1 and cond2 and cond3
			elif rain_bby:
				cond3 = (pbby >= rain_bby)
				tta_condition = cond1 and cond2 and cond3
			else:
				tta_condition = cond1 and cond2

			''' construct boolean array indicating
				hourly TTA conditions with minumm
				of nhours '''
			if tta_condition and bool_buffer.all():
				tta_bool = np.append(tta_bool, [True])
			elif tta_condition:
				bool_buffer[count] = True
				count += 1
				if bool_buffer.all():
					tta_bool = np.append(tta_bool, bool_buffer)
			else:
				bufsum = bool_buffer.sum()
				if bufsum == 0 or bufsum == nhours:
					tta_bool = np.append(tta_bool, [False])
				else:
					tta_bool = np.append(tta_bool, [False] * (bufsum + 1))
				# reset buffer
				bool_buffer = np.array([False] * nhours)
				count = 0

			count_while += 1
			time += onehr



		tta_bool = np.array(tta_bool).astype(bool)
		tta_hours = tta_bool.sum()
		notta_hours = count_while-tta_hours
		self.tta_hours = tta_hours
		self.notta_hours = notta_hours
		self.time_beg = time_beg
		self.time_end = time_end
		self.count_while = count_while
		self.count_exclude = count_exclude
		self.total_rainfall_bby = np.nansum(rainfall_bby)
		self.total_rainfall_czd = np.nansum(rainfall_czd)
		self.bool = tta_bool
		self.tta_rainfall_czd = np.nansum(rainfall_czd[tta_bool])
		self.tta_rainfall_bby = np.nansum(rainfall_bby[tta_bool])
		self.notta_rainfall_czd = np.nansum(rainfall_czd[~tta_bool])
		self.notta_rainfall_bby = np.nansum(rainfall_bby[~tta_bool])
		self.rainfall_bby_before_analysis = rbby_before
		self.rainfall_bby_after_analysis = rbby_after
		self.rainfall_czd_before_analysis = rczd_before
		self.rainfall_czd_after_analysis = rczd_after
		self.nwprof_before = nwprof_before
		self.nwprof_after = nwprof_after
		self.wprof_hgt = wprof.hgt



		print('TTA analysis finished')


	def start_df(self, wdir_surf=None, wdir_wprof=None, 
			  rain_bby=None,rain_czd=None,nhours=None):

		'''
			this version uses pandas dataframe
		'''

		import pandas as pd

		bby = parse_data.surface('bby', self.year)
		czd = parse_data.surface('czd', self.year)
		wprof = parse_data.windprof(self.year)

		beg_bby, end_bby = bby.check_beg_end()
		beg_czd, end_czd = czd.check_beg_end()
		beg_wpr, end_wpr = wprof.check_beg_end()

		''' the latest of the beg '''
		time_beg = max(beg_bby, beg_czd, beg_wpr)

		''' the earliest of the end '''
		time_end = min(end_bby, end_czd, end_wpr)

		''' rainfall before all obs start '''
		rbby_before = np.nansum(bby.dframe.loc[:time_beg].precip)
		rczd_before = np.nansum(czd.dframe.loc[:time_beg].precip)

		''' rainfall after all obs end '''
		rbby_after = np.nansum(bby.dframe.loc[time_end:].precip)
		rczd_after = np.nansum(czd.dframe.loc[time_end:].precip)

		''' number of windprofiles before (after)
			all obs start (end) '''
		nwprof_before = len(wprof.dframe.loc[:time_beg].wdir)
		nwprof_after = len(wprof.dframe.loc[time_end:].wdir)

		onehr = timedelta(hours=1)
		time = time_beg
		bool_buffer = np.array([False] * nhours)
		tta_bool = np.array([])
		rainfall_czd = np.array([])
		rainfall_bby = np.array([])
		wpr_wd_inc = []
		wpr_ws_inc = []
		count = 0
		count_while = 0
		count_exclude = 0

		rng = pd.date_range(start=time_beg,end=time_end,freq='1H')

		cols = ['wdsrf','wdwpr','rbby','rczd','tta','consecutive']
		df = pd.DataFrame(index=rng,columns=cols)

		while (time <= time_end):

			surf_wd = bby.dframe.loc[time].wdir
			wpr_wd0 = wprof.dframe.loc[time].wdir[0]  # first gate
			pbby = bby.dframe.loc[time].precip
			pczd = czd.dframe.loc[time].precip

			if surf_wd is None:
				surf_wd = np.nan

			df.loc[time].wdsrf = surf_wd
			df.loc[time].wdwpr = wpr_wd0
			df.loc[time].rbby = pbby
			df.loc[time].rczd = pczd


			''' check conditions '''
			cond1 = (surf_wd <= wdir_surf)
			cond2 = (wpr_wd0 <= wdir_wprof)
			if rain_bby and rain_czd:
				cond3 = (pbby >= rain_bby)
				cond4 = (pczd >= rain_czd)
				tta_condition = cond1 and cond2 and \
								cond3 and cond4
			elif rain_czd:
				cond3 = (pczd >= rain_czd)
				tta_condition = cond1 and cond2 and cond3
			elif rain_bby:
				cond3 = (pbby >= rain_bby)
				tta_condition = cond1 and cond2 and cond3
			else:
				tta_condition = cond1 and cond2


			df.loc[time].tta = tta_condition

			''' construct boolean array indicating
				hourly TTA conditions with minumm
				of nhours '''
			if tta_condition and bool_buffer.all():
				tta_bool = np.append(tta_bool, [True])
			elif tta_condition:
				bool_buffer[count] = True
				count += 1
				if bool_buffer.all():
					tta_bool = np.append(tta_bool, bool_buffer)
			else:
				bufsum = bool_buffer.sum()
				if bufsum == 0 or bufsum == nhours:
					tta_bool = np.append(tta_bool, [False])
				else:
					tta_bool = np.append(tta_bool, [False] * (bufsum + 1))
				# reset buffer
				bool_buffer = np.array([False] * nhours)
				count = 0

			time += onehr

		df.consecutive = tta_bool.astype(bool)

		wdsrfIsNan = np.isnan(df.wdsrf.values.astype(float))
		wdwprIsNan = np.isnan(df.wdwpr.values.astype(float))
		exclude = wdsrfIsNan + wdwprIsNan

		total_rainfall_bby = df.rbby.sum() - df[exclude].rbby.sum()
		total_rainfall_czd = df.rczd.sum() - df[exclude].rczd.sum()

		hrs_exclude = exclude.sum()
		hrs_include = df.index.size - hrs_exclude

		tta_hours = df.consecutive.sum()
		notta_hours = hrs_include - tta_hours

		tta_rainfall_bby = df[df.consecutive].rbby.sum()
		tta_rainfall_czd = df[df.consecutive].rczd.sum()
		notta_rainfall_bby = total_rainfall_bby - tta_rainfall_bby
		notta_rainfall_czd = total_rainfall_czd - tta_rainfall_czd

		self.time_beg = time_beg
		self.time_end = time_end
		self.count_hrs_include = hrs_include
		self.count_hrs_exclude = hrs_exclude
		self.total_rainfall_bby = total_rainfall_bby
		self.total_rainfall_czd = total_rainfall_czd
		self.tta_rainfall_bby = tta_rainfall_bby
		self.tta_rainfall_czd = tta_rainfall_czd
		self.notta_rainfall_bby = notta_rainfall_bby
		self.notta_rainfall_czd = notta_rainfall_czd
		self.tta_hours = tta_hours
		self.notta_hours = notta_hours
		self.wprof_hgt = wprof.hgt
		self.df = df

		# print('TTA analysis finished')


	def print_stats(self,header=False):

		if header:
			hdr='YR TOT TBB NBB TOT TCZ NCZ TTA NTT TTA NTT BBY(%) CZD(%)'.split()

			fmt='{:>5} {:>5} {:>5} {:>5} ' + \
				'{:>5} {:>5} {:>5} ' + \
				'{:>5} {:>5} ' + \
				'{:>5} {:>5} ' + \
				'{:>5} {:>5}'

			print(fmt.format(*hdr))


		bby_total = self.total_rainfall_bby
		bby_tta =  self.tta_rainfall_bby
		bby_notta =  self.notta_rainfall_bby

		czd_total = self.total_rainfall_czd
		czd_tta =  self.tta_rainfall_czd
		czd_notta =  self.notta_rainfall_czd        

		tta_ratio = czd_tta/float(bby_tta)
		notta_ratio = czd_notta/float(bby_notta)

		tta_hours = self.tta_hours
		notta_hours = self.notta_hours

		rain_perc_bby = 100*(bby_tta/float(bby_total))
		rain_perc_czd = 100*(czd_tta/float(czd_total))

		bby_col = '{:5d} {:5.0f} {:5.0f} {:5.0f} '
		czd_col = '{:5.0f} {:5.0f} {:5.0f} '
		rto_col = '{:5.1f} {:5.1f} '
		hrs_col = '{:5.0f} {:5.0f} '
		prc_col = '{:5.0f} {:5.0f}'

		col1 = bby_col.format(self.year, bby_total, bby_tta,bby_notta)
		col2 = czd_col.format(czd_total, czd_tta, czd_notta)
		col3 = rto_col.format(tta_ratio, notta_ratio)
		col4 = hrs_col.format(tta_hours, notta_hours)
		col5 = prc_col.format(rain_perc_bby, rain_perc_czd)

		print(col1+col2+col3+col4+col5)
		