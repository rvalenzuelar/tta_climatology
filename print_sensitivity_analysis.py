# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:32:27 2016

@author: raul
"""

import numpy as np
from tta_analysis import tta_analysis

years = [1998]+range(2001,2013)

params = [
          # 33% wsrf and %50 wprof
          dict(wdir_surf=130,wdir_wprof=170,
               rain_bby=None,rain_czd=0.25,nhours=1),
          dict(wdir_surf=130,wdir_wprof=170,
               rain_bby=None,rain_czd=0.25,nhours=2),
          # 50% wsrf     
          dict(wdir_surf=154,wdir_wprof=170,
               rain_bby=None,rain_czd=0.25,nhours=2),
          # 33% wprof
          dict(wdir_surf=130,wdir_wprof=146,
               rain_bby=None,rain_czd=0.25,nhours=2),               
          dict(wdir_surf=130,wdir_wprof=170,
               rain_bby=None,rain_czd=0.25,nhours=4),
          dict(wdir_surf=130,wdir_wprof=170,
               rain_bby=None,rain_czd=0.25,nhours=8),               
          ]

first = True
for p in params:
	for y in years:
		tta=tta_analysis(y)
		tta.start_df(**p)
		if y == 1998:
			results = tta.print_stats(return_results=True,skip_print=True)
		else:
			r = tta.print_stats(return_results=True,skip_print=True)
			results = np.vstack((results,r))

	bby_tta = results[:,1].sum()
	czd_tta = results[:,4].sum()
	tta_ratio = czd_tta/bby_tta
	tta_hours = results[:,8].sum()
	
	bby_notta = results[:,2].sum()
	czd_notta = results[:,5].sum()
	notta_ratio = czd_notta/bby_notta
	notta_hours = results[:,9].sum()

	
	if first:
		cols = ['mnrain','mnhours','Wd_Surf','Wd_160m',
				'TTczd','TTbby','ratio','hours',
				'NTczd','NTbby','ratio','hours']
		header = '{:>7} '*len(cols)
		print(header.format(*cols))
		first = False
	
	col =  '{:7.2f} {:7d} {:7d} {:7d} '
	col += '{:7.0f} {:7.0f} {:7.1f} {:7.0f} '
	col += '{:7.0f} {:7.0f} {:7.1f} {:7.0f}'

	print(col.format(p['rain_czd'],p['nhours'],p['wdir_surf'],p['wdir_wprof'],
                     czd_tta,bby_tta,tta_ratio,tta_hours,
                     czd_notta,bby_notta,notta_ratio,notta_hours))











