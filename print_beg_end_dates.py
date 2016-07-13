# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:09:25 2016

@author: raul
"""
from tta_analysis import tta_analysis

#years=[1998]
years=[1998]+range(2001,2013)

for y in years:

	tta=tta_analysis(y)
	tta.start_df(wdir_surf=125,wdir_wprof=170,rain_bby=0.25,nhours=2)
	beg=tta.include_dates[0].strftime('%H%M UTC %d %b %Y')
	end=tta.include_dates[-1].strftime('%H%M UTC %d %b %Y')
	year = tta.include_dates[-1].year
	txt='{} {} {}'
	print(txt.format(year,beg,end))
	

