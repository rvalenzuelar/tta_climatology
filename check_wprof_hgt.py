import parse_data

for y in [1998] + range(2001, 2013):
    wprof = parse_data.windprof(y)
    wprof.check_hgt(y)
