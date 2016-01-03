from csv import DictReader
for t, row in enumerate(DictReader(open("sample5.csv"))):
	# process id
	ID = row['id']
	del row['id']

	# process clicks
	y = 0.
	if 'click' in row:
		if row['click'] == '1':
			y = 1.
		del row['click']

	# extract date
	date = int(row['hour'][4:6])
	week = (date+2) % 7
	# turn hour really into hour, it was originally YYMMDDHH
	row['hour'] = row['hour'][6:]

	# build x
	x = []
	for key in row:
		value = row[key]

		# one-hot encode everything with hash trick
		index = abs(hash(key + '_' + value)) % 2**24
		x.append(index)
	print x
	break
