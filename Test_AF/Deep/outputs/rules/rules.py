def findDecision(obj): #obj[0]: distance
	# {"feature": "distance", "instances": 4932, "metric_value": 0.0, "depth": 1}
	if obj[0]>1.0187:
		return 'Yes'
	elif obj[0]<=1.0187:
		return 'Yes'
	else: return 'Yes'
