import matplotlib.pyplot as plt
import json

DBGD_output = open('P-DBGD.out', 'r')
DBGD_output_lines = DBGD_output.readlines()
# Extract macro here
macro = json.loads(DBGD_output_lines[0])
print(macro)

DBGD_output_lines = DBGD_output_lines[1:]

for line in DBGD_output_lines:
	run_details = json.loads(line)['run_details']
	graph_title = run_details['click model']
	NDCG_attack = []
	NDCG_label = []
	Tau = []
	num_clicks= []
	iterations = []


	run_results = json.loads(line)['run_results']

	for iteration in run_results:
		NDCG_attack.append(iteration['NDCG_attack'])
		NDCG_label.append(iteration['NDCG_label'])
		Tau.append(iteration['Kendall\'s Tau'])
		num_clicks.append(iteration['Click Number'])
		iterations.append(iteration['iteration'])

	fig_ndcg_attack, ax_ndcg_attack = plt.subplots()
	ax_ndcg_attack.plot(iterations, NDCG_attack)
	ax_ndcg_attack.set_title(graph_title)
	ax_ndcg_attack.set_xlabel("Iteration")
	ax_ndcg_attack.set_ylabel("NDCG attacker evaluation")

	fig_ndcg_label, ax_ndcg_label = plt.subplots()
	ax_ndcg_label.plot(iterations, NDCG_label)
	ax_ndcg_label.set_title(graph_title)
	ax_ndcg_label.set_xlabel("Iteration")
	ax_ndcg_label.set_ylabel("NDCG label evaluation")

	fig_Tau, ax_Tau = plt.subplots()
	ax_Tau.plot(iterations, Tau)
	ax_Tau.set_title(graph_title)
	ax_ndcg_attack.set_xlabel("Iteration")
	ax_Tau.set_ylabel("Kendall\'s Tau")

	fig_num_clicks, ax_num_clicks = plt.subplots()
	ax_num_clicks.plot(iterations, num_clicks)
	ax_num_clicks.set_title(graph_title)
	ax_num_clicks.set_xlabel("Iteration")
	ax_num_clicks.set_ylabel("Attacker clicks number")

plt.show()
