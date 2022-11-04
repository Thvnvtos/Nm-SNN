import sys
from io import StringIO
import evaluate_nmn_utils



seeds = [0]
batch_size = 2

pre_classes = [0, 1, 2, 6, 8]
Test = [5, 7, 9]
Evo = [3, 4, 10]
D6 = [3,5,4,7,10,9] 

print("\n\n")
for tasks in [Evo, Test, D6]:
	for n in [40, 98]:
		best_acc = 0
		best_lambda = 0
		for ewc in range(0, 1500, 20):
			standard_stdout = sys.stdout
			outBuffer = StringIO()
			sys.stdout = outBuffer
			
			acc = evaluate_nmn_utils.test_nmn(seeds, tasks, n, batch_size, ewc = ewc, use_nm=False, show_perm_details = False, return_acc=True)
			sys.stdout = standard_stdout

			if acc > best_acc:
				best_acc = acc
				best_lambda = ewc

		if tasks == Evo: print("(Evo | ", end='')
		elif tasks == Test: print("Test | ", end='')
		else: print("D6 | ", end='')
		print(f"{n}) => best_lambda = {best_lambda}  |  best_acc = {best_acc}\n")

		
