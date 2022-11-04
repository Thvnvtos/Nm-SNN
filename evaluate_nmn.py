import evaluate_nmn_utils



seeds = [0,1,2,11]

#n = 98
n = 40
batch_size = 2

pre_classes = [0, 1, 2, 6, 8]
Test = [5, 7, 9]
Evo = [3, 4, 10]
D6 = [3,5,4,7,10,9] 

tasks = Test


print("\n\n###################################  No-NM SNN Results : ###################################\n")
evaluate_nmn_utils.test_nmn(seeds, tasks, n, batch_size, use_nm=False, show_perm_details = False)

print("\n\n###################################  NM SNN Results : ###################################\n")
evaluate_nmn_utils.test_nmn(seeds, tasks, n, batch_size, use_nm=True, show_perm_details = False)

print("\n\n###################################  EWC Results : ###################################\n")
evaluate_nmn_utils.test_nmn(seeds, tasks, n, batch_size, ewc = 360, use_nm=False, show_perm_details = False)


'''
best EWC lambda value for each CL scenario, obtained with ewc_lambda_search.py

(Evo | 40) => best_lambda = 400  |  best_acc = 83.56481481481482

(Evo | 98) => best_lambda = 920  |  best_acc = 72.68518518518518

(Test | 40) => best_lambda = 360  |  best_acc = 85.87962962962963

(Test | 98) => best_lambda = 300  |  best_acc = 82.17592592592594

(D6 | 40) => best_lambda = 200  |  best_acc = 51.05820105820106

(D6 | 98) => best_lambda = 100  |  best_acc = 44.411375661375665
'''