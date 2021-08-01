'''
Experiment Configs
'''
DEBUG = True
model_type = "orig"

# exp_name = "Nerf_1_phase1_orig"
# relative_pose_optimisation = False

# exp_name = "Nerf_1_phase2_orig"
# relative_pose_optimisation = True

exp_name = "Nerf_1_phase1_lamda=0"
relative_pose_optimisation = False

saved_params = None
#saved_params = "inc_nerf_dumps/Exp1/dumps_4"

'''
Every new nerf that you are training will be denoted by nomenclature `g`
Every nerf that you are optimising it against will be called nerf `f`
'''
# nerf_f parameters - previous nerf
start_f = 0
end_f = 20
period_f = 1
saved_f = 'inc_nerf_dumps/Exp1/dumps_19'

# nerf_g parameters - current nerf
nerf_index = 1
start_g = 15
period_g = 1
end_g = 35

# common frame
commom_frame_f_g = [15,16,17,18,19] # global index




