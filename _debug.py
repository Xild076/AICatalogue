from engine import value
from colorama import Fore
from engine import value


print(Fore.RED + "NESTED DEBUG")

v_nested_1 = value(2)
print(v_nested_1)
v_nested_2 = value([2, 3])
print(v_nested_2)
v_nested_3 = value([[2, 3, 4], [1, 2, 3], [0, 1, 2]])
print(v_nested_3)
v_nested_4 = value([[[1, 2], [2, 3], [3, 4]], [[3, 2], [2, 1], [1, 0]]])
print(v_nested_4)

print(Fore.YELLOW + "ZERO_GRAD DEBUG")
v_zg_1 = value(2)
v_zg_1.grad = 3
v_zg_2 = value([2, 3])
v_zg_2.grad = [3, 4]
print('Pre-zg', v_zg_1)
print('Pre-zg', v_zg_2)
v_zg_1.zero_grad()
v_zg_2.zero_grad()
print('Post-zg', v_zg_1)
print('Post-zg', v_zg_2)

print(Fore.GREEN + "SHAPE DEBUG")
v_shape_1 = value(2)
print(v_shape_1, v_shape_1.shape)
v_shape_2 = value([2, 3])
print(v_shape_2, v_shape_2.shape)
v_shape_3 = value([[2, 3, 4], [1, 2, 3], [0, 1, 2]])
print(v_shape_3, v_shape_3.shape)
v_shape_4 = value([[[1, 2], [2, 3], [3, 4]], [[3, 2], [2, 1], [1, 0]]])
print(v_shape_4, v_shape_4.shape)

"""print(Fore.CYAN + "CHECK_COMPATIBILITY DEBUG")
v_cc_ = value(0)
v_cc_1 = value(2).data
v_cc_2 = value(3).data
v_cc_3 = value([2, 3, 4]).data
v_cc_4 = value([3, 4, 5]).data
v_cc_5 = value([2, 3]).data
v_cc_6 = value([3, 4]).data
v_cc_7 = value([[2, 3], [3, 4], [4, 5]]).data
v_cc_8 = value([[1, 2], [2, 3], [3, 4]]).data
v_cc_9 = value([[2, 3, 4], [3, 4, 5]]).data
v_cc_10 = value([[1, 2, 3], [2, 3, 4]]).data
print('Dim () & Dim () = ', v_cc_._check_compatibility(v_cc_1, v_cc_2))
print('Dim () & Dim (3,) = ', v_cc_._check_compatibility(v_cc_1, v_cc_3))
print('Dim (3,) & Dim () = ', v_cc_._check_compatibility(v_cc_3, v_cc_1))
print('Dim () & Dim (3, 2) = ', v_cc_._check_compatibility(v_cc_1, v_cc_7))
print('Dim (3, 2) & Dim () = ', v_cc_._check_compatibility(v_cc_7, v_cc_1))
print('Dim () & Dim (2, 3) = ', v_cc_._check_compatibility(v_cc_1, v_cc_9))
print('Dim (2, 3) & Dim () = ', v_cc_._check_compatibility(v_cc_9, v_cc_1))
print('Dim (3,) & Dim (3,) = ', v_cc_._check_compatibility(v_cc_3, v_cc_4))
print('Dim (3,) & Dim (2, 3) = ', v_cc_._check_compatibility(v_cc_3, v_cc_9))
print('Dim (2, 3) & Dim (3,) = ', v_cc_._check_compatibility(v_cc_9, v_cc_3))
print('Dim (3,) & Dim (3, 2) = ', v_cc_._check_compatibility(v_cc_3, v_cc_7))
print('Dim (3, 2) & Dim (3,) = ', v_cc_._check_compatibility(v_cc_7, v_cc_3))
print('Dim (2,) & Dim (2, 3) = ', v_cc_._check_compatibility(v_cc_5, v_cc_9))
print('Dim (2, 3) & Dim (2,) = ', v_cc_._check_compatibility(v_cc_9, v_cc_5))
print('Dim (2,) & Dim (3, 2) = ', v_cc_._check_compatibility(v_cc_5, v_cc_7))
print('Dim (3, 2) & Dim (2,) = ', v_cc_._check_compatibility(v_cc_7, v_cc_5))
print('Dim (3, 2) & Dim (3, 2) = ', v_cc_._check_compatibility(v_cc_7, v_cc_8))
print('Dim (2, 3) & Dim (2, 3) = ', v_cc_._check_compatibility(v_cc_9, v_cc_10))"""

print(Fore.BLUE + "ADD DEBUG")
v_add_1 = value(2)
v_add_2 = value(3)
v_add_3 = value([2, 3, 4])
v_add_4 = value([3, 4, 5])
v_add_5 = value([2, 3])
v_add_6 = value([3, 4])
v_add_7 = value([[2, 3], [3, 4], [4, 5]])
v_add_8 = value([[1, 2], [2, 3], [3, 4]])
v_add_9 = value([[2, 3, 4], [3, 4, 5]])
v_add_10 = value([[1, 2, 3], [2, 3, 4]])
val_add = [v_add_1, v_add_2, v_add_3, v_add_4, v_add_5, v_add_6, v_add_7, v_add_8, v_add_9, v_add_10]
def reset():
    for v in val_add:
        v.zero_grad()
v_add_1_2 = v_add_1 + v_add_2
v_add_1_3 = v_add_1 + v_add_3
v_add_3_1 = v_add_3 + v_add_1
v_add_1_7 = v_add_1 + v_add_7
v_add_7_1 = v_add_7 + v_add_1
v_add_1_9 = v_add_1 + v_add_9
v_add_9_1 = v_add_9 + v_add_1
#v_add_3_7 = v_add_3 + v_add_7
#v_add_7_3 = v_add_7 + v_add_3
v_add_3_9 = v_add_3 + v_add_9
v_add_9_3 = v_add_9 + v_add_3
v_add_5_7 = v_add_5 + v_add_7
v_add_7_5 = v_add_7 + v_add_5
#v_add_5_9 = v_add_5 + v_add_9
#v_add_9_5 = v_add_9 + v_add_5
print('Dim () + Dim () = ', v_add_1_2)
print('Dim () + Dim (3,) = ', v_add_1_3)
print('Dim (3,) + Dim () = ', v_add_3_1)
print('Dim () + Dim (3, 2) = ', v_add_1_7)
print('Dim (3, 2) + Dim () = ', v_add_7_1)
print('Dim () + Dim (2, 3) = ', v_add_1_9)
print('Dim (2, 3) + Dim () = ', v_add_9_1)
#print('Dim (3,) + Dim (3, 2) = ', v_add_3_7)
#print('Dim (3, 2) + Dim (3,) = ', v_add_7_3)
print('Dim (3,) + Dim (2, 3) = ', v_add_3_9)
print('Dim (2, 3) + Dim (3,) = ', v_add_9_3)
print('Dim (2,) + Dim (3, 2) = ', v_add_5_7)
print('Dim (3, 2) + Dim (2,) = ', v_add_7_5)
#print('Dim (2,) + Dim (2, 3) = ', v_add_5_9)
#print('Dim (2, 3) + Dim (2,) = ', v_add_9_5)
v_add_1_2.backward()
print('Backward of v_add_1_2', v_add_1, v_add_2)
reset()
v_add_1_3.backward()
print('Backward of v_add_1_3', v_add_1, v_add_3)
reset()
v_add_1_7.backward()
print('Backward of v_add_1_7', v_add_1, v_add_7)
reset()
v_add_1_9.backward()
print('Backward of v_add_1_9', v_add_1, v_add_9)
reset()
#v_add_3_7.backward()
print('Backward of v_add_3_7', v_add_3, v_add_7)
reset()
v_add_3_9.backward()
print('Backward of v_add_3_9', v_add_3, v_add_9)
reset()
v_add_5_7.backward()
print('Backward of v_add_5_7', v_add_5, v_add_7)
reset()

print(Fore.MAGENTA + "MUL DEBUG")
v_mul_1 = value(2)
v_mul_2 = value(3)
v_mul_3 = value([2, 3, 4])
v_mul_4 = value([3, 4, 5])
v_mul_5 = value([2, 3])
v_mul_6 = value([3, 4])
v_mul_7 = value([[2, 3], [3, 4], [4, 5]])
v_mul_8 = value([[1, 2], [2, 3], [3, 4]])
v_mul_9 = value([[2, 3, 4], [3, 4, 5]])
v_mul_10 = value([[1, 2, 3], [2, 3, 4]])
val_mul = [v_mul_1, v_mul_2, v_mul_3, v_mul_4, v_mul_5, v_mul_6, v_mul_7, v_mul_8, v_mul_9, v_mul_10]
def reset():
    for v in val_mul:
        v.zero_grad()
v_mul_1_2 = v_mul_1 * v_mul_2
v_mul_1_3 = v_mul_1 * v_mul_3
v_mul_3_1 = v_mul_3 * v_mul_1
v_mul_1_7 = v_mul_1 * v_mul_7
v_mul_7_1 = v_mul_7 * v_mul_1
v_mul_1_9 = v_mul_1 * v_mul_9
v_mul_9_1 = v_mul_9 * v_mul_1
#v_mul_3_7 = v_mul_3 * v_mul_7
#v_mul_7_3 = v_mul_7 * v_mul_3
v_mul_3_9 = v_mul_3 * v_mul_9
v_mul_9_3 = v_mul_9 * v_mul_3
v_mul_5_7 = v_mul_5 * v_mul_7
v_mul_7_5 = v_mul_7 * v_mul_5
#v_mul_5_9 = v_mul_5 * v_mul_9
#v_mul_9_5 = v_mul_9 * v_mul_5
print('Dim () * Dim () = ', v_mul_1_2)
print('Dim () * Dim (3,) = ', v_mul_1_3)
print('Dim (3,) * Dim () = ', v_mul_3_1)
print('Dim () * Dim (3, 2) = ', v_mul_1_7)
print('Dim (3, 2) * Dim () = ', v_mul_7_1)
print('Dim () * Dim (2, 3) = ', v_mul_1_9)
print('Dim (2, 3) * Dim () = ', v_mul_9_1)
#print('Dim (3,) * Dim (3, 2) = ', v_mul_3_7)
#print('Dim (3, 2) * Dim (3,) = ', v_mul_7_3)
print('Dim (3,) * Dim (2, 3) = ', v_mul_3_9)
print('Dim (2, 3) * Dim (3,) = ', v_mul_9_3)
print('Dim (2,) * Dim (3, 2) = ', v_mul_5_7)
print('Dim (3, 2) * Dim (2,) = ', v_mul_7_5)
#print('Dim (2,) * Dim (2, 3) = ', v_mul_5_9)
#print('Dim (2, 3) * Dim (2,) = ', v_mul_9_5)
v_mul_1_2.backward()
print('Backward of v_mul_1_2', v_mul_1, v_mul_2)
reset()
v_mul_1_3.backward()
print('Backward of v_mul_1_3', v_mul_1, v_mul_3)
reset()
v_mul_1_7.backward()
print('Backward of v_mul_1_7', v_mul_1, v_mul_7)
reset
v_mul_1_9.backward()
print('Backward of v_mul_1_9', v_mul_1, v_mul_9)
reset()
#v_mul_3_7.backward()
print('Backward of v_mul_3_7', v_mul_3, v_mul_7)
reset()
v_mul_3_9.backward()
print('Backward of v_mul_3_9', v_mul_3, v_mul_9)
reset()
v_mul_5_7.backward()
print('Backward of v_mul_5_7', v_mul_5, v_mul_7)
reset()

print(Fore.WHITE + "POW DEBUG")
v_pow_1 = value(2)
v_pow_2 = value([2, 3, 4])
v_pow_3 = value([2, 3])
v_pow_4 = value([[2, 3], [3, 4], [4, 5]])
v_pow_5 = value([[2, 3, 4], [3, 4, 5]])
val_pow = [v_pow_1, v_pow_2, v_pow_3, v_pow_4, v_pow_5]
def reset():
    for v in val_pow:
        v.zero_grad()
v_pow_1_2 = v_pow_1 ** 2
v_pow_2_2 = v_pow_2 ** 2
v_pow_3_2 = v_pow_3 ** 2
v_pow_4_2 = v_pow_4 ** 2
v_pow_5_2 = v_pow_5 ** 2
print('Dim () ** 2 = ', v_pow_1_2)
print('Dim (3,) ** 2 = ', v_pow_2_2)
print('Dim (2,) ** 2 = ', v_pow_3_2)
print('Dim (3, 2) ** 2 = ', v_pow_4_2)
print('Dim (2, 3) ** 2 = ', v_pow_5_2)
v_pow_1_2.backward()
print('Backward of v_pow_1_2', v_pow_1)
reset()
v_pow_2_2.backward()
print('Backward of v_pow_2_2', v_pow_2)
reset()
v_pow_3_2.backward()
print('Backward of v_pow_3_2', v_pow_3)
reset()
v_pow_4_2.backward()
print('Backward of v_pow_4_2', v_pow_4)
reset()
v_pow_5_2.backward()
print('Backward of v_pow_5_2', v_pow_5)
reset()

print(Fore.RED + "RELU DEBUG")
v_relu_1 = value(2)
v_relu_2 = value([2, -3, 4])
v_relu_3 = value([2, -3])
v_relu_4 = value([[2, -3], [-3, 4], [4, -5]])
v_relu_5 = value([[2, -3, 4], [-3, 4, -5]])
val_relu = [v_relu_1, v_relu_2, v_relu_3, v_relu_4, v_relu_5]
def reset():
    for v in val_pow:
        v.zero_grad()
v_relu_1_2 = v_relu_1.relu()
v_relu_2_2 = v_relu_2.relu()
v_relu_3_2 = v_relu_3.relu()
v_relu_4_2 = v_relu_4.relu()
v_relu_5_2 = v_relu_5 .relu()
print('Dim () RELU = ', v_relu_1_2)
print('Dim (3,) RELU = ', v_relu_2_2)
print('Dim (2,) RELU = ', v_relu_3_2)
print('Dim (3, 2) RELU = ', v_relu_4_2)
print('Dim (2, 3) RELU = ', v_relu_5_2)
v_relu_1_2.backward()
print('Backward of v_relu_1_2', v_relu_1)
reset()
v_relu_2_2.backward()
print('Backward of v_relu_2_2', v_relu_2)
reset()
v_relu_3_2.backward()
print('Backward of v_relu_3_2', v_relu_3)
reset()
v_relu_4_2.backward()
print('Backward of v_relu_4_2', v_relu_4)
reset()
v_relu_5_2.backward()
print('Backward of v_relu_5_2', v_relu_5)
reset()

print(Fore.YELLOW + "LOG DEBUG")
v_log_1 = value(2)
v_log_2 = value([2, 3, 4])
v_log_3 = value([2, 3])
v_log_4 = value([[2, 3], [3, 4], [4, 5]])
v_log_5 = value([[2, 3, 4], [3, 4, 5]])
val_log = [v_log_1, v_log_2, v_log_3, v_log_4, v_log_5]
def reset():
    for v in val_pow:
        v.zero_grad()
v_log_1_2 = v_log_1.log()
v_log_2_2 = v_log_2.log()
v_log_3_2 = v_log_3.log()
v_log_4_2 = v_log_4.log()
v_log_5_2 = v_log_5.log()
print('Dim () LOG = ', v_log_1_2)
print('Dim (3,) LOG = ', v_log_2_2)
print('Dim (2,) LOG = ', v_log_3_2)
print('Dim (3, 2) LOG = ', v_log_4_2)
print('Dim (2, 3) LOG = ', v_log_5_2)
v_log_1_2.backward()
print('Backward of v_log_1_2', v_log_1)
reset()
v_log_2_2.backward()
print('Backward of v_log_2_2', v_log_2)
reset()
v_log_3_2.backward()
print('Backward of v_log_3_2', v_log_3)
reset()
v_log_4_2.backward()
print('Backward of v_log_4_2', v_log_4)
reset()
v_log_5_2.backward()
print('Backward of v_log_5_2', v_log_5)
reset()

print(Fore.GREEN + "EXP DEBUG")
v_exp_1 = value(2)
v_exp_2 = value([2, 3, 4])
v_exp_3 = value([2, 3])
v_exp_4 = value([[2, 3], [3, 4], [4, 5]])
v_exp_5 = value([[2, 3, 4], [3, 4, 5]])
val_exp = [v_exp_1, v_exp_2, v_exp_3, v_exp_4, v_exp_5]
def reset():
    for v in val_exp:
        v.zero_grad()
v_exp_1_2 = v_exp_1.exp()
v_exp_2_2 = v_exp_2.exp()
v_exp_3_2 = v_exp_3.exp()
v_exp_4_2 = v_exp_4.exp()
v_exp_5_2 = v_exp_5.exp()
print('Dim () EXP = ', v_exp_1_2)
print('Dim (3,) EXP = ', v_exp_2_2)
print('Dim (2,) EXP = ', v_exp_3_2)
print('Dim (3, 2) EXP = ', v_exp_4_2)
print('Dim (2, 3) EXP = ', v_exp_5_2)
v_exp_1_2.backward()
print('Backward of v_exp_1_2', v_exp_1)
reset()
v_exp_2_2.backward()
print('Backward of v_exp_2_2', v_exp_2)
reset()
v_exp_3_2.backward()
print('Backward of v_exp_3_2', v_exp_3)
reset()
v_exp_4_2.backward()
print('Backward of v_exp_4_2', v_exp_4)
reset()
v_exp_5_2.backward()
print('Backward of v_exp_5_2', v_exp_5)
reset()

print(Fore.CYAN + "SUM DEBUG")
v_sum_1, v_sum_2, v_sum_3 = value(2), value(3), value(4)
v_sum_1_2_3 = value([2, 3, 4])
v_sum_4, v_sum_5 = value([4, 2, 1]), value([-1, 4, 2])
v_sum_4_5 = value([[4, 2, 1], [-1, 4, 2]])
val_sum = [v_sum_1, v_sum_2, v_sum_3, v_sum_1_2_3, v_sum_4, v_sum_5, v_sum_4_5]
def reset():
    for v in val_sum:
        v.zero_grad()
v_sum_added_1_2_3 = v_sum_1 + v_sum_2 + v_sum_3
v_sum_complete_1_2_3 = v_sum_1_2_3.sum()
v_sum_added_4_5 = v_sum_4 + v_sum_5
v_sum_complete_4_5 = v_sum_4_5.sum()
print('Dim () x 3 SUM = ', v_sum_added_1_2_3)
print('Dim (3,) SUM = ', v_sum_complete_1_2_3)
print('Dim (3,) x 2 SUM = ', v_sum_added_4_5)
print('Dim (2, 3) SUM = ', v_sum_complete_4_5)
v_sum_added_1_2_3_mul_2 = v_sum_added_1_2_3 * 2
v_sum_complete_1_2_3_mul_2 = v_sum_complete_1_2_3 * 2
v_sum_added_4_5_mul_2 = v_sum_added_4_5 * 2
v_sum_complete_4_5_mul_2 = v_sum_complete_4_5 * 2
print('Dim () x 3 SUM * 2 = ', v_sum_added_1_2_3_mul_2)
print('Dim (3,) SUM * 2 = ', v_sum_complete_1_2_3_mul_2)
print('Dim (3,) x 2 SUM * 2 = ', v_sum_added_4_5_mul_2)
print('Dim (2, 3) SUM * 2 = ', v_sum_complete_4_5_mul_2)
v_sum_added_1_2_3_mul_2.backward()
print('Backward of v_sum_added_1_2_3', v_sum_1, v_sum_2, v_sum_3)
reset()
v_sum_complete_1_2_3_mul_2.backward()
print('Backward of v_sum_complete_1_2_3', v_sum_1_2_3)
reset()
v_sum_added_4_5_mul_2.backward()
print('Backward of v_sum_added_4_5', v_sum_4, v_sum_5)
reset()
v_sum_complete_4_5_mul_2.backward()
print('Backward of v_sum_complete_4_5', v_sum_4_5)
reset()
"""
print(Fore.BLUE + "COMBINE DEBUG")
v_combine_1, v_combine_2, v_combine_3 = value(2), value(3), value(4)
v_combine_4, v_combine_5 = value([4, 2, 1]), value([-1, 4, 2])
val_combine = [v_combine_1, v_combine_2, v_combine_3, v_combine_4, v_combine_5]
def reset():
    for v in val_combine:
        v.zero_grad()
v_combine_1_2_3 = value.combine([v_combine_1, v_combine_2, v_combine_3])
v_combine_4_5 = value.combine([v_combine_4, v_combine_5])
print('Dim () x 3 COMBINE = ', v_combine_1_2_3)
print('Dim (3,) x 2 COMBINE = ', v_combine_4_5)
v_combine_1_2_3_mul_2 = v_combine_1_2_3 * 2
v_combine_4_5_mul_2 = v_combine_4_5 * 2
print('Dim () x 3 COMBINE * 2 = ', v_combine_1_2_3_mul_2)
print('Dim (3,) x 2 COMBINE * 2 = ', v_combine_4_5_mul_2)
v_combine_1_2_3_mul_2.backward()
print('Backward of v_combine_1_2_3_mul_2', v_combine_1, v_combine_2, v_combine_3)
reset()
v_combine_4_5_mul_2.backward()
print('Backward of v_combine_4_5_mul_2', v_combine_4, v_combine_5)
reset()

print(Fore.MAGENTA + "SPLIT DEBUG")
v_split_1 = value([2, 3, 4])
v_split_2 = value([[4, 2, 1], [-1, 4, 2]])
val_split = [v_split_1, v_split_2]
def reset():
    for v in val_split:
        v.zero_grad()
v_dsplit_1 = value.split(v_split_1)
v_dsplit_2 = value.split(v_split_2)
print('Dim (3,) SPLIT = ', v_dsplit_1)
print('Dim (2, 3) SPLIT = ', v_dsplit_2)
v_dsplit_1_mul_2 = [0] * len(v_dsplit_1)
for i in range(len(v_dsplit_1)):
    v_dsplit_1_mul_2[i] = v_dsplit_1[i] * 2
v_dsplit_2_mul_2 = [0] * len(v_dsplit_2)
for i in range(len(v_dsplit_2)):
    v_dsplit_2_mul_2[i] = v_dsplit_2[i] * 2
print('Dim (3,) SPLIT * 2 = ', v_dsplit_1_mul_2)
print('Dim (2, 3) SPLIT * 2 = ', v_dsplit_2_mul_2)
for vv in v_dsplit_1_mul_2:
    vv.backward()
print('Backward of v_dsplit_1_mul_2', v_split_1)
reset()
for vv in v_dsplit_2_mul_2:
    vv.backward()
print('Backward of v_dsplit_2_mul_2', v_split_2)

print(Fore.WHITE + "SOFTMAX DEBUG")
v_softmax_1 = value([2, 3, 4])
v_softmax_2 = value([[4, 2, 1], [-1, 4, 2]])
val_softmax = [v_softmax_1, v_softmax_2]
def reset():
    for v in val_softmax:
        v.zero_grad()
v_softmaxxed_1 = v_softmax_1.softmax()
v_softmaxxed_2 = v_softmax_2.softmax()
print('Dim (3,) SOFTMAX = ', v_softmaxxed_1)
print('Dim (2, 3) SOFTMAX = ', v_softmaxxed_2)
v_softmaxxed_1.backward()
print('Backward of v_softmaxxed_1', v_softmax_1)
reset()
v_softmaxxed_2.backward()
print('Backward of v_softmaxxed_2', v_softmax_2)
reset()

print(Fore.RED + "MAX DEBUG")
v_max_1 = value([2, 3, 4])
v_max_1_1 = v_max_1.max()
print('Dim (3,) MAX', v_max_1_1)

print(Fore.YELLOW + "MIN DEBUG")
v_min_1 = value([2, 3, 4])
v_min_1_1 = v_min_1.min()
print('Dim (3,) MIN', v_min_1_1)

print(Fore.GREEN + "MIN DEBUG")
v_mm_1 = value([[2, -3], [-3, 4], [4, -5]])
v_mm_2 = value([[2, -3, 4], [-3, 4, -5]])
v_mm_1_2 = v_mm_1 @ v_mm_2
v_mm_2_1 = v_mm_2 @ v_mm_1
val_mm = [v_mm_1, v_mm_2]
def reset():
    for v in val_mm:
        v.zero_grad()
print('Dim (3, 2) @ Dim (2, 3) = ', v_mm_1_2)
print('Dim (2, 3) @ Dim (3, 2) = ', v_mm_2_1)
v_mm_1_2.backward()
print('Backward of v_mm_1_2', v_mm_1, v_mm_2)
reset()
v_mm_2_1.backward()
print('Backward of v_mm_2_1', v_mm_1, v_mm_2)
reset()

print(Fore.CYAN + "CATEGORICAL DEBUG")
v_logit_1 = value([2, 3, 4])
v_logit_2 = value([[4, 2, 1], [-1, 4, 2]])
v_prob_1 = value([0.2, 0.3, 0.5])
v_prob_2 = value([[0.7, 0.2, 0.1], [0, 0.1, 0.9]])
v_cat_logit_1 = Categorical(logits=v_logit_1)
v_cat_logit_2 = Categorical(logits=v_logit_2)
v_cat_prob_1 = Categorical(probs=v_prob_1)
v_cat_prob_2 = Categorical(probs=v_prob_2)
print('Logit -> Categorical (3,) = ', v_cat_logit_1)
print('Logit -> Categorical (2, 3) = ', v_cat_logit_2)
print('Prob -> Categorical (3,) = ', v_cat_prob_1)
print('Prob -> Categorical (2, 3) = ', v_cat_prob_2)
v_lp_1 = v_cat_logit_1.log_prob(2)
v_lp_2 = v_cat_logit_2.log_prob([0, 1])
v_lp_3 = v_cat_prob_1.log_prob(2)
v_lp_4 = v_cat_prob_2.log_prob([0, 2])
print('Logit -> Categorical (3,) Log Prob = ', v_lp_1)
print('Logit -> Categorical (2, 3) Log Prob = ', v_lp_2)
print('Prob -> Categorical (3,) Log Prob = ', v_lp_3)
print('Prob -> Categorical (2, 3) Log Prob = ', v_lp_4)
v_lp_1.backward()
v_lp_2.backward()
v_lp_3.backward()
v_lp_4.backward()
print('Backward of v_lp_1', v_logit_1)
print('Backward of v_lp_2', v_logit_2)
print('Backward of v_lp_3', v_prob_1)
print('Backward of v_lp_4', v_prob_2)

print(Fore.BLUE + "MAX DEBUG")
v_max_1 = value([4, -1, 2, 5])
v_maxxed_1 = v_max_1.max()
print('Maxxed of (4,) = ', v_maxxed_1)
v_maxxed_1_mul_2 = v_maxxed_1 * 2
print('Maxxed of (4,) * 2 = ', v_maxxed_1)
v_maxxed_1_mul_2.backward()
print('Backward of v_maxxed_1_mul_2', v_max_1)

print(Fore.MAGENTA + "MIN DEBUG")
v_min_1 = value([4, -1, 2, 5])
v_minned_1 = v_min_1.min()
print('Minned of (4,) = ', v_minned_1)
v_minned_1_mul_2 = v_minned_1 * 2
print('Minned of (4,) * 2 = ', v_minned_1)
v_minned_1_mul_2.backward()
print('Backward of v_minned_1_mul_2', v_min_1)

print(Fore.WHITE + "INDEX DEBUG")
v_index_1 = value([4, -1, 2, 5])
v_indexxed_1 = v_index_1[2]
print('Index 2 of (4,) = ', v_indexxed_1)
v_indexxed_1_mul_2 = v_indexxed_1 * 2
print('Index 2 of (4,) * 2 = ', v_indexxed_1)
v_indexxed_1_mul_2.backward()
print('Backward of v_indexxed_1_mul_2', v_index_1)"""

print(Fore.RESET)