import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons
import numpy as np
import math
import MRAGUI
import curves_data

FUNCTION_DEPTH=50
CP_DEPTH=5
FRAC_RES=0.1

original_curve_points_array=np.array([])
cur_points=np.array([])
cur_level=0

def main():
    global original_curve_points_array, cur_points, cur_level

    original_curve_points_array=curves_data.first_try
    original_curve_points_array=curves_data.fuzzy_line
    original_curve_points_array=curves_data.hello

    cur_points=np.copy(original_curve_points_array)

    j=_calc_j_from_array_length(len(original_curve_points_array))
    cur_level=j

    MRAGUI.build_plt(original_curve_points_array, get_new_points_to_draw_for_level, update_point_in_pointlist, slider_max_level=j)
    return

#GUI
def get_new_points_to_draw_for_level(level):
    global original_curve_points_array, cur_points, cur_details
    points_to_draw=calc_new_points_and_return_to_draw(level)
    return points_to_draw
#GUI
def update_point_in_pointlist(idx, xy):
    global cur_points
    #since we're working on one pointlist and subarrays are always at the beginning
    cur_points[idx]=xy
    return

def _calc_j_from_array_length(length):
    #len(c_j)=m=2^j+1
    # <=> log(m-1,2)=j
    return int(math.log(length-1,2))
def _calc_length_from_j(j):
    #len(c_j)=m=2^j+1
    return 2**j+1

def calc_new_points_and_return_to_draw(level):
    #return _calc_new_points_and_return_to_draw_discrete(level)
    return _calc_new_points_and_return_to_draw_fractional(level)

def _calc_new_points_and_return_to_draw_fractional(level):
    level_ceil=math.ceil(level)
    level_floor=math.floor(level)
    frac_level=level-level_floor
    #print(frac_level, level_ceil, level_floor)

    global FRAC_RES
    if frac_level*1.1<FRAC_RES:
        #do if it was an integer ;)
        return _calc_new_points_and_return_to_draw_discrete(level_floor)

    #else calc fractional-level curve
    print("fractional-level from {} to {} with µ={}".format(level_floor,level_ceil,frac_level))
    #first down to level.ceil (if come from high j)
    #print("-array length j:{}".format(len(point_array_j_plus_1)))
    point_array_j_plus_1=np.copy(_calc_new_points_and_return_to_draw_discrete(level_ceil))
    #than down to level.floor
    point_array_j=np.copy(_calc_new_points_and_return_to_draw_discrete(level_floor))
    #print("-array length j+1:{}".format(len(point_array_j)))

    point_array=_interpolate_curves(point_array_j,point_array_j_plus_1,frac_level)
    return point_array

def _interpolate_curves(point_array_j,point_array_j_plus_1, frac_level):

    doubled_j=np.repeat(point_array_j,2,axis=0)
    #pop last element
    doubled_j=doubled_j[:-1]
    print("doubled j: {}".format(doubled_j.shape))

    print("#j: {} #j+1: {}".format(point_array_j.shape,point_array_j_plus_1.shape))

    point_array=(1.0-frac_level)*doubled_j+frac_level*point_array_j_plus_1
    print("interpolated shape: {}".format(point_array.shape))

    return point_array

def _calc_new_points_and_return_to_draw_discrete(level):
    global cur_level, cur_points
    #make sure, it's an integer
    level=int(level)
    if cur_level==level:
        #nothing to do
        c_length=_calc_length_from_j(level)
        return cur_points[:c_length]

    #be sure, we have integer levels
    j_target=int(level)
    j_actual=int(cur_level)
    cur_level=level

    if j_target==0:
        j_target=1
        cur_level=1

    if j_target>j_actual:
        print("-synthesis from {} up to j: {}".format(j_actual, j_target))
        while j_target>j_actual:
            j_actual+=1
            print("--synthesis up to j: {}".format(j_actual))
            #make synthesis
            cd_length = _calc_length_from_j(j_actual)
            cd_j_minus_1=cur_points[:cd_length]
            PQ=_calc_synthesis_matrices_linear_b_splines(j_actual)
            c_j = np.dot(PQ, cd_j_minus_1)
            #update global points
            cur_points[:cd_length]=c_j
            #return points to render
            #points_to_return=c_j
        return cur_points[:_calc_length_from_j(cur_level)]

    if j_target<j_actual:
        print("-analysis from {} down to j: {}".format(j_actual, j_target))
        while j_target<j_actual:
            #make analysis
            print("--analysis down to j: {}".format(j_actual-1))
            c_length=_calc_length_from_j(j_actual)
            c_j=cur_points[:c_length]

            # calc: cj-1=cj*Aj
            # calc: dj-1=cj*Bj
            #AB=_calc_analysis_matrices_from_semiorthogonal_pq(j_actual)
            #cd_j_minus_1 = np.dot(AB, c_j)
            # "easier": solve P|Q * \frac{c^(j-1), d^(j-1)}=c^j}
            PQ=_calc_synthesis_matrices_linear_b_splines(j_actual)
            cd_j_minus_1 = np.linalg.solve(PQ, c_j)
            # cd_j_minus_1 now are values without using AB ;)

            #update global points
            cur_points[:c_length]=cd_j_minus_1

            c_minus_1_length = _calc_length_from_j(j_actual-1)
            #return points to render (only c not d)
            #points_to_return=cd_j_minus_1[:c_minus_1_length]
            j_actual-=1
        return cur_points[:_calc_length_from_j(cur_level)]



def _calc_analysis_matrices_from_semiorthogonal_pq(j):
    PQ=_calc_synthesis_matrices_linear_b_splines(j)
    #print("shapes: P: {}, Q: {}".format(P.shape, Q.shape))

    #only for orthogonal
        #A=P.T
        #B=Q.T
    #B-Splines W are semi-orthpogonal
    #Build inverse
    #\frac{A,B}=P|Q^-1
    AB=np.linalg.inv(PQ)

    #split up AB
    #A,B=np.vsplit(AB,[2**(j-1)+1])
    return AB


def _calc_synthesis_matrices_linear_b_splines(j):

    norm=True
    #calc mat-sizes
    cols_of_P=2**(j-1)+1

    #shifted down by 2 (for linear) plus endpoint
    #rows_of_P=2*j+1
    rows_of_P=2**j+1
    #quadratic
    rows_of_Q=rows_of_P
    cols_of_Q=rows_of_P-cols_of_P

    P = np.zeros((rows_of_P, cols_of_P))
    Q = np.zeros((rows_of_Q, cols_of_Q))

    #print("shape P: {}, Q: {}".format(P.shape, Q.shape))

    #P is always the same
    for col in range(0, cols_of_P):
        if col*2-1>0:P[col*2-1, col] = 1
        P[col*2-1+1, col] = 2
        if col*2-1+2<rows_of_P:P[col*2-1+2, col] = 1

    if norm:P=0.5*P

    #Q depends on size
    if j==1:
        Q[0,0]=-1
        Q[1,0]=1
        Q[2,0]=-1

        sqrt_3=math.sqrt(3.0)
        if norm:Q=sqrt_3*Q
    if j==2:
        Q[0,0]=-12
        Q[4,1]=-12

        Q[1,0]=11
        Q[3,1]=11

        Q[2,0]=-6
        Q[2,1]=-6

        Q[3,0]=1
        Q[1,1]=1
        sqrt_3_64=math.sqrt(3.0/64)
        if norm:Q=sqrt_3_64*Q
    if j>=3:
        for col in range(1, cols_of_Q - 1):
            Q[col*2-1, col] = 1
            Q[col*2-1+1, col] = -6
            Q[col*2-1+2, col] = 10
            Q[col*2-1+3, col] = -6
            Q[col*2-1+4, col] = 1
            #Q[col*2-1, col] = 1
            #Q[col*2-1+1, col] = -6
            #Q[col*2-1+2, col] = 10
            #Q[col*2-1+3, col] = -6
            #Q[col*2-1+4, col] = 1

        #fix values
        Q[0,0]=-11.022704
        Q[rows_of_Q-1,cols_of_Q-1]=-11.022704

        Q[1,0]=10.1041145
        Q[rows_of_Q-2,cols_of_Q-1]=10.1041145

        Q[2,0]=-5.511352
        Q[rows_of_Q-3,cols_of_Q-1]=-5.511352

        Q[3,0]=0.918559
        Q[rows_of_Q-4,cols_of_Q-1]=0.918559

        sqrt_2j_72=math.sqrt(2.0**j/72)
        if norm:Q=sqrt_2j_72*Q

    PQ = np.concatenate((P, Q), axis=1)
    return PQ




if __name__ == "__main__":
    # execute only if run as a script
    main()