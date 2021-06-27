import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons
import numpy as np
import math
import MRAGUI

FUNCTION_DEPTH=50
CP_DEPTH=5

original_curve_points_array=np.array([])
cur_points=np.array([])
cur_level=0

def main():
    global original_curve_points_array, cur_points, cur_level
    original_curve_points_array = [[0.12043010752688174, 0.7251082251082251],
                                   [0.25806451612903225, 0.8777056277056279],
                                   [0.42258064516129035, 0.5876623376623378],
                                   [0.6709677419354839, 0.5822510822510824],
                                   [0.5591397849462366, 0.8614718614718615],
                                   [0.8849462365591397, 0.8354978354978355],
                                   [0.7935483870967742, 0.7424242424242425],
                                   [0.9419354838709678, 0.48917748917748927],
                                   [0.7516129032258064, 0.06709956709956713],
                                   [0.48602150537634414, 0.22077922077922082],
                                   [0.1870967741935484, 0.04112554112554115]]

    original_curve_points_array = np.array(
        [[0.09354838709677418, 0.7142857142857143],
         [0.16989247311827957, 0.8041125541125542],
         [0.3182795698924731, 0.9253246753246753],
         [0.3344086021505377, 0.7716450216450217],
         [0.36989247311827955, 0.6147186147186148],
         [0.5344086021505376, 0.6017316017316018],
         [0.7107526881720431, 0.5963203463203464],
         [0.6193548387096774, 0.6872294372294373],
         [0.5494623655913978, 0.8452380952380953],
         [0.7645161290322581, 0.8484848484848485],
         [0.8860215053763442, 0.8257575757575758],
         [0.8440860215053764, 0.8041125541125542],
         [0.7967741935483871, 0.7456709956709957],
         [0.8612903225806451, 0.6201298701298702],
         [0.9397849462365592, 0.48484848484848486],
         [0.8591397849462367, 0.27489177489177496],
         [0.7354838709677419, 0.07034632034632038],
         [0.646236559139785, 0.1103896103896104],
         [0.49139784946236564, 0.21428571428571433],
         [0.3193548387096774, 0.1060606060606061],
         [0.1849462365591398, 0.02922077922077926],
         [0.11397849462365592, 0.4188311688311689],
         [0.1268817204301075, 0.7077922077922079],
         ])

    original_curve_points_array=np.array([
        [0.15698924731182798, 0.31926406926406925],
        [0.24516129032258063, 0.30519480519480524],
        [0.17311827956989248, 0.25],
        [0.1849462365591398, 0.18290043290043292],
        [0.12258064516129033, 0.1774891774891775],
        [0.1763440860215054, 0.09632034632034636],
        [0.2172043010752688, 0.13095238095238101],
        [0.33225806451612905, 0.04329004329004332],
        [0.3193548387096774, 0.13419913419913423],
        [0.42795698924731185, 0.1287878787878788],
        [0.42365591397849467, 0.2088744588744589],
        [0.4903225806451613, 0.1677489177489178],
        [0.546236559139785, 0.26623376623376627],
        [0.546236559139785, 0.1482683982683983],
        [0.6193548387096774, 0.16883116883116886],
        [0.578494623655914, 0.10389610389610393],
        [0.6913978494623656, 0.08766233766233769],
        [0.6817204301075269, 0.15692640692640694],
        [0.789247311827957, 0.15584415584415587],
        [0.7623655913978495, 0.23376623376623382],
        [0.8161290322580645, 0.20995670995670998],
        [0.8172043010752689, 0.27922077922077926],
        [0.9118279569892472, 0.26515151515151514],
        [0.9139784946236559, 0.26515151515151514],
        [0.8913978494623657, 0.2900432900432901],
        [0.9075268817204302, 0.31493506493506496],
        [0.8537634408602152, 0.3344155844155845],
        [0.9376344086021505, 0.4112554112554113],
        [0.8548387096774194, 0.47186147186147187],
        [0.921505376344086, 0.49675324675324684],
        [0.889247311827957, 0.5476190476190477],
        [0.8924731182795699, 0.6017316017316018],
        [0.7956989247311829, 0.6244588744588745],
        [0.8494623655913979, 0.683982683982684],
        [0.8075268817204302, 0.7402597402597403],
        [0.8075268817204302, 0.7943722943722944],
        [0.7494623655913979, 0.7911255411255412],
        [0.7709677419354839, 0.8430735930735931],
        [0.6817204301075269, 0.8571428571428572],
        [0.6645161290322581, 0.8722943722943723],
        [0.6516129032258065, 0.8409090909090909],
        [0.5451612903225806, 0.8647186147186148],
        [0.5817204301075269, 0.8203463203463204],
        [0.5548387096774194, 0.7727272727272728],
        [0.6397849462365591, 0.7727272727272728],
        [0.5903225806451613, 0.7175324675324676],
        [0.6548387096774194, 0.6872294372294373],
        [0.6290322580645161, 0.6287878787878789],
        [0.7075268817204301, 0.5984848484848485],
        [0.635483870967742, 0.5562770562770564],
        [0.6333333333333334, 0.6233766233766235],
        [0.5408602150537635, 0.5627705627705628],
        [0.5064516129032258, 0.6028138528138529],
        [0.4688172043010753, 0.5238095238095238],
        [0.3817204301075269, 0.6114718614718615],
        [0.45376344086021514, 0.6136363636363636],
        [0.413978494623656, 0.6547619047619049],
        [0.32688172043010755, 0.6136363636363636],
        [0.32688172043010755, 0.6926406926406927],
        [0.4483870967741935, 0.7218614718614719],
        [0.3838709677419355, 0.7792207792207793],
        [0.35376344086021505, 0.7554112554112554],
        [0.3075268817204301, 0.8181818181818182],
        [0.346236559139785, 0.8181818181818182],
        [0.3075268817204301, 0.8766233766233766],
    ])
    cur_points=np.copy(original_curve_points_array)

    j=_calc_j_from_array_length(len(original_curve_points_array))
    cur_level=j

    MRAGUI.build_plt(original_curve_points_array, get_new_points_to_draw_for_level, update_point_in_pointlist, slider_max_level=j)
    return

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

def get_new_points_to_draw_for_level(level):
    global original_curve_points_array, cur_points, cur_details
    points_to_draw=calc_new_points_and_return_to_draw(level)
    return points_to_draw

def calc_new_points_and_return_to_draw(level=0):
    global cur_level, cur_points
    if cur_level==level:
        #nothing to do
        c_length=_calc_length_from_j(level)
        return cur_points[:c_length]

    #TODO change for fractional level
    #for now j=level
    j_target=int(level)
    j_actual=int(cur_level)
    cur_level=level

    if j_target>j_actual:
        #make synthesis
        PQ=_calc_synthesis_matrices_linear_b_splines(j_target)
        cd_length = _calc_length_from_j(j_target)
        cd_j_minus_1=cur_points[:cd_length]
        #cd_j = np.concatenate((points_array, points_array), axis=1)
        c_j = np.dot(PQ, cd_j_minus_1)
        print("shape c_j: {}".format(c_j.shape))

        cur_points[:cd_length]=c_j

        return c_j

    if j_target<j_actual:
        #make analysis
        #TODO: Do until target is achieved
        AB=_calc_analysis_matrices_from_semiorthogonal_pq(j_actual)
        c_length=_calc_length_from_j(j_actual)
        print("j: {}".format(j_actual))
        sub_points=cur_points[:c_length]
        print("subarray shape: {}".format(sub_points.shape))

        cd_j_minus_1 = np.dot(AB, sub_points)
        #split into c and d
        # A,B=np.vsplit(AB,[2**(j-1)+1])
        c_j_minus_1 = cd_j_minus_1[:_calc_length_from_j(j_target)]
        d_j_minus_1 = cd_j_minus_1[:_calc_length_from_j(j_target)-1]
        # c_j_minus_1=np.dot(A.T,original_curve_points_array)
        # d_j_minus_1=np.dot(Q,original_curve_points_array)

        print("c^(j-1) shape: {}".format(c_j_minus_1.shape))
        print("d^(j-1) shape: {}".format(d_j_minus_1.shape))

        cur_points[:c_length]=cd_j_minus_1

        # analysis (decomposotion)
        # calc: cj-1=cj*Aj
        # calc: dj-1=cj*Bj
        #c_j_minus_1 = A * c_j
        #d_j_minus_1 = B * c_j
        # easier: solve P|Q * \frac{c^(j-1), d^(j-1)}=c^j}
        #cj_1_dj_1 = np.linalg.solve(PQ, c_j)
        # cj_1_dj_1 now are values without using AB ;)

        return c_j_minus_1



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

    print("shape P: {}, Q: {}".format(P.shape, Q.shape))

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