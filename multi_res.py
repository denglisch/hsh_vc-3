import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons
import numpy as np
import math


FUNCTION_DEPTH=50
CP_DEPTH=5

def main():
    P,Q=get_synthesis_matrices_linear_b_splines(3)
    print(P)
    print(Q)
    exit(0)
    render()
    return

def render_subdivision_step_update(control_points_array, points_array, step, ax):
    ax.clear()
    ax.axis([0.0, 1.0, 0.0, 1.0])

    #radius
    r_cp=0.0025
    r_p=0.0025
    #color
    c_cp='black'
    c_p='blue'
    c_sc='red'
    c_cc='gray'

    ann_offset = np.array([0.006, 0.003])  # offset for annotations

    if control_points_array is not None:
        for i, cp in enumerate(control_points_array):
            disc = "c{}".format(i)
            ax.add_patch(plt.Circle(cp, radius=r_cp, fc=c_cp, fill=True))
            ax.annotate(disc, cp + ann_offset)

            if i == 0:
                from_p = control_points_array[len(control_points_array) - 1]
            else:
                from_p = control_points_array[i - 1]
            to_p = cp
            ax.plot([from_p[0], to_p[0]], [from_p[1], to_p[1]], 'o', ms=2.0, ls='-', lw=1.0, color=c_cc)

    if points_array is not None:
        for i,p in enumerate(points_array):
            ax.add_patch(plt.Circle(p, radius=r_p, fc=c_p, fill=False))
            if i==0:
                from_p=points_array[len(points_array)-1]
            else:
                from_p=points_array[i-1]
            to_p=p
            ax.plot([from_p[0], to_p[0]], [from_p[1], to_p[1]], 'o', ms=2.0, ls='-', lw=1.0, color=c_sc)
    return

def render(control_points_array, points_array):
    #prepare plot
    fig, ax = plt.subplots(figsize=[12, 12])
    #init vis
    cur_step=0
    points_array = calc_subdivision(control_points_array, cur_step)
    render_subdivision_step_update(control_points_array, points_array, cur_step, ax)
    #Slider (widget example adapted from: https://riptutorial.com/matplotlib/example/23577/interactive-controls-with-matplotlib-widgets)
    #slider axes
    slider_ax = plt.axes([0.35, .03, 0.50, 0.02])
    subdivision_slider = Slider(slider_ax, "Level", 0, 10, valinit=cur_step, valstep=1)

    #defined locally to have all values here
    def update_vis(step):
        #get selected step
        nonlocal cur_step
        cur_step=step

        points_array=calc_subdivision(control_points_array, step)

        count_cp=len(control_points_array)
        count_p=count_cp*math.pow(2,step)#+step
        print("Replot at Level {} on {} control points with {} total points".format(step, count_cp, count_p))

        #rebuild vis on axes
        render_subdivision_step_update(control_points_array, points_array, step, ax)
        fig.canvas.draw_idle()

    # call update function on slider value change
    subdivision_slider.on_changed(update_vis)

    # this was to define points for initial curve ;)
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print("[{}, {}],".format(ix,iy))
    fig.canvas.mpl_connect('button_press_event', onclick)

    #finally show plot
    plt.show()
    return

def splitting_step(points_array):
    points_array=np.repeat(points_array,2,axis=0)
    # each point is now doubled
    cp_points_array = np.copy(points_array)
    for i, p in enumerate(points_array):
        if i%2==1:
            next_p = points_array[i + 1] if i<len(points_array)-1 else points_array[0]
            cp_points_array[i]=0.5*(p+next_p)
    return cp_points_array

def splitting_step_nonuniform(points_array):
    points_array=np.repeat(points_array,2,axis=0)
    # each point is now doubled
    cp_points_array = np.copy(points_array)
    for i, p in enumerate(points_array):
        if i<len(points_array)-1:
            if i%2==1:
                next_p = points_array[i + 1]
                cp_points_array[i]=0.5*(p+next_p)
    #pop last (doubled) element
    return cp_points_array[:-1]

def subdivision_b_spline_cubic_nonuniform(points_array):
    cp_points_array=points_array
    c_size=len(cp_points_array)

    if c_size<8:
        print("too few values in point_array. Abort subdivision")
        return points_array

    #build up averaging matrix
    mat_r=np.zeros((c_size, c_size))
    for i in range(3,c_size-3):
        mat_r[i,i]=2
        if i>0: mat_r[i,i-1]=1
        if i<c_size-1: mat_r[i,i+1]=1
    mat_r[0,0]=4
    mat_r[c_size-1,c_size-1]=4

    mat_r[1,1]=4
    mat_r[c_size-2,c_size-2]=4

    mat_r[2,2]=2
    mat_r[2,1]=2

    mat_r[c_size-3,c_size-3]=2
    mat_r[c_size-3,c_size-2]=2

    mat_r[3,2]=3.0/2
    mat_r[3,3]=3.0/2
    mat_r[c_size-4,c_size-3]=3.0/2
    mat_r[c_size-4,c_size-4]=3.0/2

    mat_r*=1.0/4

    #print(mat_r)
    points_array[:,1]=np.dot(mat_r,cp_points_array[:,1])

    #for i in range(0,c_size):
        #if i%2==1:
            #points_array[:,1]=cp_points_array[:,1]
    #print(points_array)

    return points_array

def get_synthesis_matrices_linear_b_splines(j):

    #calc mat-sizes
    cols_of_P=j+1
    #shifted down by 2 (for linear) plus endpoint
    rows_of_P=2*j+1
    #quadratic
    rows_of_Q=rows_of_P
    cols_of_Q=rows_of_P-cols_of_P

    P = np.zeros((rows_of_P, cols_of_P))
    Q = np.zeros((rows_of_Q, cols_of_Q))

    #P is always the same
    for col in range(0, cols_of_P):
        if col*2-1>0:P[col*2-1, col] = 1
        P[col*2-1+1, col] = 2
        if col*2-1+2<rows_of_P:P[col*2-1+2, col] = 1

    P=0.5*P

    #Q depends on size
    if j==1:
        Q[0,0]=-1
        Q[1,0]=1
        Q[2,0]=-1

        sqrt_3=math.sqrt(3.0)
        Q=sqrt_3*Q
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
        Q=sqrt_3_64*Q
    if j>=3:
        for col in range(1, cols_of_Q - 1):
            Q[col*2-1, col] = 1
            Q[col*2-1+1, col] = -6
            Q[col*2-1+2, col] = 10
            Q[col*2-1+3, col] = -6
            Q[col*2-1+4, col] = 1

        #fix values
        Q[0,0]=-11.022704
        Q[rows_of_Q-1,cols_of_Q-1]=-11.022704

        Q[1,0]=10.1041145
        Q[rows_of_Q-1,cols_of_Q-1]=10.1041145

        Q[2,0]=-5.511352
        Q[rows_of_Q-3,cols_of_Q-1]=-5.511352

        Q[3,0]=0.918559
        Q[rows_of_Q-4,cols_of_Q-1]=0.918559

        sqrt_2j_72=math.sqrt(2.0**j/72)
        Q=sqrt_2j_72*Q

    return P,Q

def calc_multi_res():
    #TODO
    j=3
    #sythesis
    P,Q=get_synthesis_matrices_linear_b_splines(j)
    #analysis
    #only for orthogonal
    A=P.T
    B=Q.T
    #B-Splines W are semi-orthpogonal
    #\frac{A,B}=P|Q^-1
    #easier: solve P|Q * \frac{c^(j-1), d^(j-1)}=c^j}

    c_j=[]
    #RECURSIVLY

    #analysis (decomposotion)
    c_j_minus_1=A*c_j
    d_j_minus_1=B*c_j

    #synthesis (reconstruction)
    c_j=P*c_j_minus_1+Q*d_j_minus_1


    return

def calc_subdivision(control_points_array, steps):
    points_array=control_points_array
    for i in range(0,steps):
        #SPLITTING STEP
        points_array=splitting_step_nonuniform(points_array)
        #print(points_array)
        #print(points_array.size)

        # AVG STEP
        #points_array=subdivision_scheme_function(points_array)
        #points_array=subdivision_b_spline_cubic_nonuniform(points_array)

    return points_array

if __name__ == "__main__":
    # execute only if run as a script
    main()