# Visual Computing: Wavelets for Computer Graphics
# team01

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons
import numpy as np
import math

#resolution of control-function (rendered dashed line)
FUNCTION_DEPTH=50
#resolution of control points (for calculation)
CP_DEPTH=10

def main():
    calc_and_render_subdivision()
    return

def function1(x,x0=0):
    return (x-0.5)**2*10#10*x**7+0.3
def function2(x,x0=0):
    return np.sin(x*10)*0.5+0.5
def function3(x,x0=0):
    fp_array=np.zeros(len(x))
    idx = (np.abs(x - x0)).argmin()
    fp_array[idx]=1
    return fp_array

def get_function_values(steps, function=function1, f_param=0):
    from_v=0.0
    to_v=1.0
    step=to_v/steps
    stop=to_v+step
    x=np.arange(from_v,stop,step)
    function_values=function(x, f_param)
    fp_array = np.vstack((x, function_values)).T
    #print(fp_array)
    return fp_array

def render_subdivision_step_update(function_points_array, control_points_array, points_array, step, ax):
    ax.clear()
    #ax.axis([0.0, 1.0, 0.0, 1.0])
    ax.axis([0.0, 1.0, -0.5, 1.5])

    #x = np.arange(0.0, 1.2, 0.01)
    #function = x ** 7 + 0.3
    #function = np.sin(x * 10) * 0.5 + 0.5
    ## control_points_array = np.random.rand(10, 2)
    #control_points_array = np.vstack((x, function)).T

    #radius
    r_cp=0.0025
    r_p=0.0025
    #color
    c_cp='black'
    c_p='blue'
    c_sc='red'
    c_cc='gray'

    ann_offset = np.array([0.006, 0.003])  # offset for annotations

    if function_points_array is not None:
        for i, fp in enumerate(function_points_array):
            if i>0:
                from_p = function_points_array[i - 1]
                to_p = fp
                ax.plot([from_p[0], to_p[0]], [from_p[1], to_p[1]], 'o', ms=0.0, ls='--', lw=1.0, color=c_cc)

    if control_points_array is not None:
        for i, cp in enumerate(control_points_array):
            disc = "c{}".format(i)
            ax.add_patch(plt.Circle(cp, radius=r_cp, color=c_cp, fc=c_cp, fill=True))
            ax.annotate(disc, cp + ann_offset)

            #if i>0:
            #    from_p = control_points_array[i - 1]
            #    to_p = cp
            #    ax.plot([from_p[0], to_p[0]], [from_p[1], to_p[1]], 'o', ms=0.0, ls='-', lw=1.0, color=c_cc)

    if points_array is not None:
        for i,p in enumerate(points_array):
            ax.add_patch(plt.Circle(p, radius=r_p, color=c_p, fc=c_p, fill=False))
            if i>0:
                from_p = points_array[i - 1]
                to_p = p
                ax.plot([from_p[0], to_p[0]], [from_p[1], to_p[1]], 'o', ms=2.0, ls='-', lw=1.0, color=c_p)
    return

def calc_and_render_subdivision():
    # controlpoints as dots
    # points as circles
    # curve as line (acc. 2.2)

    #prepare plot
    fig, ax = plt.subplots(figsize=[12, 12])
    #init vis
    cur_step=0
    kronecker=0
    global FUNCTION_DEPTH, CP_DEPTH
    function_points_array = get_function_values(FUNCTION_DEPTH, function1,kronecker)
    control_points_array = get_function_values(CP_DEPTH, function1,kronecker)

    points_array = calc_subdivision(control_points_array, cur_step)
    render_subdivision_step_update(function_points_array, control_points_array, points_array, cur_step, ax)
    #Slider (widget example adapted from: https://riptutorial.com/matplotlib/example/23577/interactive-controls-with-matplotlib-widgets)
    #slider axes
    slider_ax = plt.axes([0.35, .03, 0.50, 0.02])
    subdivision_slider = Slider(slider_ax, "Subdivision Step", 0, 10, valinit=cur_step, valstep=1)

    subdivision_scheme_function=subdivision_b_spline_cubic_uniform
    function=function1

    #defined locally to have all values here
    def update_vis(step):
        #get selected step
        nonlocal cur_step, kronecker
        cur_step=step

        function_points_array=get_function_values(FUNCTION_DEPTH,function,kronecker)
        control_points_array=get_function_values(CP_DEPTH,function,kronecker)

        points_array=calc_subdivision(control_points_array, step, subdivision_scheme_function=subdivision_scheme_function)

        count_cp=len(control_points_array)
        count_p=int((count_cp-1)*math.pow(2,step)+1)#+step
        print("Replot Subdivision Step {} on {} control points with {} total points".format(step, count_cp, count_p))

        #rebuild vis on axes
        render_subdivision_step_update(function_points_array,control_points_array, points_array, step, ax)
        fig.canvas.draw_idle()

    # call update function on slider value change
    subdivision_slider.on_changed(update_vis)

    # from: https://matplotlib.org/2.0.2/examples/widgets/radio_buttons.html
    #axcolor = 'lightgoldenrodyellow'
    radio_ax = plt.axes([0.01, 0.03, 0.1, 0.15])#, facecolor=axcolor)
    radio = RadioButtons(radio_ax, ('uniform\ncub. B-Spline','uniform\nDaubechie','nonuniform\ncub. B-Spline'))
    def select_scheme(label):
        hzdict = {'uniform\ncub. B-Spline': subdivision_b_spline_cubic_uniform,
                  'nonuniform\ncub. B-Spline': subdivision_b_spline_cubic_nonuniform,
                  'uniform\nDaubechie': subdivision_daubechie_uniform}
        nonlocal subdivision_scheme_function
        subdivision_scheme_function = hzdict[label]
        update_vis(cur_step)
    radio.on_clicked(select_scheme)

    radio2_ax = plt.axes([0.01, 0.21, 0.1, 0.15])  # , facecolor=axcolor)
    radio2 = RadioButtons(radio2_ax, ('function1', 'function2', 'scaling\nfunction'))
    def select_function(label):
        hzdict = {'function1': function1,
                  'function2': function2,
                  'scaling\nfunction': function3}
        nonlocal function
        function = hzdict[label]
        update_vis(cur_step)
    radio2.on_clicked(select_function)

    kronecker_ax = plt.axes([0.01, 0.18, 0.1, 0.03])  # , facecolor=axcolor)
    kronecker_slider = Slider(kronecker_ax, "Kronecker", 0.0, 1.0, valinit=kronecker, valstep=1.0/CP_DEPTH)
    #kronecker_slider = Slider(kronecker_ax, "Kronecker", 0, CP_DEPTH, valinit=int(CP_DEPTH/2), valstep=1)
    def set_kronecker(val):
        nonlocal kronecker
        kronecker=val
        update_vis(cur_step)
    kronecker_slider.on_changed(set_kronecker)


    # this was to define points for initial curve ;)
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print("[{}, {}],".format(ix,iy))
    fig.canvas.mpl_connect('button_press_event', onclick)

    #finally show plot
    plt.show()
    return

def splitting_step_nonuniform(points_array):
    #double array
    points_array=np.repeat(points_array,2,axis=0)
    # each point is now doubled
    cp_points_array = np.copy(points_array)
    #average every second entry (control points will stay)
    for i, p in enumerate(points_array):
        if i<len(points_array)-1:
            if i%2==1:
                next_p = points_array[i + 1]
                cp_points_array[i]=0.5*(p+next_p)
    #pop last (doubled) element
    return cp_points_array[:-1]

def subdivision_b_spline_cubic_uniform(points_array):
    cp_points_array = points_array
    c_size=len(points_array)
    if c_size<3:
        print("too few values in point_array. Abort subdivision")
        return points_array
    #build up averaging matrix
    mat_r=np.zeros((c_size, c_size))
    for i in range(0,c_size):
        mat_r[i,i]=2
        if i>0: mat_r[i,i-1]=1
        if i<c_size-1: mat_r[i,i+1]=1
    #print(mat_r)
    mat_r*=1.0/4

    #apply on functions values
    points_array[:,1]=np.dot(mat_r,cp_points_array[:,1])
    return points_array

def subdivision_daubechie_uniform(points_array):
    cp_points_array = np.copy(points_array)
    one_plus_sqrt3=1.0+math.sqrt(3)
    one_minus_sqrt3=1.0-math.sqrt(3)

    for i, p in enumerate(points_array):
        if i<len(points_array)-1:
            next_p = points_array[i + 1]
            cp_points_array[i,1] = 0.5 * (one_plus_sqrt3*p[1] + one_minus_sqrt3*next_p[1])
    return cp_points_array

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

    #print(mat_r)
    mat_r*=1.0/4

    points_array[:,1]=np.dot(mat_r,cp_points_array[:,1])

    return points_array

def calc_subdivision(control_points_array, steps, subdivision_scheme_function=subdivision_b_spline_cubic_uniform):
    points_array=control_points_array
    for i in range(0,steps):
        #SPLITTING STEP
        points_array=splitting_step_nonuniform(points_array)
        #print(points_array)

        # AVG STEP
        points_array=subdivision_scheme_function(points_array)
        #points_array=subdivision_b_spline_cubic_nonuniform(points_array)

    return points_array

if __name__ == "__main__":
    # execute only if run as a script
    main()
