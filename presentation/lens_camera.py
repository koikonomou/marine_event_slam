import numpy
import matplotlib.pyplot as plt


def draw_lens( lens_x, lens_height, lens_y0=0 ):
    lens_y = numpy.linspace( -lens_height/2, +lens_height/2, 200 )
    lens_thickness = 0.1

    # Left and right outer curvature vectors
    x_lens_left = lens_x - lens_thickness * (1 - 3*(lens_y / lens_height)**2)
    x_lens_right = lens_x + lens_thickness * (1 - 3*(lens_y / lens_height)**2)

    # Fill and outline the curved physical glass element
    ax.fill_betweenx(lens_y0+lens_y, x_lens_left, x_lens_right, color='skyblue', alpha=0.4, zorder=2)
    ax.plot(x_lens_left, lens_y0+lens_y, color='dodgerblue', lw=1.5, zorder=3)
    ax.plot(x_lens_right, lens_y0+lens_y, color='dodgerblue', lw=1.5, zorder=3)


def draw_camera( lens_x, sensor_x, aperture_radius ):

    draw_lens( lens_x, 2*aperture_radius )

    # Lens and Sensor
    ax.axhline(0, color='gray', linestyle=':', linewidth=1) # Optical Axis
    ax.plot([lens_x, lens_x], [-2, 2], 'b-', lw=2, label='Lens Plane') # Lens
    ax.plot([sensor_x, sensor_x], [-1.5,1.5], 'b-', lw=3, label='Sensor Plane') # Sensor
    ax.text(sensor_x, 2.2, "Sensor Plane", color='black', ha='center', fontweight='bold')

    # Aperture Blades
    ax.plot([lens_x, lens_x], [aperture_radius, 2], 'k-', lw=6)
    ax.plot([lens_x, lens_x], [-2, -aperture_radius], 'k-', lw=6)
    ax.text(lens_x, 2.2, "Lens Plane", color='black', ha='center', fontweight='bold')


def draw_object( obj_x, obj_y, focal_x, focal_y, c ):
    if c == "lightgreen": textcol = "green"
    elif c == "cyan": textcol = "darkcyan"
    ax.scatter( [obj_x], [obj_y], color=textcol, s=120, zorder=5 )
    ax.text(obj_x, obj_y + 0.3, "Light-reflecting\nsurface", color=textcol, ha='center', fontweight='bold')

    ax.plot( [obj_x, 0], [obj_y,aperture_radius], color=c, alpha=0.5, lw=1.5, linestyle="--" )
    ax.plot( [obj_x, 0], [obj_y,-aperture_radius], color=c, alpha=0.5, lw=1.5, linestyle="--" )

    slope_top_trace = (focal_y-aperture_radius)/focal_x
    slope_bot_trace = (focal_y+aperture_radius)/focal_x
    y_top = slope_top_trace*sensor_x+aperture_radius
    y_bot = slope_bot_trace*sensor_x-aperture_radius
    if focal_x >= sensor_x:
        ax.plot( [0, focal_x], [aperture_radius,focal_y], color=c, alpha=0.5, lw=1.5, linestyle="--" )
        ax.plot( [0, focal_x], [-aperture_radius,focal_y], color=c, alpha=0.5, lw=1.5, linestyle="--" ) 
        # Mark the print on the sensor
        ax.plot( [sensor_x-0.01,sensor_x-0.01], [y_top,y_bot], color="red" )
    else:
        # Find where the lines cross
        #cross_x = 2*aperture_radius * slope_bot_trace / slope_top_trace
        ax.plot( [0, sensor_x], [aperture_radius,y_top], color=c, alpha=0.5, lw=1.5, linestyle="--" ) 
        ax.plot( [0, sensor_x], [-aperture_radius,y_bot], color=c, alpha=0.5, lw=1.5, linestyle="--" ) 
        ax.plot( [sensor_x-0.01,sensor_x-0.01], [y_top,y_bot], color="red" )


lens_x = 0
sensor_x = 3
aperture_radius = 0.6


def make_schema0():
    draw_camera( lens_x, sensor_x, aperture_radius )
    draw_object( -4, 0.5, sensor_x-0.5,-0.3, 'lightgreen' )
    draw_object( -3,-0.5, sensor_x+0.3, 1.1, 'cyan' )

    plt.tight_layout()
    plt.savefig('lens_camera.png', dpi=300, bbox_inches='tight')
    plt.close()


def make_schema1():
    draw_lens( lens_x, 2*aperture_radius )
    ax.axhline(0, color='gray', linestyle=':', linewidth=1) # Optical Axis
    ax.plot([sensor_x, sensor_x], [-1.5,1.5], 'b-', lw=3, label='Sensor Plane') 

    ax.plot( [0, sensor_x], [0.8,0.8], color="grey", linestyle="--", lw=1 )
    plt.plot( 0, 0.8, marker="<", markersize=3, color="grey" )
    plt.plot( sensor_x-0.01, 0.8, marker=">", markersize=3, color="grey" )
    ax.text( sensor_x/2, 0.9, "focal length f", color='grey', ha='center' )
    
    ax.plot( [sensor_x+0.2, sensor_x+0.2], [-1.5,1.5], color="grey", linestyle="--", lw=1 )
    plt.plot( sensor_x+0.2, 1.5, marker="^", markersize=3, color="grey" )
    plt.plot( sensor_x+0.2,-1.5, marker="v", markersize=3, color="grey" )
    ax.text( sensor_x+0.3,  0, "sensor dimension d", color='grey', ha='left', va="center", rotation=90 )
    
    plt.tight_layout()
    plt.savefig('camera_intrinsics.png', dpi=300, bbox_inches='tight')
    plt.close()


def make_schema2():

    cameraL_y = -1
    cameraR_y =  1
    lens_x = 0
    draw_lens( lens_x, 0.3, cameraR_y )
    draw_lens( lens_x, 0.3, cameraL_y )

    sensor_x = 1.5
    sensor_d = 0.6
    ax.plot([sensor_x, sensor_x], [cameraR_y-sensor_d/2,cameraR_y+sensor_d/2], 'b-', lw=3, label='Righr Sensor') 
    ax.plot([sensor_x, sensor_x], [cameraL_y-sensor_d/2,cameraL_y+sensor_d/2], 'b-', lw=3, label='Left Sensor') 

    ax.text(sensor_x + 0.4, cameraR_y+0.3, "Right\nCamera", color='black', ha='left', va='center' )
    ax.text(sensor_x + 0.4, cameraL_y-0.3, "Left\nCamera", color='black', ha='left', va='center' )

    # ray traces
    obj_x, obj_y = -4, 0
    ax.scatter( [obj_x], [obj_y], color="green", s=120, zorder=5 )
    plt.plot( [obj_x,0,sensor_x], [0, 1,cameraR_y-0.25], color="grey", lw=1 )
    plt.plot( [obj_x,0,sensor_x], [0,-1,cameraL_y+0.2], color="grey", lw=1 )

    ax.text(sensor_x + 0.2, cameraR_y-0.251, r"$w_R$", color='black', ha='left', va='center' )
    ax.text(sensor_x + 0.2, cameraL_y+0.2, r"$w_L$", color='black', ha='left', va='center' )

    # Focal length
    ax.plot( [lens_x,sensor_x], [cameraR_y,cameraR_y], color="grey", linestyle="--", lw=1 )
    #plt.plot( lens_x,  cameraR_y, marker="<", markersize=3, color="grey" )
    #plt.plot( sensor_x,cameraR_y, marker=">", markersize=3, color="grey" )
    ax.text( (lens_x+sensor_x)/2, cameraR_y+0.1, r"$f_x$", color='black', ha='center', va='center' )

    # Depth
    ax.plot( [obj_x,0], [cameraR_y,cameraR_y], color="grey", linestyle="--", lw=1 )
    #plt.plot( obj_x, cameraR_y, marker="<", markersize=3, color="grey" )
    #plt.plot( 0,     cameraR_y, marker=">", markersize=3, color="grey" )
    ax.text( obj_x/2, cameraR_y+0.1, r"$D$", color='black', ha='center', va='center' )
    
    # Width
    ax.plot( [obj_x,obj_x], [cameraR_y,0], color="grey", linestyle="--", lw=1 )
    #plt.plot( obj_x,cameraR_y, marker="^", markersize=3, color="grey" )
    #plt.plot( obj_x,0        , marker="v", markersize=3, color="grey" )
    ax.text( obj_x+0.1, cameraR_y/2, r"$W_L$", color='black', ha='left', va='center' )
    
    # Baseline
    ax.plot( [lens_x,lens_x], [cameraR_y,cameraL_y], color="grey", linestyle="--", lw=1 )
    #plt.plot( lens_x,cameraR_y, marker="^", markersize=3, color="grey" )
    #plt.plot( lens_x,cameraL_y, marker="v", markersize=3, color="grey" )
    ax.text( lens_x+0.1,0, r"$B$", color='black', ha='left', va='center' )


    plt.tight_layout()
    plt.savefig('stereo.png', dpi=300, bbox_inches='tight')
    plt.close()



fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(-5, 4)
ax.set_ylim(-2, 2.5)
ax.axis('off')
#make_schema0()

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-5, 4)
ax.set_ylim(-2, 2.5)
ax.axis('off')
#make_schema1()

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-5, 4)
ax.set_ylim(-2, 2.5)
ax.axis('off')
make_schema2()


