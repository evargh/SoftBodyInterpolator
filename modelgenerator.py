import bpy
import os
import bpy_extras
import mathutils

# TODO: rotate camera around center axis, change focal length, change up/down
#       change ball weight, speed, angle, deformation

# to measure the ball's size,
# for each camera setting, create two empties around the origin such that their distance is
# 2m apart. ensure that theyre being made in a way that that they form an isoceles traiangle
# with the camera point - ALMOST DONE
# then add a copy location constraint to each of them on the empty
# that way we can get a rough estimate of the ball size in the image plane

scn = bpy.context.scene
bb = bpy.data.objects['Empty']
cam = bpy.data.objects['Camera']
cursor = bpy.context.scene.cursor

zaxis = Vector((0, 0, 1))

get_normal_axis = cross(cam.location,zaxis).normalized()
get_inverse_normal_axis = get_normal_axis * -1

if not os.path.exists('animation1'):
    os.mkdir('animation1')
bpy.context.scene.render.filepath = "animation1/example"
bpy.ops.render.render(write_still=True, use_viewport=True)
os.chdir('animation1')

with open("3d_location.txt", 'w') as location_file_3d, open("planar_location.txt", 'w') as location_file_planar:
    for f in range(scn.frame_start, scn.frame_end):
        scn.frame_set(f)
        ballpos = bb.matrix_world.translation

        # get 3-space coordinates of ball center according to global axes
        location_file_3d.write('{frame},{coord1},{coord2},{coord3}\n'.format(frame=f,
            coord1=ballpos[0],coord2=ballpos[1],coord3=ballpos[2]))
        
        # get coordinates of ball center according to global axes
        co_2d = bpy_extras.object_utils.world_to_camera_view(scn, cam, ballpos)
        location_file_planar.write('{frame},{coord1},{coord2},{coord3}\n'.
            format(frame=f,coord1=co_2d[0]*scn.render.resolution_x,coord2=co_2d[1]*scn.render.resolution_y,coord3=co_2d[2]))
        
os.chdir('..')