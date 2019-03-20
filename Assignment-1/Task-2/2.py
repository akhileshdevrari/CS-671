import numpy as np
import tensorflow as tf

mass = tf.Variable(np.load('q2_input/masses.npy'))
velocity = tf.Variable(np.load('q2_input/velocities.npy'))
position = tf.Variable(np.load('q2_input/positions.npy'))
G =  tf.cast(tf.constant(6.67*(10**(5))), tf.float64)
time = tf.placeholder(tf.float64, name="HereTime",)
n=tf.constant(100)
inf=tf.constant(999999999.,tf.float64)
itr=tf.Variable(0,tf.int64)
xvel = velocity[:n,0]
yvel = velocity[:n,1]
xpos = position[:n,0]
ypos = position[:n,1]
mas=mass

xvel=tf.reshape(xvel,(n,1))
yvel=tf.reshape(yvel,(n,1))

def cond(xpos,ypos,xvel,yvel,itr):
    rx=tf.transpose([xpos])-xpos
    ry=tf.transpose([ypos])-ypos
    rdis=tf.sqrt(tf.square(rx)+tf.square(ry))
#     print(rdis)
    dig=tf.ones(rdis.shape[0:-1],tf.float64)

    dig_inf=tf.multiply(dig,inf)
#     print(rdis,dig_inf)
    rdis=tf.matrix_set_diag(rdis,dig_inf)
    min_dis=tf.math.reduce_min(rdis)
    rdis=tf.matrix_set_diag(rdis,dig)
    tf.print(min_dis)
    return min_dis>0.1

def body(xpos,ypos,xvel,yvel,itr):
    itr=itr+1
    rx=tf.transpose([xpos])-xpos
    ry=tf.transpose([ypos])-ypos
    rdis=tf.sqrt(tf.square(rx)+tf.square(ry))
    dig=tf.ones(rdis.shape[0:-1],tf.float64)

    rdis=tf.matrix_set_diag(rdis,dig)

    rxpos=tf.divide(rx,rdis**3)
    rypos=tf.divide(ry,rdis**3)

    mx=tf.matmul(rxpos,mas)
    ax=tf.multiply(mx,-1.*G)

    my=tf.matmul(rypos,mas)
    ay=tf.multiply(my,-1.*G)

    nxvel=tf.add(xvel, tf.multiply(ax,time))
    nyvel=tf.add(yvel, tf.multiply(ay,time))
    
    nx=tf.add(tf.multiply(xvel,time),tf.multiply(tf.divide(ax,2),time**2))
    nx=tf.reshape(nx,(n,))
    xpos=tf.add(xpos,nx)
    
    ny=tf.add(tf.multiply(yvel,time),tf.multiply(tf.divide(ay,2),time**2))
    ny=tf.reshape(ny,(n,))
    ypos=tf.add(ypos,ny)
    
    xvel=nxvel
    yvel=nyvel
    xvel=tf.reshape(xvel,(n,1))
    yvel=tf.reshape(yvel,(n,1))
#     print(xpos.shape,ypos.shape,xvel.shape,yvel.shape)
    return xpos,ypos,xvel,yvel,itr
init_g = tf.global_variables_initializer()
res=tf.while_loop(cond,body,loop_vars=[xpos,ypos,xvel,yvel,itr])
with tf.Session() as ss:
    ss.run(init_g)
    mn=(ss.run((res),feed_dict={time:1e-4}))
    # print(mn[3])
    
    ####### For Printing Final Positions and Velocities ######
    # pos=tf.concat([tf.expand_dims(mn[0],1),tf.expand_dims(mn[1],1)],-1)
    # vel=tf.concat([tf.expand_dims(mn[2],1),tf.expand_dims(mn[3],1)],-1)
    
    # poss=(ss.run(pos,feed_dict={time:1e-4}))
    # np.save('positions.npy',poss)

    # vels=(ss.run(vel,feed_dict={time:1e-4}))
    # np.save('velocities.npy',vels)

