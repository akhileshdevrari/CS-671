import numpy as np
import tensorflow as tf

mass = tf.Variable(np.load('q2_input/masses.npy'), name="mass")
velocity = np.load('q2_input/velocities.npy')
position = np.load('q2_input/positions.npy')

G =  tf.cast(tf.constant(6.67*(10**(5))), tf.float64,name="G")
time = tf.placeholder(tf.float64, name="time",)
n=tf.constant(100,name="n")
inf=tf.constant(999999999.,tf.float64,name="inf")

xvel = tf.Variable(velocity[:,0],tf.float64, name="xvel")
yvel = tf.Variable(velocity[:,1],tf.float64,name="yvel")

xpos = tf.Variable(position[:,0],tf.float64, name="xpos")
ypos = tf.Variable(position[:,1],tf.float64,name="ypos")

xve=tf.reshape(xvel,(n,1),name="xve")
yve=tf.reshape(yvel,(n,1),name="yve")

mas=mass

rx=tf.transpose([xpos])-xpos
ry=tf.transpose([ypos])-ypos
rdis=tf.sqrt(tf.square(rx)+tf.square(ry))

dig=tf.ones((n),tf.float64)

dig_inf=tf.multiply(dig,inf)

rdis=tf.matrix_set_diag(rdis,dig_inf)

min_dis=tf.math.reduce_min(rdis)
rdis=tf.matrix_set_diag(rdis,dig)

rxpos=tf.divide(rx,rdis**3)
rypos=tf.divide(ry,rdis**3)

mx=tf.matmul(rxpos,mas)
ax=tf.multiply(mx,-1.*G,name="ax")

my=tf.matmul(rypos,mas)
ay=tf.multiply(my,-1.*G,name="ay")

nx=tf.add(tf.multiply(xve,time),tf.multiply(tf.divide(ax,2),time**2))
nx=tf.reshape(nx,(n,))
xposs=tf.add(xpos,nx,name="new_xpos")

ny=tf.add(tf.multiply(yve,time),tf.multiply(tf.divide(ay,2),time**2))
ny=tf.reshape(ny,(n,))
yposs=tf.add(ypos,ny,name="new_ypos")

nxvel=tf.add(xve, tf.multiply(ax,time),name="vx")
nyvel=tf.add(yve, tf.multiply(ay,time),name="vy")

nxvel=tf.reshape(nxvel,(n,),name="new_xvel")
nyvel=tf.reshape(nyvel,(n,),name="new_yvel")

tf.summary.scalar("min_dis",min_dis)
tf.summary.histogram("xpos", xpos)
tf.summary.histogram("ypos", ypos)
tf.summary.histogram("xvel", xvel)
tf.summary.histogram("yvel", yvel)

init_g = tf.global_variables_initializer()

with tf.Session() as ss:
    ss.run(init_g)
    train_writer = tf.summary.FileWriter( './logs ', ss.graph)
    
    for i in range(332):
        merge = tf.summary.merge_all()
        summary,zz=ss.run([merge,min_dis],feed_dict={time:1e-4})
        print(i,"-> MIN Distance :",zz)
        train_writer.add_summary(summary, i)
        if(zz<0.1):
            break
        xx=xpos.assign(xposs)
        yy=ypos.assign(yposs)
        uu=xvel.assign(nxvel)
        vv=yvel.assign(nyvel)
        ss.run((xx,yy,uu,vv),feed_dict={time:1e-4})
