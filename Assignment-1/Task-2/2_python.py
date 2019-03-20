import numpy as np

mass=np.load("q2_input/masses.npy")
velocity=np.load("q2_input/velocities.npy")
position=np.load("q2_input/positions.npy")

n=100
G=6.67*1e+5

xvel=velocity[:n,0]
xvel=np.reshape(xvel,(n,1))

yvel=velocity[:n,1]
yvel=np.reshape(yvel,(n,1))

xpos=position[:n,0]
ypos=position[:n,1]

mas=mass[:n]
mas=np.reshape(mas,(n,1))

for z in range(332):

    nxpos=np.repeat(xpos,1,axis=0)
    txpos=np.transpose([nxpos])
    rxpos=txpos-nxpos

    nypos=np.repeat(ypos,1,axis=0)
    typos=np.transpose([nypos])
    rypos=typos-nypos

    rpos=np.array([[[rxpos[x][i],rypos[x][i]] for i in range(0,n)] for x in range(0,n)])

    rdis=[]
    for i in range(0,n):
        r=[]
        for j in range(0,n):
            if (i==j):
                r.append(1.0)
            else:
                r.append(np.sqrt(np.add(rpos[i][j][0]**2,rpos[i][j][1]**2)))
        rdis.append(r)

    rdis=np.array(rdis)

    rxpos=np.divide(rxpos,rdis**3)
    rypos=np.divide(rypos,rdis**3)

    dis=[]
    for i in range(0,n):
        for j in range(i+1,n):
            dis.append(np.sqrt(np.add(rpos[i][j][0]**2,rpos[i][j][1]**2)))
    print(z,"-> MIN Distance :", min(dis))
    if(min(dis)<0.1):
        break
    
    mx=np.dot(rxpos,mas)
    ax=np.multiply(mx, -1.0*G)
    my=np.dot(rypos,mas)
    ay=np.multiply(my, -1.0*G)

    time=1e-4

    nxvel=np.add(xvel, np.multiply(ax,time))
    nyvel=np.add(yvel, np.multiply(ay,time))

    nx=np.add(np.multiply(xvel,time),np.multiply(np.divide(ax,2),time**2))
    nrxpos=nx
    ny=np.add(np.multiply(yvel,time),np.multiply(np.divide(ay,2),time**2))
    nrypos=ny
    
    nrxpos=np.reshape(nrxpos,(n,))
    nrypos=np.reshape(nrypos,(n,))

    xpos=np.add(xpos,nrxpos)
    ypos=np.add(ypos,nrypos)
    xvel=nxvel
    yvel=nyvel

rpos=np.array([[xpos[i],ypos[i]] for i in range(0,n)])
xvel=np.reshape(xvel,(n,))
yvel=np.reshape(yvel,(n,))
rvel=np.array([[xvel[i],yvel[i]] for i in range(0,n)])

print("POSITION :\n",rpos,rpos.shape)
print("VELOCITY :\n",rvel,rvel.shape)
