import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.patches import Polygon

X=[]
n = int(input('Enter the number of points you want to click: '))
h=0.01
T=100


#plotting
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.set_title(f'Click {n} points on the graph')


def onclick(event):
    # Get the x and y coordinates of the clicked point
    global X
    x, y = event.xdata, event.ydata

    # Append the point to the list of vertices
    X.append((x,y))

    # Plot the new point as a red dot
    ax.plot(x, y, 'ro')

    #Start joining them by lines
    if len(X) >= 2:
        ax.plot([X[-2][0], x], [X[-2][1], y], 'k-')

    # If there are at least three points, plot the polygon
    if len(X) == n:
        poly = plt.Polygon(X, closed=True, alpha=0.5, fill=False)
        ax.add_patch(poly)
        fig.canvas.mpl_disconnect(cid)

    # Update the plot
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()


# n = 4
# X=[(5,0),(5,2),(0,2),(0,0)]
# fig, ax = plt.subplots()
# polygon = plt.Polygon(X, closed=True, fill=False)
# ax.add_patch(polygon)
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# plt.show()
# print("initial polygon=",X)




"""
k=0
for i in range(100001):
    X=flow()
    if i==10**k:
        print("Iteration", i, "Side lengths are", lengths(D(X)))
        k+=1
"""       

#Visualisation of flow
fig, ax = plt.subplots()
# plt.autoscale(enable=True, axis='both', tight=True)
polygon = plt.Polygon(X, animated=True, fill=False)
ax.add_patch(polygon)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')


#difference between successive vectors
def D(x):
    return np.roll(x,-2)-x

#unitise vectors
def unit(x):
    x=np.array(x)
    y=np.empty_like(x)
    for i in range(n):
        y[i]=x[i]/np.linalg.norm(x[i])
    return y

#rotation by 90 CCW
def J(arr):
    rotated_arr = []
    for tup in arr:
        rotated_tup = np.array([[0, -1], [1, 0]]).dot(tup)
        rotated_arr.append(rotated_tup)
    return rotated_arr

#
def lengths (X):
    a=np.zeros((n,1))
    for i in range(n):
        a[i]=np.linalg.norm(X[i])
    return a

def convex(X):
    DX=D(X)
    RX=J(DX)
    RX=np.roll(RX,2)
    A=[]
    for i in range(n):
        A[i]=DX[2*i]*RX[2*i]+DX[2*i+1]*RX[2*i+1]
    return min(A)

def flow():
    global X

    #Forward Edges
    DX=np.array(D(X))

    #Forward Tangents
    T=np.array(unit(DX))

    #Angle Bisectors
    B=T-np.roll(T,2)

    #Chords
    chords=DX+np.roll(DX,2)

    #90deg rotation of chords, aka normal
    N=J(chords)

    #dotpr of normal and bisector
    R=np.zeros(((n,1)))
    for i in range(n):
        R[i]=2*np.dot(N[i],B[i])

    #norms of angle bisectors
    bisecnorms=np.zeros(((n,1)))
    for i in range(n):
        bisecnorms[i]=np.dot(B[i],B[i])

    #avgdotpr of normal and bisector
    K=np.sum(R)/np.sum(bisecnorms)

    #Velocities
    V=np.zeros_like(DX)
    for i in range(n):
        V[i]=-2*N[i]+K*B[i]

    Y = X + h*V
    X=Y
    return Y



#Animating the flow
def update(frame):
    global X,polygon, Z
    X=flow()
    # Z.append(convex(X))
    polygon.set_xy(X)
    return [polygon]

ani=FuncAnimation(fig, update, frames=100, interval=5, blit=True )
writer = animation.PillowWriter(fps=60, bitrate=1800)
ani.save('ip_flow.gif', writer=writer)
plt.xticks([])
plt.yticks([])
plt.show()

























