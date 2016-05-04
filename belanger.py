import numpy as np

#formula 5
def expectedCk (k, omega, t1, t2):
	return 1.0/k/omega/(t2-t1)*(np.sin(k*omega*t2)-np.sin(k*omega*t1))
#formula 6
def expectedSk (k, omega, t1, t2):
	return -1.0/k/omega/(t2-t1)*(np.cos(k*omega*t2)-np.cos(k*omega*t1))
#formula 7
def varCk (k, omega, t1, t2, N):
	return 0.5/N*(1+0.5*(np.sin(2*k*omega*t2)-np.sin(2*k*omega*t1))/k/omega/(t2-t1))-expectedCk(k, omega, t1, t2)
#formula 8
def varSk (k, omega, t1, t2, N):
	return 0.5/N*(1-0.5*(np.sin(2*k*omega*t2)-np.sin(2*k*omega*t1))/k/omega/(t2-t1))-expectedSk(k, omega, t1, t2)
#formula 9
def covarCkSk (k, omega, t1, t2, N):
	return 0.5/k/omega/N/(t2-t1)*(np.power(np.sin(k*omega*t2),2)-np.power(np.sin(k*omega*t1),2))-expectedCk(k, omega, t1, t2)*expectedSk(k, omega, t1, t2)
#formula 4a
def Ck(a,k,omega):
	return 1.0/a.size*np.cos(omega*k*a).sum()
#formula 4b
def Sk(a,k,omega):
	return 1.0/a.size*np.sin(omega*k*a).sum()
#formula 3
def Rk(a,k,omega):
	t1 = np.min(a)
	t2 = np.max(a)
	N = a.size
	q = np.array([Ck(a,k,omega)-expectedCk (k, omega, t1, t2),Sk(a,k,omega)-expectedSk (k, omega, t1, t2)])
	co = covarCkSk(k,omega,t1,t2,N)
	M = np.array([[varCk(k, omega, t1, t2, N),co],[co,varSk(k, omega, t1, t2, N)]])
	return q.dot(np.linalg.inv(M)).dot(q.transpose())
	#return 2*a.size*(np.power(Ck(a,k,omega),2.0)+np.power(Sk(a,k,omega),2.0))

#formula 10
def Zk(a,k,omega):
	cumul = 0
	for i in range(1,k+1):
		cumul += Rk(a,i,omega)
	return cumul







##(Function tests)
#generate white noise
noise = np.random.rand(5000)*10000
signal = 1.0/40.5*np.random.randint(0,90,size=[1000]).astype(np.float)*5000
#combine
recv = np.clip(np.concatenate((noise,signal)), 0, 10000)

freqs = np.arange(1e-3,1e-2,step=1e-5)
omegas = 2*np.pi*freqs
Rks = []
for omega in omegas:
	Rks.append(Zk(recv,3,omega))

import matplotlib.pyplot as plt
plt.plot(Rks)
plt.show()
