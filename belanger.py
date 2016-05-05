import numpy as np
#formula 5
def Ck(eventTimes,harmonic,omega):
	return 1.0/eventTimes.size*np.cos(harmonic*eventTimes*omega).sum()
#formula 6
def Sk(eventTimes,harmonic,omega):
	return 1.0/eventTimes.size*np.sin(harmonic*eventTimes*omega).sum()
#formula 7
def expectedCk(harmonic,omega,minTime,maxTime):
	return 1.0/harmonic/omega/(maxTime-minTime)* \
			(np.sin(harmonic*omega*maxTime)- \
			 np.sin(harmonic*omega*minTime))
#formula 8
def expectedSk(harmonic,omega,minTime,maxTime):
	return -1.0/harmonic/omega/(maxTime-minTime)* \
			(np.cos(harmonic*omega*maxTime)- \
			 np.cos(harmonic*omega*minTime))
#formula 9
def varianceCk(eventTimes,harmonic,omega,minTime,maxTime):
	return abs(0.5/eventTimes.size*(1.0+0.5/harmonic/(maxTime-minTime)*\
			(np.sin(2*harmonic*omega*maxTime)-\
				np.sin(2*harmonic*omega*minTime)))-\
			Ck(eventTimes,harmonic,omega))
#formula 4a
def varianceSk(eventTimes,harmonic,omega,minTime,maxTime):
	return abs(0.5/eventTimes.size*(1.0-0.5/harmonic/(maxTime-minTime)*\
			(np.sin(2*harmonic*omega*maxTime)-\
				np.sin(2*harmonic*omega*minTime)))-\
			Sk(eventTimes,harmonic,omega))
#formula 4b
def covarianceCkSk(eventTimes,harmonic,omega,minTime,maxTime):
	return abs(0.5/harmonic/omega/(maxTime-minTime)/eventTimes.size*\
			(np.power(np.sin(harmonic*omega*maxTime),2)-\
			 np.power(np.sin(harmonic*omega*minTime),2))-\
			expectedCk(harmonic,omega,minTime,maxTime)*\
			expectedCk(harmonic,omega,minTime,maxTime))
#formula 3
def modifiedRkSquared(eventTimes,harmonic,omega):
	minTime = np.min(eventTimes)
	maxTime = np.max(eventTimes)
	rkVector = np.array([Ck(eventTimes,harmonic,omega)-\
						expectedCk(harmonic,omega,minTime,maxTime),
						Sk(eventTimes,harmonic,omega)-\
						expectedSk(harmonic,omega,minTime,maxTime)])
	varianceRemovalMatrix = \
		np.array([
			[varianceCk(eventTimes,harmonic,omega,minTime,maxTime),
			covarianceCkSk(eventTimes,harmonic,omega,minTime,maxTime)],
			[covarianceCkSk(eventTimes,harmonic,omega,minTime,maxTime),
			varianceSk(eventTimes,harmonic,omega,minTime,maxTime)]])

	return abs(rkVector.dot(np.linalg.inv(varianceRemovalMatrix)).dot(rkVector.transpose()))

dataFile = open('arrivaltimes.txt','r')
signal = np.array([float(line) for line in dataFile.readlines()])
freqs = np.arange(0.008,0.012,0.00003)
omegas = freqs*np.pi*2

Rks = [sum([modifiedRkSquared(signal, n, omega) for n in range(1,2)]) \
		for omega in omegas]

import matplotlib.pyplot as plt

plt.plot(freqs, Rks)
plt.show()