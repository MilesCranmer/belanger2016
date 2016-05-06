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
	return (0.5/eventTimes.size*(1.0+0.5/harmonic/(maxTime-minTime)*\
			(np.sin(2*harmonic*omega*maxTime)-\
				np.sin(2*harmonic*omega*minTime)))-\
			Ck(eventTimes,harmonic,omega))
#formula 4a
def varianceSk(eventTimes,harmonic,omega,minTime,maxTime):
	return (0.5/eventTimes.size*(1.0-0.5/harmonic/(maxTime-minTime)*\
			(np.sin(2*harmonic*omega*maxTime)-\
				np.sin(2*harmonic*omega*minTime)))-\
			Sk(eventTimes,harmonic,omega))
#formula 4b
def covarianceCkSk(eventTimes,harmonic,omega,minTime,maxTime):
	return (0.5/harmonic/omega/(maxTime-minTime)/eventTimes.size*\
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
	return abs(rkVector.dot(np.linalg.inv(varianceRemovalMatrix))\
				.dot(rkVector.transpose()))
#H-test as defined in
#Jager et al. 2010, and Jager et al. 1989
def hTestBestScore(eventTimes,maxHarmonic,omega):
    Zscores = np.cumsum([modifiedRkSquared(eventTimes,harmonic,omega)
    					for harmonic in range(1,maxHarmonic+1)])
    candidateHscores = 4+Zscores-4*np.arange(1,maxHarmonic+1)
    return np.max(candidateHscores)

############
#test code
############
dataFile = open('../../Dropbox/arrivaltimes.txt','r')
signal = np.array([float(line)*86400 for line in dataFile.readlines()])
freqs = np.arange(0.0005,0.0015,0.0000005)
omegas = freqs*np.pi*2
Hscores = [hTestBestScore(signal,5,omega)\
			for omega in omegas]
import matplotlib.pyplot as plt
plt.plot(freqs, Hscores)
plt.show()