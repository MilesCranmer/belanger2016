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
	return rkVector.dot(np.linalg.inv(varianceRemovalMatrix))\
				.dot(rkVector.transpose())
#formula 10
def modifiedZnSquared(eventTimes,maxHarmonic,omega):
	return 1.0/maxHarmonic*\
			sum([modifiedRkSquared(eventTimes,harmonic, omega)
					for harmonic in range(1,maxHarmonic+1)])
#Jager et al. 2010, Jager et al. 1989
def hTestBestHarmonic(eventTimes,maxHarmonic,omega):
    eventPhases = np.mod(omega/2.0/np.pi*eventTimes,1.0)
    allHarmonics = np.arange(1,maxHarmonic+1)
    #allCks = np.array([Ck(eventTimes,harmonic,omega) \
    #			for harmonic in allHarmonics])
    #allSks = np.array([Sk(eventTimes,harmonic,omega) \
    #			for harmonic in allHarmonics])
    cs = np.sum(
    	np.exp(2.j*np.pi*np.arange(1,maxHarmonic+1)*eventPhases[:,None]),
    			axis=0)/eventTimes.size
    Zm2 = 2*eventTimes.size*np.cumsum(np.abs(cs)**2)
    Zm22 = np.cumsum([modifiedRkSquared(eventTimes,harmonic,omega)
    					for harmonic in range(1,maxHarmonic+1)])
    candidateHarmonics = 4+Zm22-4*np.arange(1,maxHarmonic+1)
    bestNumberOfHarmonics = np.argmax(candidateHarmonics)+1
    return bestNumberOfHarmonics

#test code
dataFile = open('../../Dropbox/arrivaltimes.txt','r')
signal = np.array([float(line)*86400 for line in dataFile.readlines()])
freqs = np.arange(0.0005,0.0012,0.000003)
omegas = freqs*np.pi*2
bestHarmonic = hTestBestHarmonic(signal, 5,0.003)
print bestHarmonic

"""
Rks = [sum([modifiedRkSquared(signal, n, omega) for n in range(1,4)]) \
		for omega in omegas]

import matplotlib.pyplot as plt

plt.plot(freqs, Rks)
plt.show()
"""