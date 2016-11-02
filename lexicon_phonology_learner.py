import sys
import math
import re
import timeit
import random
import argparse  #TO DO: Make sensible command-line handling?

try:
	from scipy.stats import rv_discrete
	import numpy as np 
	hasNumpy=True
except:
	print "It looks like you don't have one or both of numpy and scipy installed.  You should probably just google how to install those on your machine.  What you really need here is the function numpy.random.choice().  For some reason, this function doesn't always work when you install numpy with pip, so if that happens to you, maybe try installing Anaconda or something similar.  You can still use the program, but it will be really slow, probably intolerably slow with larger datasets."
	hasNumpy=False

try:
	import matplotlib.pyplot as plt
	import matplotlib.colors as colors
except:
	print "It looks like you don't have matplotlib installed.  That's ok, you can still run the program, but you won't be able to use the plotting functions in the Results object."

# Questions.  Do you need numpy to have scipy?  Should this test be different?  Should I also test specifically for random.choice?



class candidate:
	def __init__(self,c,violations,observedProb,surfaceForm=None):
		self.c = c # the actual candidate that's got a violation profile to learn from. If using hidden structure, this is the hidden structure.
		self.violations = violations # list of violations, in same order as constraints
		if surfaceForm is None:
			self.surfaceForm = c # If no surface form given, assume c is the surface form.
		else:
			self.surfaceForm = surfaceForm # This is surface version if there's hidden structure
		self.observedProb = observedProb # The observed probability of the candidate
		# NOTE/TODO: What should this be in case of hidden structure?
		try: # make sure observedProb is a float, or can be converted to float
			self.observedProb = float(self.observedProb)
		except ValueError:
			print "It looks like your candidates' probabilities can't all be converted to floats.  Check if there's some text in that column"
		self.harmony = 0 # Not specified at initialization.  This will be filled out in learning
		self.predictedProb = 0 # Again, to be filled out during learning.
		self.checkViolationsSign() # On initialization, make sure all violations are negative
	def checkViolationsSign(self): 
		'''convert all violations to negative numbers, and by the way change blank cells to 0's'''
		for i in range(0,len(self.violations)):
			# Convert to float - they'll probably be text when read from file
			# First, change blanks into 0's
			if self.violations[i]=="":
				self.violations[i]=0
			try:
				self.violations[i] = float(self.violations[i])
			except ValueError:
				print "It looks like some of your violations can't be interpreted as numbers.  Check whether there's some text in your violation profiles"
			if self.violations[i]>0:
				self.violations[i] = - self.violations[i]

class UR:
	def __init__(self,ur,prob=1,lexC=None):
		self.ur = ur
		self.prob = prob # Lexical frequency of this form
		# Check if the probability can be sensibly converted to a float
		try:
			self.prob = float(prob)
		except ValueError:
			print "It looks like your token probabilities can't all be converted to floats.  Check if there's some text in that column"
		self.candidates = [] # To be filled during read in function
		if lexC is None:
			self.lexC = [] # This can later be filled with tuples sub-lists
		# e.g. [output, weight]
		# it's important that output is a surface form, not a hidden structure
		self.probDenom = 1 # denominator for calculating MaxEnt prob of each candidate
						   # this is the sum of each exponentiated harmony score
		self.predProbsList = [] # list of each candidate's predicted probability; used in sampling
		self.obsProbsList = [] # list of each candidate's observed probability; used in sampling
		self.lastSeen = 0 # Keep track of what timestep the learner last saw this word
		self.nSeen = 0 # Keep track of how many times the UR has been seen
		self.probableCandidates = [] #List of all candidates with observed probabilities greater than 5%, or 0.05
	def addCandidate(self,cand):
		self.candidates.append(cand)
		self.obsProbsList.append(cand.observedProb)
	def getProbableCandidates(self):
		#This should get run after all the candidates are added to the UR
		#Returns a list of surface forms for all candidates with an observed probability greater than 5%
		for c in self.candidates:
			if c.observedProb >= 0.05:
				self.probableCandidates.append(c.surfaceForm)
	def decayLexC(self,t,decayRate,decayType='static'):
		if len(self.lexC)>0:
			#try:
				indie=0
				for i in range(0,len(self.lexC)):
					if decayType=='static':
						#print self.lexC
						#print i
						#print self.lexC[i][1]
						self.lexC[i][1]-=(t-self.lastSeen)*decayRate
						#indie=i

					if decayType=='linear':
						#Create the crazy time series
						W=self.lexC[i][1]
						for timestep in range(self.lastSeen,t):
							W=W-decayRate*W
						self.lexC[i][1]=W

					if decayType=='nonlinear':
						#Create the crazy time series
						W=self.lexC[i][1]
						for timestep in range(self.lastSeen,t):
							W=W-decayRate*pow(W,2)
						self.lexC[i][1]=W

				rmlist=[]
				for i in range(0,len(self.lexC)):
					
					if self.lexC[i][1]<=0: #If it's gotten as low as zero remove the constraint
						rmlist.append(self.lexC[i])

				for i in rmlist:
					#print i
					self.lexC.remove(i)

		#NOTE lastSeen gets updated whenever the UR gets sampled, inside Tableaux.update()

			#except IndexError:
			#	print "Um, yikes. There's an index error!  Better fix that..."
			#	print "UR: "+self.ur
			#	print "LexC's: ",self.lexC
			#	print "length of lexC's: ",len(self.lexC)
			#	print indie
			#	print self.lexC[indie][1]
			#	print self.lastSeen
			#	print self.lexC[indie][1]-(t-self.lastSeen)*decayRate
	def checkViolationLength(self): 
		'''check if all the violation vectors are the same length'''
		pass
	def calculateHarmony(self, w, t=None, decayRate=None, decayType=None, suppressLexC=False): 
		'''Takes a vector of weights, equal in length to the violation vectors of the candidates.  Populates the harmony parameter of each candidate based on the weight vector w'''
		self.probDenom=0 # Reset probDenom
		if (decayRate is not None) and (t is None):
			sys.exit("You have to pass a time parameter to the calculateHarmony function if you're using a decay rate")
		if decayRate is not None:
			self.decayLexC(t,decayRate,decayType)
		for cand in self.candidates:
			# dot product
			cand.harmony = sum(viol*weight for viol,weight in zip(cand.violations,w))
			# Assuming the candidate's violations and weights are the correct sign, this'll be a negative number
			#IF we're taking the lexical constraints into account:
			if not suppressLexC:
				for i in self.lexC: # now add in stuff for the lexical constraint
					cand.harmony += i[1]*(0 if cand.surfaceForm==i[0] else -1)
			try:
				self.probDenom += pow(math.e,cand.harmony)
			except OverflowError:
				print "Something's wrong with the exponentiation!  Check to make sure decay is working properly.  Python's patience with giant exponents only stretches so far..."
				print self.ur
				print cand
				print cand.harmony
				print w
				print self.lexC		
	def predictProbs(self,w, t=None, decayRate=None, decayType=None,suppressLexC=False):
		#print 'predictProbs: ', decayRate
		self.calculateHarmony(w, t, decayRate, decayType,suppressLexC)
		self.predProbsList=[]
		for cand in self.candidates:
			try:
				cand.predictedProb = pow(math.e,cand.harmony)/self.probDenom
			except OverflowError:
				print "Something's wrong with the exponentiation!  Check to make sure decay is working properly.  Python's patience with giant exponents only stretches so far..."
				print self.ur
				print cand.c
				print cand.harmony
				print w
				print self.lexC

			self.predProbsList.append(cand.predictedProb)
	def getPredWinner(self,theory):
		if theory =='HG':
			pass # Figure out which candidate has the best harmony
		if theory =='MaxEnt': # sample from distribution
			winCandidate = self.candidates[np.random.choice(range(0,len(self.candidates)),1,p=self.predProbsList)[0]]
			#winCandidate = self.candidates[random_distr(zip(range(0,len(self.candidates)),self.predProbsList))]
			#winCandidate = self.candidates[rv_discrete(values=(range(0,len(self.candidates)),self.predProbsList)).rvs(size=1)] 
			winner = winCandidate.surfaceForm
		return winner, winCandidate
	def getObsWinner(self,theory):
		if theory =='HG':
			pass # Figure out which candidate has the best harmony
		if theory =='MaxEnt': # sample from distribution
			# Get candidate list
			# Get probability list
			winCandidate = self.candidates[np.random.choice(range(0,len(self.candidates)),1,p=self.obsProbsList)[0]]
			#winCandidate = self.candidates[random_distr(zip(range(0,len(self.candidates)),self.obsProbsList) )]
			#winCandidate = self.candidates[rv_discrete(values=(range(0,len(self.candidates)),self.obsProbsList)).rvs(size=1)] # rv_discrete(values=(l,[0,0,.8,0.05,0.05,0,0.1])).rvs(size=100)
			winner = winCandidate.surfaceForm
		return winner, winCandidate
	def compareObsPred(self,theory,w, t=None, decayRate=None, decayType=None,strategy = 'sample'):
		if theory =='batchGD':
			pass
			# Have to do some kind of vector comparison for batch gradient descent
		if theory =='MaxEnt':
			self.predictProbs(w, t, decayRate, decayType)
			pred, predCandidate=self.getPredWinner(theory) # Sample from predicted distribution
			obs, obsCandidate=self.getObsWinner(theory) # Sample from observed distribution

			if strategy == 'sample':
				error = (0 if obs==pred else 1)
			elif strategy == 'HDI':
				error = (0 if pred in self.probableCandidates else 1)

		return error, obsCandidate, predCandidate

class Tableaux:
	def __init__(self,theory='MaxEnt'):
		self.urList = []
		self.urIDlist = []
		self.urProbsList = []
		self.theory = theory
		self.constraints = []
		self.w = []
		self.t = 0
		self.initializeWeights()
		self.lexicon = [[],[]] #This will get populated during learning - the first list by URs (UR objects), and the second list by integer number of times they've been seen

		# Starting probability of inducing a lexically specific constraint is 1
		# If lexCSample is True, this will slowly get updated based on how constraints are being acquired in the dataset
		self.pLexC = 1
		#self.lexCstartW = 10
	def addUR(self,ur):
		self.urList.append(ur)
		self.urIDlist.append(ur.ur)
		self.urProbsList.append(ur.prob)
	def initializeWeights(self,w=None):
		if w is None:
			self.w = [0]*len(self.constraints)
		else:
			if len(w)==len(self.constraints):
				self.w=w
			else:
				print "WARNING: you tried to initialize weights with a list that's not the same length as your list of constraints.  This probably didn't work so well."
	def getReady(self):
		self.URsampleVector = [i/sum(self.urProbsList) for i in self.urProbsList]
		if hasNumpy:
			self.URsampleVector=np.array(self.URsampleVector)
	def sample(self):
		# if URsampleVector is not defined, call getReady()
		try:
			if hasNumpy:
				th=np.random.choice(range(0,len(self.URsampleVector)),1,p=self.URsampleVector)
				# for if you have scipy, but to access to the random.choice() function for some reason (as I did)
				# th = rv_discrete(values=(range(0,len(sampleVector)),sampleVector)).rvs(size=1)
			else:
				th=random_distr(zip(range(0,len(sampleVector)),sampleVector))
		except IndexError:
			print("Ack!  Something's wrong.  Maybe double check that your Tableaux really has any URs in it?  You can do this from inside python by printing 'mytableaux.urIDlist' where 'mytableaux' is your Tableaux name of course")

		theUR = self.urList[th]
		return theUR
	def resetTime(self):
		''' Resets time.  Sets Tableaux.t to 0, and also goes through each UR and resets its last seen time (UR.lastSeen) to 0'''
		self.t = 0
		for x in self.urList:
			x.lastSeen = 0
	def resetLexC(self):
		'''Reset all the lexically specific constraints to 0 - that is, delete them all'''
		for i in self.urList:
			i.lexC = []
	def downSample(self,size,disregardFrequency=False): 
		newtabs=Tableaux()
		possibleURs=self.urList
		if disregardFrequency:
			probs=[1/sum(possibleURs) for i in possibleURs]
		else:
			probs=self.URsampleVector
		if hasNumpy:
			sample=np.random.choice(range(0,len(probs)),size,p=probs,replace=False)
			for i in sample:
				newtabs.addUR(possibleURs[i])
		else:
			# my kludgey way of getting sampling without replacement
			print "WARNING slooooooooooooooow.  Maybe you should try installing numpy, or getting it to work properly?"
			sample=[]
			while size:
				th=random_distr(possibleURs,probs)
				newtabs.addUR(possibleURs.pop(th))
				probs.pop(th)
				size-=1	

		#Add constraints etc to newtabs
		newtabs.theory=self.theory
		newtabs.constraints=self.constraints
		newtabs.w=self.w
		newtabs.t=self.t
		newtabs.initializeWeights()
		newtabs.lexicon = [[],[]]
		newtabs.pLexC=self.pLexC
		#newtabs.lexCstartW=self.lexCstartW
		return newtabs

	def update(self,theory,learnRate,lexCstartW, lexLearnRate, lexCSample=False, lexCSampleSize=10, decayRate=None, decayType=None,haveLexC=False,comparisonStrategy='sample',urToUse=None):
		# If the caller didn't provide a ur
		if urToUse==None:
			# Sample an input form
			theUR = self.sample()
		else:
			theUR = urToUse
		theUR.lastSeen=self.t
		#print theUR.ur
		if theUR in self.lexicon[0]:
			# If you've seen this form before
			# Up its count
			self.lexicon[1][self.lexicon[0].index(theUR)] +=1
		else:
			self.lexicon[0].append(theUR)
			self.lexicon[1].append(1)
		#print self.lexicon


		e, o, p = theUR.compareObsPred(theory,self.w, self.t, decayRate, decayType, comparisonStrategy)
		if e: # on error
			# update general constraints with perceptron update
			newW = perceptronUpdate(p.violations, o.violations, self.w, learnRate)
			#print newW
			self.w = newW
			self.w=[i if i>0 else 0 for i in self.w]

			if haveLexC:
				# update lexC's
				existsLexC = False
				for c in theUR.lexC:
					# Check if lex C favors the observed form
					if c[0]==o.surfaceForm:
						c[1]+=lexLearnRate # increment lexC by learning rate
						if c[1]>=700:  #Impose hard upper limit on constraint values
							c[1]=700
						existsLexC = True
					else:
						c[1]-=lexLearnRate #else, decrement by learning rate
				if not existsLexC: # if there's no lexical constraint for the observed output, make one
					if lexCSample and len(self.lexicon[0])>lexCSampleSize: 
					# If we're employing a sampling approach
					# And, if we've got a big enough sample to work with
						# Go on a random walk through the known lexicon
						# Grab lexCSampleSize forms, and sample with replacement
						if hasNumpy:
							sampleVec=np.array([float(i)/sum(self.lexicon[1]) for i in self.lexicon[1]])
							th=np.random.choice(range(0,len(sampleVec)),lexCSampleSize,p=sampleVec)
						else:
							sampleVec=[float(i)/sum(self.lexicon[1]) for i in self.lexicon[1]]
							th=[]
							for k in range(0,lexCSampleSize):
								th.append(random_distr(range(0,len(sampleVec)),sampleVec))
						# th is a list of indices of the sampled UR's
						# Have to meet a threshold to sample - need a lexicon of size greater than the sample size
						for i in th:
							#Check whether there are lexC's in each UR
							if len(self.lexicon[0][i].lexC)>0:
								# If you find one, go ahead and induce the new lexical constraint
								theUR.lexC.append([o.surfaceForm,lexCstartW])
								break

						###- empiricalpLexC = nLexC/lexCSampleSize

						# k. now update the pLexC globally
						# Update is currently kindasorta bayesian.  Average 'prior' and probability based on the random walk
						#print self.pLexC
						#print empiricalpLexC
						###- if len(self.lexicon[0])>lexCSampleSize:
							# Only update if the lexicon is at least as big as the sample size
							###- self.pLexC = (49*float(self.pLexC) + float(empiricalpLexC))/50.0

						# NOW decide whether to induce a lexC on this iteration
						###- r = random.random()
						###- if r < self.pLexC:
							###- theUR.lexC.append([o.surfaceForm,lexCstartW])



					else: # if we're not sampling, just always make a lexC
						theUR.lexC.append([o.surfaceForm,lexCstartW])
		self.t+=1
		theUR.lastSeen = self.t
		theUR.nSeen += 1
		return theUR, e #return the UR and whether or not there was an error

	def epoch(self,theory,iterations,learnRate,lexCstartW, lexLearnRate, lexCSample=True, lexCSampleSize=10, decayRate=None, decayType=None,haveLexC=False,comparisonStrategy='sample'):
		
		#Sample all the UR's for this epoch
		sampleVector = [i/sum(self.urProbsList) for i in self.urProbsList]
		urIndices=np.random.choice(range(0,len(sampleVector)),iterations,p=sampleVector)

		errRate = 0
		for i in range(0,int(iterations)):
			UR, err = self.update(theory,learnRate,lexCstartW,lexLearnRate,lexCSample, lexCSampleSize,decayRate,decayType,haveLexC,comparisonStrategy,urToUse=self.urList[urIndices[i]])
			errRate += err
		errRate = float(errRate)/float(iterations)


		# Lexically specific constraints and their weights
		lexCs  = []
		lexCws = []
		for i in self.urList:
			#Decay all the lexical C's at the end of each epoch
			#try:
			i.decayLexC(self.t,decayRate,decayType)
			#except:
			#	print "ack"
			#	for i in self.urList:
			#		print i.lexC


			for j in i.lexC:
				lexCs.append((i.ur,j[0]))
				lexCws.append(j[1])

		# calculate the probability of inducing a lexically specific constraint
		# that rv.discrete guy samples with replacement, so it's simple.
		# P= (nlexC/nUR) * sampleSize
		nlexC = 0
		for i in self.urList:
			if len(i.lexC)>0:
				nlexC += 1
		print nlexC
		# P(at least once) = 1-P(never)
		# P(never) = (1-nlexC/nURs)^lexCSampleSize
		self.pLexC = 1-pow((1-(float(nlexC)/float(len(self.urList)))),lexCSampleSize)
		print self.pLexC
		print len(self.urList)
		sse = self.SSE()
		return errRate, sse, lexCs, lexCws, self.pLexC

	def learn(self, iterations, nEpochs,learnRate,lexCstartW=0,lexLearnRate=0,lexCSample=False,lexCSampleSize=10, decayRate=None, decayType='static',theory='MaxEnt',haveLexC=False,reset=True,comparisonStrategy='sample'):

		'''Main wrapper for learning. Calls the epoch function nEpochs times, and each epoch is iterations number of learning steps.  learnRate can be a real number, or a 'schedule' of two numbers - a starting rate and an ending rate.'''
		# First, grab all the inputs, so the function call can be saved in the results object

		functionCall={}
		k=locals().keys()
		for i in k:
			functionCall[i]=locals()[i]

		#TODO: there's a bug right now that if youjust write '0' for the learning rate it's all 'omg int not float omg'.  Just add a little conversion to float at some point.  This might be tricky if we want to maintain the ability to take a plasticity schedule as input.  Or maybe not.  I just don't feel like thinking about it right now, so.


		if type(learnRate) is float: # If learnRate is a real number
			localLearnRate = learnRate
			plasticity = 0
		else: # If it's some kind of schedule
			localLearnRate = learnRate[0]
			plasticity = (learnRate[0]-learnRate[1])/nEpochs # learnRate decrement per epoch
			# TODO check length of the schedule; allow either a tuple or a list
			# TODO check that second number is less than first
			# Allow for non-linear schedules?  i think this would be pretty hard, actually.
		start = timeit.default_timer()
		if reset:
			self.initializeWeights()
			self.resetTime()
			self.resetLexC()
			self.lexicon = [[],[]]
			self.pLexC = 1
			#Find the most probable observed candidates for each UR
			for u in self.urList:
				u.getProbableCandidates()
			print "learning..."
		else:
			print "continuing to learn..."
		if theory == 'MaxEnt':
			ep = []
			err = []
			weights = []
			sse = []
			pLexCs = []

			lexiWeights = []
			lexiNames = []

			firstEpoch=timeit.default_timer()
			for n in range(0,nEpochs):
				print "Epoch ",n
				localLearnRate -=plasticity #Update the learning rate with plasticity
				w,x,y,z,p = self.epoch(theory,iterations,localLearnRate,lexCstartW,lexLearnRate, lexCSample, lexCSampleSize, decayRate,decayType,haveLexC,comparisonStrategy)
				ep.append(n)
				err.append(w)
				sse.append(x)
				pLexCs.append(p)
				weights.append(self.w)

				# Arrange lexical constraint information
				# y is list of lexical constraints (tuples)
				# z is list of weights
				for c in y:
					if c in lexiNames:
						# find the index of c in both lists
						cIndex = lexiNames.index(c)
						# find index of c in z, the list of weights from the epoch
						cWindex = y.index(c)
						# Add the appropriate weight and time to lexiWeights
						lexiWeights[cIndex][0].append(n) #n=epoch
						lexiWeights[cIndex][1].append(z[cWindex])
					else: # Add the constraint to lexiNames and lexiWeights
						lexiNames.append(c)
						lexiWeights.append([[n],[z[y.index(c)]]])

				if n==0:
					endFirstEpoch=timeit.default_timer()
					oneEpochtime=endFirstEpoch-firstEpoch
					predictedRuntime=oneEpochtime*nEpochs
					print "Predicted runtime at this point: " + str(predictedRuntime/60.0) + "minutes"



		stop = timeit.default_timer()
		runtime=stop-start

		print type(ep)
		print type(weights)
		print type(sse)
		print type(err)
		print type(runtime)
		print type(predictedRuntime)
		print type(functionCall)
		print type(self.constraints)
		print type([lexiNames,lexiWeights])
		print type(pLexCs)
		print type(self.lexicon)


		results = Results(ep,weights,sse,err,runtime,predictedRuntime,functionCall,self.constraints,[lexiNames,lexiWeights],pLexCs,self.lexicon)

		return results

	def calcLikelihood(self):
		logLikelihood=0
		for i in self.urList:
			i.calculateHarmony(self.w)
			i.predictProbs(self.w)
			for j in i.candidates:
				pass
				# Log j.observedProb * j.predictedProb ?
		print "This function doesn't work yet.  Please check back later"
		return logLikelihood

	def SSE(self):
		sse=0
		for i in self.urList:
			i.calculateHarmony(self.w)
			i.predictProbs(self.w)
			for j in i.candidates:
				sse+=pow(j.observedProb-j.predictedProb,2)
		return sse

class Society:
	def __init__(self,generations,startTableaux,outputName):
		self.generations = 0
		self.startTableaux = startTableaux
		self.currentTableaux = startTableaux
		self.resultses = [] #Fill this with results objects from each iteration
		self.outputName=outputName

	def iterate(self,nGenerations,iterations, nEpochs,learnRate,lexCstartW=0,lexLearnRate=0,lexCSample=False,lexCSampleSize=10, decayRate=None, decayType='static',theory='MaxEnt',haveLexC=False,reset=True,comparisonStrategy='sample',updateURprobs = True):

		for i in range(1,nGenerations):
			#Update current tableaux based on last set of results
			if i > 1:
				self.updateTableaux(URprobs = updateURprobs)

			#Learn on new tableaux
			results = self.currentTableaux.learn(iterations, nEpochs,learnRate,lexCstartW,lexLearnRate,lexCSample,lexCSampleSize, decayRate, decayType,theory,haveLexC,reset,comparisonStrategy)
			results.save(self.outputName+"-"+str(i))

			#Append results
			self.resultses.append(results)

	def updateTableaux(self,candidateProbs = True, URprobs = True):
		'''Updates the current tableaux set for learning for the next generation.
		Note that this will overwrite currentTableaux - the only tableaux set that is saved forever is the initial tableaux set.

		candidateProbs: if True, the final predicted probs from the end-state of the last learning phase are used as the new training data
		
		URprobs: if True, the lexical frequencies of each UR are updated, to the number of times each UR was observed by the last generation.  NOTE: with this, especially with small numbers of learning iterations, words could dissappear from the vocabulary.

		Interesting future idea NOT IMPLEMENTED IN ANY WAY: create mechanism for introduction of new words, either by mutating existing words, or by inventing phonotactically plausible new strings '''

		self.newTableaux = Tableaux(theory='MaxEnt')
		self.newTableaux.constraints = self.currentTableaux.constraints
		# Update URs, candidates, probabilities
		if not candidateProbs:  #I'm not sure what the option to make this false will actually be useful for at this point.  
			print "What are you even doing?"

		for i in self.currentTableaux.urList:
			if URprobs:
				newUR = UR(i.ur,i.nSeen)
			else:
				newUR = UR(i.ur,i.prob)

			for j in i.candidates:
				newUR.addCandidate(candidate(j.c,j.violations,j.predictedProb))

			self.newTableaux.addUR(newUR)

		self.currentTableaux = self.newTableaux

class Results:
	def __init__(self,t,w,sse,err,runtime,predruntime,functionCall,Cnames,lexCinfo,pLexCs,lexicon):
		self.t = t
		self.w = w
		self.sse = sse
		self.err = err
		self.runtime = runtime
		self.predruntime=predruntime
		self.functionCall=functionCall
		self.Cnames=Cnames
		self.lexCinfo = lexCinfo
		self.pLexCs = pLexCs
		self.lexicon = lexicon

		# First, group the lexical constraints into classes
		lexCclasses = [[],[],[]]
		for c in self.lexCinfo[0]:
			if c[1] in lexCclasses[0]:
				# extend that output's time vector
				lexCclasses[1][lexCclasses[0].index(c[1])][0].extend(self.lexCinfo[1][self.lexCinfo[0].index(c)][0])
				# extend that output's weight vector
				lexCclasses[1][lexCclasses[0].index(c[1])][1].extend(self.lexCinfo[1][self.lexCinfo[0].index(c)][1])
				# add this input to the list of inputs for that class
				lexCclasses[2][lexCclasses[0].index(c[1])].append(c[0])
			else:
				lexCclasses[0].append(c[1])
				lexCclasses[1].append(self.lexCinfo[1][self.lexCinfo[0].index(c)])
				lexCclasses[2].append([c[0]])
			#print lexCclasses
		self.lexCclasses = lexCclasses

		lexCMeans = [lexCclasses[0],[],lexCclasses[2]]
		for i in lexCclasses[1]:
			lexCmean = []
			# Start by making a structure like [t][w] where each w is a mean of all the weights at the indexed time.
			t = range(1,max(i[0])) #i[0] is a list of timestamps, i[1] corresponding weights
			lexCmean.append(t)
			lexCmean.append([]) # this will hold the sets of weights at each timestamp
			lexCmean.append([]) # and this will hold the numbers of weights at each timestamp
			for ti in t:
				# Grab all the indices in the list of timestamps for each time
				# e.g. all the indices of the timestamp '32'  (when ti is 32)
				indices = [k for k,x in enumerate(i[0]) if x==ti]
				wVect=[]
				for index in indices:
					wVect.append(i[1][index])
				wMean=[sum(wVect)/len(wVect) if len(wVect)>0 else 0] #Calculate and append the mean of the set of weights observed at time ti
				lexCmean[1].append(wMean)
				lexCmean[2].append(len(wVect))
			lexCMeans[1].append(lexCmean) # Add this list of t,w for this class of items.
		self.lexCMeans = lexCMeans


	def save(self,filename,folder=""):
		#try:
		with open(folder+filename+"_metadata",'w') as f:
			f.write("Function Call: " )
			for param in self.functionCall:
				f.write("\n")
				f.write(param+": "+str(self.functionCall[param]))
			f.write("\n")
			f.write("Actual runtime: " + str(float(self.runtime)/float(60))+" minutes")
			f.write("\n")
			f.write("Predicted runtime: "+ str(float(self.predruntime)/float(60))+" minutes")
			f.write("\n")
			f.write("Probability of inducing a lexC: "+str(self.pLexCs))
			f.write("\n")
			f.write("Note: the above probability may be meaningless unless you're using the probabilistic inducing functionality")
		#except:
		#	print "oops!  There's some kinda problem with printing the metadata"
		with open(filename+"_weights",'w') as f:
			f.write("\t".join(["t"]+self.Cnames))
			f.write("\n")
			for i in range(len(self.w)):
				f.write("\t".join([str(self.t[i])]+[str(j) for j in self.w[i]]))
				f.write("\n")
		with open(filename+"_SSE",'w') as f:
			f.write("\t".join(["t","SSE"]))
			f.write("\n")
			for i in range(len(self.sse)):
				f.write("\t".join([str(self.t[i])]+[str(self.sse[i])]))
				f.write("\n")
		with open(filename+"_err",'w') as f:
			f.write("\t".join(["t","err"]))
			f.write("\n")
			for i in range(len(self.err)):
				f.write("\t".join([str(self.t[i])]+[str(self.err[i])]))
				f.write("\n")
		with open(filename+"_lexicon",'w') as f:
			f.write("\t".join(["word","count"]))
			f.write("\n")
			for i in self.lexicon[0]:
				f.write("\t".join([i.ur]+[str(self.lexicon[1][self.lexicon[0].index(i)])]))
				f.write("\n")

		#Groundwork for printing lexical constraint weights over time
		structuredLexiWeights={}
		for i in range(len(self.lexCinfo[0])): #iterate through the indices of the constraints (lexCinfo[0]) and their time vector/weight vector pairs (lexCinfo[1])
			structuredLexiWeights[self.lexCinfo[0][i]]=[] # add the current constraint name, in the form of a tuple of (input, prescribed output), as a new key in this new dictionary thing
			for j in self.t: #Iterate through timesteps - these are epochs of the learning process
				if j in self.lexCinfo[1][i][0]: # if there's an entry for that timestep in the weight vector of the current constraint...
					structuredLexiWeights[self.lexCinfo[0][i]].append(self.lexCinfo[1][i][1][self.lexCinfo[1][i][0].index(j)]) # add that constraint's weight at the appropriate time index
				else:
					structuredLexiWeights[self.lexCinfo[0][i]].append(0) #otherwise put a zero in that spot, as a placeholder.

		# k now we can try to print to a file.  there will be t columns in this crazy file.  More reason to be thrifty about your sampling of the process, and keep nEpochs low
		with open(filename+"_lexCinfo",'w') as f:
			for i in structuredLexiWeights:
				f.write("_".join(i))
				f.write("\t")
				f.write("\t".join([str(k) for k in structuredLexiWeights[i]]))
				f.write("\n")

		with open(filename+"_finalState",'w') as f:
			f.write("\t".join(["UR","Candidate","ObservedProb","PredictedProb","PredictedProbNoLexC","observedFreq","trainedFrequency","lexCW"]))
			f.write("\n")
			for i in range(len(self.lexicon[0])):
				#Predict probabilities for each candidate without their lexical constraints
				self.lexicon[0][i].predictProbs(self.w[-1],suppressLexC=True)
				noLexPreds=[]
				for c in self.lexicon[0][i].candidates:
					noLexPreds.append(c.PredictedProb)
				# Use the final weights to predict probabilities with lexical constraints
				self.lexicon[0][i].predictProbs(self.w[-1])
				for c in self.lexicon[0][i].candidates:
					lexCW = 0
					for lexicalConstraint in self.lexicon[0][i].lexC:
						if lexicalConstraint[0]==c.surfaceForm:
							lexCW = lexicalConstraint[1]
					line = "\t".join([self.lexicon[0][i].ur, c.c, str(c.observedProb), str(c.predictedProb),noLexPreds[self.lexicon[0][i].candidates.index(c)], str(self.lexicon[0][i].prob), str(self.lexicon[1][i]),str(lexCW)])
					f.write(line)
					f.write('\n')

	def plotSSE(self):
		try:
			fig = plt.figure()
			errorplot = fig.add_subplot(111)
			errorplot.plot(self.t,self.sse,color='red',lw=2)
			errorplot.set_xlabel('Epoch')
			errorplot.set_ylabel('SSE')
			plt.show(block=False)
		except:
			print "Oops!  It looks like you don't have matplotlib installed."

	def plotpLexCs(self):
		try:
			fig = plt.figure()
			errorplot = fig.add_subplot(111)
			errorplot.plot(self.t,self.pLexCs,color='red',lw=2)
			errorplot.set_xlabel('Epoch')
			errorplot.set_ylabel('pLexC')
			plt.show(block=False)
		except:
			print "Gak!  No matplotlib installed!  No plots for you."

	def plotW(self):
		try:
			#First, 'transpose' the weight vectors
			weightVect=[]
			for i in range(len(self.w[0])):
				wi = []
				for j in self.w:
					wi.append(j[i])
				weightVect.append(wi)
			fig = plt.figure()
			weightplot = fig.add_subplot(111)
			for i in range(len(weightVect)):
				weightplot.plot(self.t,weightVect[i],color=colors.cnames[colors.cnames.keys()[i]],lw=2,label=self.Cnames[i])
				print colors.cnames.keys()[i]
			weightplot.set_xlabel('Epoch')
			weightplot.set_ylabel('W')
			box = weightplot.get_position()
			weightplot.set_position([box.x0, box.y0, box.width * 0.8, box.height])
			weightplot.legend(bbox_to_anchor=(1,0.5), loc=2, borderaxespad=0.5)
			plt.show(block=False)
		except:
			print "You haven't installed matplotlib, but that's ok.  Just save using the handy save() function, and then use your favorite plotting software."

	def plotMeanLexW(self):
		try:
			# Now, plot the things
			fig = plt.figure()
			weightplot = fig.add_subplot(111)
			for i in range(len(self.lexCMeans[0])): #iterate over classes
				weightplot.plot(self.lexCMeans[1][i][0],self.lexCMeans[1][i][1],color=colors.cnames[colors.cnames.keys()[i]],label=self.lexCMeans[0][i])
			weightplot.set_xlabel('Epoch')
			weightplot.set_ylabel('W')
			box = weightplot.get_position()
			weightplot.set_position([box.x0, box.y0, box.width * 0.8, box.height])
			weightplot.legend(bbox_to_anchor=(1,1), loc=2, borderaxespad=0.5)
			plt.show(block=False)

			fig = plt.figure()
			weightplot = fig.add_subplot(111)
			for i in range(len(self.lexCMeans[0])): #iterate over classes
				weightplot.plot(self.lexCMeans[1][i][0],self.lexCMeans[1][i][2],color=colors.cnames[colors.cnames.keys()[i]],label=self.lexCMeans[0][i])
			weightplot.set_xlabel('Epoch')
			weightplot.set_ylabel('nW')
			box = weightplot.get_position()
			weightplot.set_position([box.x0, box.y0, box.width * 0.8, box.height])
			weightplot.legend(bbox_to_anchor=(1,1), loc=2, borderaxespad=0.5)
			plt.show(block=False)
		except:
			print "Yea, I can't find matplotlib.  No plots for you this time!"


def perceptronUpdate(error,target,weights,rate):
	"""error and target should be violation vectors, weights should be a vector of weights, and rate is the learning rate."""
	if len(error)!=len(target):
		sys.exit("It looks like you tried to update with two violation vectors that are different lengths.  Maybe check your input file?")
	if min(error+target)<0 and max(error+target)>0:
		sys.exit("It looks like your tableaux contain both positive and negative violations.  Check your input file.  If this isn't an error, consider using both positive and negative weights instead.")
	error=[abs(i) for i in error]
	target=[abs(i) for i in target]
	return [w+(e-t)*rate for w,e,t in zip(weights,error,target)]


def random_distr(l):
    r = random.uniform(0, 1)
    s = 0
    for item, prob in l:
        s += prob
        if s >= r:
            return item
    return item  # Might occur because of floating point inaccuracies

def readOTSoft(file):
	"""Reads in an OTSoft file type"""
	print 'reading tableaux...'
	tableaux = Tableaux()
	with open(file,"r") as f:
		lines=f.read().split('\n')
		# Check if the linebreak character is correct.  If it's not, the symptoms will be (a) only one line and (b) '\r' somewhere inside the line
		if bool(re.match('.*\r.*',lines[0])):
			if len(lines)==1:
				lines=lines[0].split('\r')
			else:
				print "EEK something is wrong with your linebreaks"

		lineno=0
		for line in lines:
			line=line.split('\t')
			#print line
			if lineno==0:
				firstLine = line
				#print firstLine
				if firstLine[1]=="":
					# OTSoft file? They have empty cells at the top
					pass
				if firstLine[0]=="input":
					# hgr file? They have the first three columns labeled
					inputType='hgr'
					#print inputType
					# Constraints are in the first line too, so grab those
					# Headers of these files:
					# input output (hidden) probability (tab.prob) Constraint1 Constraint2 ...
					offset = (3 if firstLine[2]=='hidden' else 2)
					hidden = (True if firstLine[2]=='hidden' else False)
					tokenFrequency = (True if firstLine[offset+1]=='tab.prob' else False)
					offset = offset + (2 if firstLine[offset+1]=='tab.prob' else 1)
					constraints=firstLine[offset:]
					#print offset
					#print tokenFrequency
					#print constraints
					tableaux.constraints = constraints
				else:
					print "I can't tell what type of file this is :("

			elif inputType=='hgr':
				ur = line[0]
				surfaceForm = (line[1] if hidden else None)
				c = (line[2] if hidden else line[1])
				observedProb = (line[3] if hidden else line[2])
				tokenProb = (line[offset-1] if tokenFrequency else 1)
				violations = line[offset:]
				print ur
				if ur not in tableaux.urIDlist:
					#print 'add'
					tableaux.addUR(UR(ur,tokenProb))
				for i in tableaux.urList:
					if i.ur==ur:
						i.addCandidate(candidate(c,violations,observedProb,surfaceForm))
						break
			lineno+=1

	return tableaux


