import sys
import math
import re
import timeit
import random
import argparse	 #TO DO: Make sensible command-line handling?

try:
	from scipy.stats import rv_discrete
	import numpy as np 
	hasNumpy=True
except:
	print("It looks like you don't have one or both of numpy and scipy installed.  You should probably just google how to install those on your machine.  What you really need here is the function numpy.random.choice().	For some reason, this function doesn't always work when you install numpy with pip, so if that happens to you, maybe try installing Anaconda or something similar.	You can still use the program, but it will be really slow, probably intolerably slow with larger datasets.")
	hasNumpy=False

try:
	import matplotlib.pyplot as plt
	import matplotlib.colors as colors
except:
	print("It looks like you don't have matplotlib installed.  That's ok, you can still run the program, but you won't be able to use the plotting functions in the Results object.")

# Questions.  Do you need numpy to have scipy?	Should this test be different?	Should I also test specifically for random.choice?

# could remove plotting functionality?  Or split it off into a separate file?



class candidate: # Includes infrastructure for future ability to do hidden structure learning
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
			print("It looks like your candidates' probabilities can't all be converted to floats.  Check if there's some text in that column")
		self.harmony = 0 # Not specified at initialization.	 This will be filled out in learning
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
				print("It looks like some of your violations can't be interpreted as numbers.  Check whether there's some text in your violation profiles")
			if self.violations[i]>0:
				self.violations[i] = - self.violations[i]

class LexEntry: 
	def __init__(self,inpt,prob=1,pfc=None):
		self.inpt = inpt # text specifying the input.  This is more or less arbitrary.  When doing a non-Correspondence-Free-MaxEnt model, this is just the UR.  It's always the value from the 'input' column of the data file
		self.prob = prob # Lexical frequency of this lexical entry, specified in tab.prob of the input file.  if left unspecified, all will be set to 1
		# Check if the probability can be sensibly converted to a float
		try:
			self.prob = float(prob)
		except ValueError:
			print("It looks like your token probabilities can't all be converted to floats.	 Check if there's some text in that column")
		self.candidates = [] # To be filled during reading in of data - list of candidate objects that are candidate outputs affiliated with this lexical entry
		if pfc is None:
			self.pfc = [] # This can later be filled with tuples of PFCs that go with this lexical entry.  They can also be specified in the call of the LexEntry class, if there's ever a reason to do that.
		# The tuples will be (PFC name, weight)
		# And PFCs will be named according to the output they demand
		# Ex: a PFC demanding penultimate stress on 'banana' will be named 'banana_010'
		# Once hidden structure is incorporated, it will be important that this output be a surface form, not an actual candidate with hidden structure
		self.probDenom = 1 # denominator for calculating MaxEnt prob of each candidate
						   # this is the sum of each exponentiated harmony score
		self.predProbsList = [] # list of each candidate's predicted probability; used in sampling
								# gets filled (and updated) during learning
		self.obsProbsList = [] 	# list of each candidate's observed probability; used in sampling
							    # gets filled from the 'probability' column of the data file, during read-in
		self.lastSeen = 0 # Keep track of what learning iteration (timestep) the learner last saw this word
		self.nSeen = 0 	# Keep track of how many times the lexical item has been seen during learning
		self.probableCandidates = [] #List of all candidates with observed probabilities greater than some threshold, set to 5%, or 0.05 by default.  Think of these as 'contenders'

	def addCandidate(self,cand):
		self.candidates.append(cand)
		self.obsProbsList.append(cand.observedProb)
	def getProbableCandidates(self):
		#This should get run after all the candidates are added to the LexEntry
		#Returns a list of surface forms for all candidates with an observed probability greater than 5%
		for c in self.candidates:
			if c.observedProb >= 0.05: #<--- or you could decide you want this to be different
				self.probableCandidates.append(c.surfaceForm)
	def decayPFC(self,t,decayRate,decayType='static'):
		'''Decay all the PFCs for this LexEntry.  Need to know the decay rate set for the current learning run, and also need to know t, the current learning iteration (timestep). '''
		if len(self.pfc)>0:
			t = float(t)
			for i in range(0,len(self.pfc)): # run through all the PFCs for this LexEntry
											 # TODO make this faster?  any way to convert this into some kind of matrix math?
				if decayType=='static':
					# Each PFC decays by a set amount per timestep
					self.pfc[i][1]-=(t-self.lastSeen)*decayRate

				if decayType=='linear':
					# Each PFC decays according to its current weight: Higher weights decay more, linearly
					W=self.pfc[i][1]
					for timestep in range(self.lastSeen,t):
						W=W-decayRate*W #execute decay for each timestep since lastSeen
					self.pfc[i][1]=W

				if decayType=='nonlinear':
					# Each PFC decays according to the square of its weight
					W=self.pfc[i][1]
					for timestep in range(self.lastSeen,t):
						W=W-decayRate*pow(W,2) #execute decay for each timestep since lastSeen
					self.pfc[i][1]=W
						
				if decayType=='root':
					# Decay square roots, rather than actual weights
					# TODO This is buggy - getting bizarre decay trajectories that I do not understand.
					exponent = 2
					W=self.pfc[i][1]
					if W>=0:
						self.pfc[i][1]=pow(pow(W,1/exponent)-(t-self.lastSeen)*decayRate,exponent)
						# 			       root of the weight  decay that root statically  unroot

			for i in range(0,len(self.pfc)):
				if self.pfc[i][1]<=0.0001: #If it's gotten as low as zero remove the constraint
					self.pfc.remove(self.pfc[i])  #remove() is a function with the list class, like append
			
	def checkViolationLength(self): 
		'''check if all the violation vectors are the same length'''
		l = len(self.candidates[0])
		for i in range(0,len(self.candidates)):
			if len(i.violations) != l:
				print("Not all violation vectors are the same length!\n", "Candidate ",i.c, "has ", len(i.violations), ", while the previous candidate has ", l, " violations.")
				return 0
			else:
				return 1

	def calculateHarmony(self, w, t=None, decayRate=None, decayType='static', suppressPFC=False): 
		'''Takes a vector of weights, equal in length to the violation vectors of the candidates.  Populates the harmony parameter of each candidate based on the weight vector w'''
		# Requires t, decayRate, and decayType if decay is active in learning.  First step in calculating harmony is to make sure you do it with updated PFC weights
		# suppressPFC is used in case you would normally calculate harmony with PFCs but you want to ignore the PFC weights just this time (like, to predict the harmonies if this LexEntry were really a nonword)

		self.probDenom=0 # Reset probDenom.  Even though this function is dedicated to calculating the harmony, it's still useful to take this opportunity to calculate the MaxEnt denominator.  If you're managing to use this program to do Noisy HG or something, this will be meaningless.
		if (decayRate is not None) and (t is None):
			print("You have to pass a time parameter to the calculateHarmony function if you're using a decay rate.  Calculating harmony without PFCs.")
			suppressPFC = True
			decayRate = None
		if decayRate is not None:
			self.decayPFC(t,decayRate,decayType) 
		for cand in self.candidates:
			# Calculate that good 'ole dot product!
			cand.harmony = sum(viol*weight for viol,weight in zip(cand.violations,w))
			# Assuming the candidate's violations and weights are the correct sign, this dot product will be a negative number
			# Now add in the opinions of the PFCs
			if not suppressPFC:
				for i in self.pfc:
					if i[1]>=700: # Impose a hard upper limit on PFC weights
						i[1]=700  # otherwise the calculation of probability, in particular probDenom, will fail
					cand.harmony += i[1]*(0 if cand.surfaceForm==i[0] else -1)
					# Assume always a single violation from each PFC for forms that don't match it
			try:
				self.probDenom += pow(math.e,cand.harmony)
			except OverflowError:
				print("Something's wrong with the exponentiation in calculating the MaxEnt denominator.  Check to make sure decay is working properly.  Python's patience with giant exponents only stretches so far...")
				print(self.inpt)
				print(cand)
				print(cand.harmony)
				print(w)
				print(self.pfc)		


	def predictProbs(self,w, t=None, decayRate=None, decayType=None,suppressPFC=False):
		'''Convert harmony scores, and the MaxEnt denominator, to predicted probabilities for each candidate output in the LexEntry '''
		self.calculateHarmony(w, t, decayRate, decayType,suppressPFC) # start by getting those harmonies
		self.predProbsList=[]
		for cand in self.candidates:
			try:
				cand.predictedProb = pow(math.e,cand.harmony)/self.probDenom
			except OverflowError:
				print("Something's wrong with the exponentiation!  Check to make sure decay is working properly.  Python's patience with giant exponents only stretches so far...")
				print(self.ur)
				print(cand.c)
				print(cand.harmony)
				print(w)
				print(self.pfc)
			except ZeroDivisionError:
				print("Somehow your MaxEnt denominator is zero!  Can't calculate probabilities this way")
				print(self.ur)
				print(cand.c)
				print(cand.harmony)
				print(w)
				print(self.pfc)
				print(self.probDenom)

			self.predProbsList.append(cand.predictedProb)

	def getPredWinner(self,theory):
		if theory =='HG':
			pass # Figure out which candidate has the best harmony
		if theory =='MaxEnt': # sample from distribution
			winCandidate = self.candidates[np.random.choice(range(0,len(self.candidates)),1,p=self.predProbsList)[0]]
			# TODO add in slower sampling option for ppl who can't manage to get numpy.random.choice
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
			# TODO: check that probableCandidates is always sensible.  Should getProbableCandidates be run here?
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
		self.lexList = [] # List of LexEntry objects
		self.lexIDlist = [] # List of actual, text, inputs, from that 'input' column in the data file
		self.lexProbsList = [] # List of token frequencies of lexical items (tab.prob)
		self.theory = theory # Right now only MaxEnt is implemented.  Could also do HG with some ease
		self.constraints = [] # Read these off the top of the input data file.
		self.w = [] # Weights, starts empty
		self.initializeWeights() # Initialize weights when Tableaux object is created (to zero, or some other vector depending on how you call this function.)
		self.t = 0	# For learning, t starts at 0
		self.lexicon = [[],[]] # This will get populated during learning - the first list by LexEntry objects, and the second list by integer number of times they've been seen

		self.pPFC = 1  # Starting probability of inducing a lexically specific constraint is 1
		# If PFCSample is True, this will slowly get updated based on how constraints are being acquired in the dataset

	def addLexEntry(self,LexEntry):
		'''Function for adding a lexical entry to the Tableaux '''
		self.lexList.append(LexEntry)
		self.lexIDlist.append(LexEntry.inpt)
		self.lexProbsList.append(LexEntry.prob)

	def initializeWeights(self,w=None):
		'''Function to initialize the weights - can take an argument, which is hopefully the same length as the number of constraints in the Tableaux object.  If it doesn't get that argument, it will initialize them to zero. '''
		# TODO (low priority atm) Add functionality to initialize weights to random values
		if w is None:
			self.w = [0]*len(self.constraints)
		else:
			if len(w)==len(self.constraints):
				self.w=w
			else:
				print("WARNING: you tried to initialize weights with a list that's not the same length as your list of constraints.	 This probably didn't work so well.")
				# This will print a warning, but it won't actually stop you from initializing constraints badly?  Maybe this is a problem that should be fixed.

	def getReady(self):
		'''Prep for sampling by creating a vector of probabilities that sums to 1 '''
		self.LexSampleVector = [i/sum(self.lexProbsList) for i in self.lexProbsList]
		if hasNumpy:
			self.LexSampleVector=np.array(self.LexSampleVector)

	def sample(self):
		'''Sample a single lexical entry based on lexical frequency.  Return the LexEntry object that was sampled. '''
		# TODO (high priority) if LexSampleVector is not defined, call getReady()

		try:
			if hasNumpy:
				th=np.random.choice(range(0,len(self.LexSampleVector)),1,p=self.LexSampleVector)

				# for if you have scipy, but no access to the random.choice() function for some reason (as I did):
				# th = rv_discrete(values=(range(0,len(sampleVector)),sampleVector)).rvs(size=1)
			else:
				th=random_distr(zip(range(0,len(sampleVector)),sampleVector))
		except IndexError:
			print("Ack!	 Something's wrong.	 Maybe double check that your Tableaux really has any lexical entries in it?  You can do this from inside python by printing 'mytableaux.lexIDlist' where 'mytableaux' is your Tableaux name of course")

		theLexEntry = self.lexList[th]
		return theLexEntry

	def resetTime(self):
		''' Resets time.  Sets Tableaux.t to 0, and also goes through each LexEntry and resets its last seen time (LexEntry.lastSeen) to 0'''
		self.t = 0
		for x in self.lexList:
			x.lastSeen = 0

	def resetPFC(self):
		'''Reset all the PFCs to 0 - that is, delete them all'''
		for i in self.lexList:
			i.pfc = []

	def downSample(self,size,disregardFrequency=False): 
		'''Function for downsampling a Tableaux object - maybe you want to try learning on a random subset of whatever dataset you've got.  This could be useful if you want to test things, for example. 

		This returns a new Tableaux object, so should be called as something like
		newTableaux = downSample(oldTableaux,size)
		size: how many lexical entries do you wanna sample?
		disregardFrequency: defaults to False, but if true, will ignore the lexical frequency information in the Tableaux file that you're downsampling.  So all forms are equally likely to make it into the downsampled Tableaux.
		'''
		newtabs=Tableaux()
		possibleLexEntries=self.lexList
		if disregardFrequency:
			probs=[1/sum(possibleLexEntries) for i in possibleLexEntries]
		else:
			probs=self.LexSampleVector
		if hasNumpy:
			sample=np.random.choice(range(0,len(probs)),size,p=probs,replace=False)
			for i in sample:
				newtabs.addLexEntry(possibleLexEntries[i])
		else:
			# my kludgey way of getting sampling without replacement
			print("WARNING slooooooooooooooow.	Maybe you should try installing numpy, or getting it to work properly?")
			sample=[]
			while size:
				th=random_distr(possibleLexEntries,probs)
				newtabs.addLexEntry(possibleLexEntries.pop(th))
				probs.pop(th)
				size-=1 

		#Add constraints etc to newtabs
		newtabs.theory=self.theory
		newtabs.constraints=self.constraints
		newtabs.w=self.w
		newtabs.t=self.t
		newtabs.initializeWeights()
		newtabs.lexicon = [[],[]]
		newtabs.pPFC=self.pPFC
		#newtabs.PFCstartW=self.PFCstartW
		return newtabs

	def update(self,theory,learnRate,PFCstartW, pfcLearnRate, PFCSample=False, PFCSampleSize=10, decayRate=None, decayType=None, havePFC=False, comparisonStrategy='sample', LexEntryToUse=None):
		''' Update function: Conducts a single update of the grammar, with a single lexical item.
			Return the lexical entry that was used for the update, as well as whether or not the update yielded an error.'''
		# If the caller didn't provide a ur
		if LexEntryToUse==None:
			# Sample an input form
			theLexEntry = self.sample()
		else:
			theLexEntry = LexEntryToUse
		theLexEntry.lastSeen=self.t
		if theLexEntry in self.lexicon[0]:
			# If you've seen this form before
			# Up its count
			self.lexicon[1][self.lexicon[0].index(theLexEntry)] +=1
		else:
			self.lexicon[0].append(theLexEntry)
			self.lexicon[1].append(1)

		e, o, p = theLexEntry.compareObsPred(theory,self.w, self.t, decayRate, decayType, comparisonStrategy)  # Compare observed to predicted for the lexical entry: returns the candidate corresponding to the observed form (o), the candidate corresponding to the predicted form (p), and whether or not there's an error (e)
		if e: 
			# If there was an error
			# update general constraints with perceptron update
			newW = perceptronUpdate(p.violations, o.violations, self.w, learnRate)
			self.w = newW
			self.w=[i if i>0 else 0 for i in self.w]
			# If any weights have fallen below zero, set them to zero  (imposes hard limit at zero)

			if havePFC:  # If we're using PFC's in this learning run
						 # update PFC's
				existsPFC = False  # The variable that says whether the lexical entry we're updating on has a PFC associated with it.  Set to false by default - change if you find one
				for pfc in theLexEntry.pfc:
					# iterate through all PFCs affiliated with the current lexical entry
					# Check if each PFC favors the observed form
					if pfc[0]==o.surfaceForm:
						pfc[1]+=pfcLearnRate # Increment PFC by learning rate
						if pfc[1]>=700:  # Impose hard upper limit on PFC weights
							pfc[1]=700
						existsPFC = True
					else:
						pfc[1]-=pfcLearnRate #If the PFC favors the wrong form, decrement

				if not existsPFC: # if there's no PFC demanding the observed output, make one
					if PFCSample and len(self.lexicon[0])>PFCSampleSize: 
					# If we're employing a sampling approach
					# AND if the lexicon is big enough
						# Go on a random walk through the known lexicon
						# Grab PFCSampleSize forms, and sample with replacement
						if hasNumpy:
							sampleVec=np.array([float(i)/sum(self.lexicon[1]) for i in self.lexicon[1]])
							th=np.random.choice(range(0,len(sampleVec)),PFCSampleSize,p=sampleVec)
						else:
							sampleVec=[float(i)/sum(self.lexicon[1]) for i in self.lexicon[1]]
							th=[]
							for k in range(0,PFCSampleSize):
								th.append(random_distr(range(0,len(sampleVec)),sampleVec))
						# th is a list of indices of the sampled LexEntries
						# They index into the Tableaux object's lexicon list
						for i in th:
							#Check whether there are PFCs in each lexical entry
							if len(self.lexicon[0][i].pfc)>0:
								# If you find one, go ahead and induce the new lexical constraint
								theLexEntry.pfc.append([o.surfaceForm,float(PFCstartW)])
								break

					else: # if we're not sampling, just always make a PFC
						theLexEntry.pfc.append([o.surfaceForm,PFCstartW])
		self.t+=1
		theLexEntry.lastSeen = self.t
		theLexEntry.nSeen += 1
		return theLexEntry, e #return the lexical entry and whether or not there was an error

	def sampleLexEntry(self, totalN):
		'''Sample totalN lexical entries from the Tableaux '''
		sampleVector = [i/sum(self.lexProbsList) for i in self.lexProbsList]
		lexIndices=np.random.choice(range(0,len(sampleVector)),totalN,p=sampleVector)
		return lexIndices
		
	def epoch(self,theory,lexIndices,iterations,learnRate,PFCstartW, pfcLearnRate, PFCSample=True, PFCSampleSize=10, decayRate=None, decayType=None,havePFC=False,comparisonStrategy='sample',silent=False):
		'''Function to do a single epoch of learning.  This will execute 'iterations' runs of the update() function.
		epoch() is an intermediate function between iterate() and learn(), designed to allow certain information about the learning run, like the constraint weights, to be sampled every epoch instead of every single iteration. 
	
		Updates the whole Tableaux file, so what it actually returns is:
		error rate, sum-squared-error, list of all PFCs, list of all PFC weights, probability of inducing a PFC
		'''

		# Main learning loop
		errRate = 0  # to store the number of errors just in this iteration
		for i in range(0,int(iterations)): # Update 'iterations' times
			LexEntry, err = self.update(theory,learnRate,PFCstartW,pfcLearnRate,PFCSample, PFCSampleSize,decayRate,decayType,havePFC,comparisonStrategy,LexEntryToUse=self.lexList[lexIndices[self.t]])
			errRate += err # update number of errors

		# Info to save after each epoch (so we can understand the learning process well)
		# We don't save after every iteration because that would make learning take way longer and also produce unnecessarily large output data files.

		# error rate - percent of iterations in this epoch that resulted in an error (and therefore learning)
		errRate = float(errRate)/float(iterations)

		# PFC's that exist at the end of the epoch, and their weights
		# Only write them down at the end of an Epoch, to save time
		pfcs  = []
		pfcws = []

		for i in self.lexList:
			# Before you save, resolve all the decay that should have taken place during the epoch
			i.decayPFC(self.t,decayRate,decayType)
			for j in i.pfc:
				pfcs.append((i.inpt,str(i.prob),j[0])) # Save PFC info: (input, lexical frequency, output demanded by PFC)
				pfcws.append(j[1]) #Save PFC weight in a separate list, co-indexed 

		# calculate the estimated probability of inducing a PFC on error during this epoch
		nPFC = 0
		for i in self.lexList:
			if len(i.pfc)>0:
				nPFC += 1  # count up PFCs

		# P(at least once) = 1-P(never)
		# P(never) = (1-nPFC/nLexEntries)^PFCSampleSize
		self.pPFC = 1-pow((1-(float(nPFC)/float(len(self.lexList)))),PFCSampleSize)

		sse = self.SSE() # calculate current sum-squared error

		if not silent: # If silent is set to False (the default) print out updates on all this info 
			print("PFC count: ", nPFC)
			print("PFC prob: " , self.pPFC)
			print("Lexical entries count: ", len(self.lexList))
			print("Current SSE: ", sse)
		return errRate, sse, pfcs, pfcws, self.pPFC

	def learn(self, iterations, nEpochs,learnRate,w=None, PFCstartW=0,pfcLearnRate=0,PFCSample=False,PFCSampleSize=10, decayRate=None, decayType='static',theory='MaxEnt',havePFC=False,reset=True,comparisonStrategy='sample',silent=False):

		'''Main wrapper for learning. Calls the epoch function nEpochs times, and each epoch is iterations number of learning steps.  learnRate can be a real number, or a 'schedule' of two numbers - a starting rate and an ending rate.

		Returns a results file, so it should be run as something like:
		MyResults = MyTableaux.learn(...) '''

		# First, grab all the inputs, so the function call can be saved in the results object
		functionCall=locals()
		k=locals().keys()
		print(k) # Print the function call, just for funzies
		try:
			for i in k:
				functionCall[i]=locals()[i]
		except:
			print("Woah!")

		#TODO: there's a bug right now that if youjust write '0' for the learning rate it's all 'omg int not float omg'.  Just add a little conversion to float at some point.	This might be tricky if we want to maintain the ability to take a plasticity schedule as input.	 Or maybe not.	I just don't feel like thinking about it right now, so.


		if type(learnRate) is float: # If learnRate is a real number
			localLearnRate = learnRate
			plasticity = 0
		else: # If it's some kind of schedule
			localLearnRate = learnRate[0]
			plasticity = (learnRate[0]-learnRate[1])/nEpochs # learnRate decrement per epoch
			# TODO check length of the schedule; allow either a tuple or a list
			# TODO check that second number is less than first
			# Allow for non-linear schedules???
		start = timeit.default_timer() # Time how long it's going to take
		if reset:  # if reset = True, start learning over from the beginning, setting weights to zero.  otherwise, continue where you left off
			# Do a bunch of stuff to start off learning 
			self.initializeWeights()
			self.resetTime()
			self.resetPFC()
			self.lexicon = [[],[]]
			self.pPFC = 1
			#Find the most probable observed candidates for each Lexical Entry
			for u in self.lexList:
				u.getProbableCandidates()
			print("learning...")
			print(theory)
			print(nEpochs)
		else:
			print("continuing to learn...")
		if theory == 'MaxEnt':
			ep = []
			err = []
			weights = []
			sse = []
			pPFCs = []

			pfcWeights = []
			pfcNames = []

			# Create the playlist of URs to learn on
			lexIndices = self.sampleLexEntry(iterations*nEpochs)
			
			firstEpoch=timeit.default_timer()

			for n in range(0,nEpochs):  # Main learning loop
				if not silent:
					print("Epoch ",n)
				localLearnRate -=plasticity #Update the learning rate with plasticity
				w,x,y,z,p = self.epoch(theory,lexIndices,iterations,localLearnRate,PFCstartW,pfcLearnRate, PFCSample, PFCSampleSize, decayRate,decayType,havePFC,comparisonStrategy, silent)
				ep.append(n)
				err.append(w)
				sse.append(x)
				pPFCs.append(p)
				weights.append(self.w)

				# Arrange PFC information
				# y is list of PFCs (tuples)  (LexEntry,tab.prob,PFCname)
				# z is list of weights
				for c in y:
					if c in pfcNames:
						# find the index of c in both lists
						cIndex = pfcNames.index(c)
						# find index of c in z, the list of weights from the epoch
						cWindex = y.index(c)
						# Add the appropriate weight and time to lexiWeights
						pfcWeights[cIndex][0].append(n) #n=epoch
						pfcWeights[cIndex][1].append(z[cWindex])
					else: # Add the constraint to lexiNames and lexiWeights
						pfcNames.append(c)
						pfcWeights.append([[n],[z[y.index(c)]]])

				if n==0:
					endFirstEpoch=timeit.default_timer()
					oneEpochtime=endFirstEpoch-firstEpoch
					predictedRuntime=oneEpochtime*nEpochs
					print("Predicted runtime at this point: " + str(predictedRuntime/60.0) + "minutes")



		stop = timeit.default_timer()
		runtime=stop-start

		results = Results(ep,weights,sse,err,runtime,predictedRuntime,functionCall,self.constraints,[pfcNames,pfcWeights],pPFCs,self.lexicon)

		return results

	def calcLikelihood(self):
		logLikelihood=0
		for i in self.lexList:
			i.calculateHarmony(self.w)
			i.predictProbs(self.w)
			for j in i.candidates:
				pass
				# Log j.observedProb * j.predictedProb ?
		print("This function doesn't work yet.	Please check back later")
		return logLikelihood

	def SSE(self):
		sse=0
		for i in self.lexList:
			i.calculateHarmony(self.w)
			i.predictProbs(self.w)
			for j in i.candidates:
				sse+=pow(j.observedProb-j.predictedProb,2)
		return sse

class Society:	
	'''shell class meant to run iterated learning simulations, or agent based.	Right now it's just iterated  learning, where the final state of one generation is the initial state of the next. '''
	def __init__(self,generations,startTableaux,outputName):
		self.generations = 0
		self.startTableaux = startTableaux
		self.currentTableaux = startTableaux
		self.resultses = [] #Fill this with results objects from each iteration
		self.outputName=outputName

	def iterate(self,nGenerations,iterations, nEpochs,learnRate,w=[],PFCstartW=0,pfcLearnRate=0,PFCSample=False,PFCSampleSize=10, decayRate=None, decayType='static',theory='MaxEnt',havePFC=False,reset=True,comparisonStrategy='sample',updateLexProbs = True, silent = False):

		for i in range(1,nGenerations):
			#Update current tableaux based on last set of results
			if i > 1:
				self.updateTableaux(LexProbs = updateLexProbs)

			#Learn on new tableaux
			results = self.currentTableaux.learn(iterations, nEpochs, learnRate,w,PFCstartW, pfcLearnRate, PFCSample, PFCSampleSize, decayRate, decayType, theory, havePFC, reset, comparisonStrategy, silent)
			results.save(self.outputName+"-"+str(i))

			#Append results
			self.resultses.append(results)

	def updateTableaux(self,candidateProbs = True, LexProbs = True):
		'''Updates the current tableaux set for learning for the next generation.
		Note that this will overwrite currentTableaux - the only tableaux set that is saved forever is the initial tableaux set.

		candidateProbs: if True, the final predicted probs from the end-state of the last learning phase are used as the new training data
		
		LexProbs: if True, the lexical frequencies of each LexEntry are updated, to the number of times each LexEntry was observed by the last generation.  NOTE: with this, especially with small numbers of learning iterations, words could dissappear from the vocabulary.

		Interesting future idea NOT IMPLEMENTED IN ANY WAY: create mechanism for introduction of new words, either by mutating existing words, or by inventing phonotactically plausible new strings '''

		self.newTableaux = Tableaux(theory='MaxEnt')
		self.newTableaux.constraints = self.currentTableaux.constraints
		# Update LexEntries, candidates, probabilities
		if not candidateProbs:	#I'm not sure what the option to make this false will actually be useful for at this point.	 
			print("What are you even doing?")

		for i in self.currentTableaux.lexList:
			if LexProbs:
				newLex = LexEntry(i.inpt,i.nSeen)
			else:
				newLex = LexEntry(i.inpt,i.prob)

			for j in i.candidates:
				newLex.addCandidate(candidate(j.c,j.violations,j.predictedProb))

			self.newTableaux.addLexEntry(newLex)

		self.currentTableaux = self.newTableaux

class Results:
	def __init__(self,t,w,sse,err,runtime,predruntime,functionCall,Cnames,PFCinfo,pPFCs,lexicon):
		self.t = t
		self.w = w
		self.sse = sse
		self.err = err
		self.runtime = runtime
		self.predruntime=predruntime
		self.functionCall=functionCall
		self.Cnames=Cnames
		self.PFCinfo = PFCinfo  
		self.pPFCs = pPFCs
		self.lexicon = lexicon

		# First, group the lexical constraints into classes
		PFCclasses = [[],[],[]]
		for c in self.PFCinfo[0]:
			if c[1] in PFCclasses[0]:
				# extend that output's time vector
				PFCclasses[1][PFCclasses[0].index(c[1])][0].extend(self.PFCinfo[1][self.PFCinfo[0].index(c)][0])
				# extend that output's weight vector
				PFCclasses[1][PFCclasses[0].index(c[1])][1].extend(self.PFCinfo[1][self.PFCinfo[0].index(c)][1])
				# add this input to the list of inputs for that class
				PFCclasses[2][PFCclasses[0].index(c[1])].append(c[0])
			else:
				PFCclasses[0].append(c[1])
				PFCclasses[1].append(self.PFCinfo[1][self.PFCinfo[0].index(c)])
				PFCclasses[2].append([c[0]])
			#print PFCclasses
		self.PFCclasses = PFCclasses

		PFCMeans = [PFCclasses[0],[],PFCclasses[2]]
		for i in PFCclasses[1]:
			PFCmean = []
			# Start by making a structure like [t][w] where each w is a mean of all the weights at the indexed time.
			t = range(1,max(i[0])) #i[0] is a list of timestamps, i[1] corresponding weights
			PFCmean.append(t)
			PFCmean.append([]) # this will hold the sets of weights at each timestamp
			PFCmean.append([]) # and this will hold the numbers of weights at each timestamp
			for ti in t:
				# Grab all the indices in the list of timestamps for each time
				# e.g. all the indices of the timestamp '32'  (when ti is 32)
				indices = [k for k,x in enumerate(i[0]) if x==ti]
				wVect=[]
				for index in indices:
					wVect.append(i[1][index])
				wMean=[sum(wVect)/len(wVect) if len(wVect)>0 else 0] #Calculate and append the mean of the set of weights observed at time ti
				PFCmean[1].append(wMean)
				PFCmean[2].append(len(wVect))
			PFCMeans[1].append(PFCmean) # Add this list of t,w for this class of items.
		self.PFCMeans = PFCMeans


	def save(self,filename,folder=""):
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
			f.write("Probability of inducing a PFC: "+str(self.pPFCs))
			f.write("\n")
			f.write("Note: the above probability may be meaningless unless you're using the probabilistic inducing functionality")

		with open(folder+filename+"_weights",'w') as f:
			f.write("\t".join(["t"]+self.Cnames))
			f.write("\n")
			for i in range(len(self.w)):
				f.write("\t".join([str(self.t[i])]+[str(j) for j in self.w[i]]))
				f.write("\n")
		with open(folder+filename+"_SSE",'w') as f:
			f.write("\t".join(["t","SSE"]))
			f.write("\n")
			for i in range(len(self.sse)):
				f.write("\t".join([str(self.t[i])]+[str(self.sse[i])]))
				f.write("\n")
		with open(folder+filename+"_err",'w') as f:
			f.write("\t".join(["t","err"]))
			f.write("\n")
			for i in range(len(self.err)):
				f.write("\t".join([str(self.t[i])]+[str(self.err[i])]))
				f.write("\n")
		with open(folder+filename+"_pPFC",'w') as f:
			f.write("\t".join(["t","pPFCs"]))
			f.write("\n")
			for i in range(len(self.pPFCs)):
				f.write("\t".join([str(self.t[i])]+[str(self.pPFCs[i])]))
				f.write("\n")
		with open(folder+filename+"_lexicon",'w') as f:
			f.write("\t".join(["word","count"]))
			f.write("\n")
			for i in self.lexicon[0]:
				f.write("\t".join([i.inpt]+[str(self.lexicon[1][self.lexicon[0].index(i)])]))
				f.write("\n")


		#Groundwork for printing PFC weights over time
		structuredPFCWeights={}
		for i in range(len(self.PFCinfo[0])): #iterate through the indices of the constraints (PFCinfo[0]) and their time vector/weight vector pairs (PFCinfo[1])
			structuredPFCWeights[self.PFCinfo[0][i]]=[] # add the current constraint name, in the form of a tuple of (input, tab.prob, prescribed output), as a new key in this new dictionary thing
			for j in self.t: #Iterate through timesteps - these are epochs of the learning process
				if j in self.PFCinfo[1][i][0]: # if there's an entry for that timestep in the weight vector of the current constraint...
					structuredPFCWeights[self.PFCinfo[0][i]].append(self.PFCinfo[1][i][1][self.PFCinfo[1][i][0].index(j)]) # add that constraint's weight at the appropriate time index
				else:
					structuredPFCWeights[self.PFCinfo[0][i]].append(0) #otherwise put a zero in that spot, as a placeholder.

		# k now we can try to print to a file.	there will be t columns in this crazy file.	 More reason to be thrifty about your sampling of the process, and keep nEpochs low
		with open(filename+"_PFCinfo",'w') as f:
			for i in structuredPFCWeights:
				f.write("\t".join(i))
				f.write("\t")
				f.write("\t".join([str(k) for k in structuredPFCWeights[i]]))
				f.write("\n")

		with open(filename+"_finalState",'w') as f:
			f.write("\t".join(["LexEntry","Candidate","ObservedProb","PredictedProb","PredictedProbNoPFC","observedFreq","trainedFrequency","PFC_W"]))
			f.write("\n")
			for i in range(len(self.lexicon[0])):
				#Predict probabilities for each candidate without their PFCs
				self.lexicon[0][i].predictProbs(self.w[-1],suppressPFC=True)
				noPFCPreds=[]
				for c in self.lexicon[0][i].candidates:
					noPFCPreds.append(c.predictedProb)
				# Use the final weights to predict probabilities with PFCs
				self.lexicon[0][i].predictProbs(self.w[-1])
				for c in self.lexicon[0][i].candidates:
					pfcW = "NA"
					for p in self.lexicon[0][i].pfc:
						if p[0]==c.surfaceForm:
							pfcW = p[1]
					line = "\t".join([self.lexicon[0][i].inpt, c.c, str(c.observedProb), str(c.predictedProb),str(noPFCPreds[self.lexicon[0][i].candidates.index(c)]), str(self.lexicon[0][i].prob), str(self.lexicon[1][i]),str(pfcW)])
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
			print("Oops!  It looks like you don't have matplotlib installed.")

	def plotpPFCs(self):
		try:
			fig = plt.figure()
			errorplot = fig.add_subplot(111)
			errorplot.plot(self.t,self.pPFCs,color='red',lw=2)
			errorplot.set_xlabel('Epoch')
			errorplot.set_ylabel('pPFC')
			plt.show(block=False)
		except:
			print("Gak!	 No matplotlib installed!  No plots for you.")

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
				print(colors.cnames.keys()[i])
			weightplot.set_xlabel('Epoch')
			weightplot.set_ylabel('W')
			box = weightplot.get_position()
			weightplot.set_position([box.x0, box.y0, box.width * 0.8, box.height])
			weightplot.legend(bbox_to_anchor=(1,0.5), loc=2, borderaxespad=0.5)
			plt.show(block=False)
		except:
			print("You haven't installed matplotlib, but that's ok.	 Just save using the handy save() function, and then use your favorite plotting software.")

	def plotMeanPFCW(self):
		try:
			# Now, plot the things
			fig = plt.figure()
			weightplot = fig.add_subplot(111)
			for i in range(len(self.PFCMeans[0])): #iterate over classes
				weightplot.plot(self.PFCMeans[1][i][0],self.PFCMeans[1][i][1],color=colors.cnames[colors.cnames.keys()[i]],label=self.PFCMeans[0][i])
			weightplot.set_xlabel('Epoch')
			weightplot.set_ylabel('W')
			box = weightplot.get_position()
			weightplot.set_position([box.x0, box.y0, box.width * 0.8, box.height])
			weightplot.legend(bbox_to_anchor=(1,1), loc=2, borderaxespad=0.5)
			plt.show(block=False)

			fig = plt.figure()
			weightplot = fig.add_subplot(111)
			for i in range(len(self.PFCMeans[0])): #iterate over classes
				weightplot.plot(self.PFCMeans[1][i][0],self.PFCMeans[1][i][2],color=colors.cnames[colors.cnames.keys()[i]],label=self.pfcMeans[0][i])
			weightplot.set_xlabel('Epoch')
			weightplot.set_ylabel('nW')
			box = weightplot.get_position()
			weightplot.set_position([box.x0, box.y0, box.width * 0.8, box.height])
			weightplot.legend(bbox_to_anchor=(1,1), loc=2, borderaxespad=0.5)
			plt.show(block=False)
		except:
			print("Yea, I can't find matplotlib.  No plots for you this time!")


def perceptronUpdate(error,target,weights,rate):
	"""error and target should be violation vectors, weights should be a vector of weights, and rate is the learning rate."""
	if len(error)!=len(target):
		sys.exit("It looks like you tried to update with two violation vectors that are different lengths.	Maybe check your input file?")
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
	return item	 # Might occur because of floating point inaccuracies

def readOTSoft(file):
	"""Reads in an OTSoft file type"""
	print('reading tableaux...')
	tableaux = Tableaux()
	with open(file,"r") as f:
		lines=f.read().split('\n')
		# Check if the linebreak character is correct.	If it's not, the symptoms will be (a) only one line and (b) '\r' somewhere inside the line
		if bool(re.match('.*\r.*',lines[0])):
			if len(lines)==1:
				lines=lines[0].split('\r')
			else:
				print("EEK something is wrong with your linebreaks")

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
					print("I can't tell what type of file this is :(")

			elif inputType=='hgr':
				inpt = line[0]
				surfaceForm = (line[1] if hidden else None)
				c = (line[2] if hidden else line[1])
				observedProb = (line[3] if hidden else line[2])
				tokenProb = (line[offset-1] if tokenFrequency else 1)
				violations = line[offset:]
				#print(ur)
				if inpt not in tableaux.lexIDlist:
					#print 'add'
					tableaux.addLexEntry(LexEntry(inpt,tokenProb))
				for i in tableaux.lexList:
					if i.inpt==inpt:
						i.addCandidate(candidate(c,violations,observedProb,surfaceForm))
						break
			lineno+=1

	return tableaux


