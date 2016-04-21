import sys
import math
import re
from scipy.stats import rv_discrete
import numpy as np 
import matplotlib.pyplot as plt

class candidate:
	def __init__(self,c,violations,observedProb,surfaceForm=None):
		self.c = c # This is the hidden structure, or the actual candidate that's got a violation profile to learn from
		self.violations = violations
		if surfaceForm is None:
			self.surfaceForm = c
		else:
			self.surfaceForm = surfaceForm # This is the output
		self.observedProb = observedProb
		try:
			self.observedProb = float(self.observedProb)
		except ValueError:
			print "It looks like your candidates' probabilities can't all be converted to floats.  Check if there's some text in that column"
		self.harmony = 0
		self.predictedProb = 0
		self.checkViolationsSign()

	def checkViolationsSign(self): #convert all violations to negative numbers
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
		self.candidates = []
		if lexC is None:
			self.lexC = [] # This can later be filled with tuples sub-lists
		# e.g. [output, weight]
		# it's important that output is a surface form, not a hidden structure
		self.probDenom = 1 # denominator for calculating MaxEnt prob of each candidate
						   # this is the sum of each exponentiated harmony score
		self.predProbsList = []
		self.obsProbsList = []
		self.lastSeen = 0 # Keep track of what timestep the learner last saw this word
		self.nSeen = 0 # Keep track of how many times the UR has been seen

	def addCandidate(self,cand):
		self.candidates.append(cand)
		self.obsProbsList.append(cand.observedProb)

	def decayLexC(self,t,decayRate,decayType='linear'):
		if len(self.lexC)>0:
			if decayType=='linear':
				for i in range(0,len(self.lexC)):
					self.lexC[i][1]-=(t-self.lastSeen)*decayRate
					
			if decayType=='logarithmic':
				for i in range(0,len(self.lexC)):
					self.lexC[i][1]-=pow(t-self.lastSeen,decayRate)

			if self.lexC[i][1]<0: #If it's gotten as low as zero remove the constraint
				self.lexC.remove(self.lexC[i])

	def checkViolationLength(self): #check if all the violation vectors are the same length
		pass

	def calculateHarmony(self, w, t=None, decayRate=None, decayType=None): # Takes a vector of weights, equal in length to the violation vectors of the candidates
		self.probDenom=0 # Reset probDenom
		if (decayRate is not None) and (t is None):
			sys.exit("You have to pass a time parameter to the calculateHarmony function if you're using a decay rate")
		if decayRate is not None:
			self.decayLexC(t,decayRate,decayType)
		for cand in self.candidates:
			# dot product
			cand.harmony = sum(viol*weight for viol,weight in zip(cand.violations,w))
			# Assuming the candidate's violations and weights are the correct sign, this'll be a negative number
			for i in self.lexC: # now add in stuff for the lexical constraint
				cand.harmony += i[1]*(0 if cand.surfaceForm==i[0] else -1)
			self.probDenom += pow(math.e,cand.harmony)

	def predictProbs(self,w, t=None, decayRate=None, decayType=None):
		#print 'predictProbs: ', decayRate
		self.calculateHarmony(w, t, decayRate, decayType)
		self.predProbsList=[]
		for cand in self.candidates:
			cand.predictedProb = pow(math.e,cand.harmony)/self.probDenom
			self.predProbsList.append(cand.predictedProb)

	def getPredWinner(self,theory):
		if theory =='HG':
			pass # Figure out which candidate has the highest harmony
		if theory =='MaxEnt': # sample from distribution
			winCandidate = self.candidates[rv_discrete(values=(range(0,len(self.candidates)),self.predProbsList)).rvs(size=1)] # rv_discrete(values=(l,[0,0,.8,0.05,0.05,0,0.1])).rvs(size=100)
			winner = winCandidate.surfaceForm
		return winner, winCandidate

	def getObsWinner(self,theory):
		if theory =='HG':
			pass # Figure out which candidate has the highest harmony
		if theory =='MaxEnt': # sample from distribution
			# Get candidate list
			# Get probability list
			winCandidate = self.candidates[rv_discrete(values=(range(0,len(self.candidates)),self.obsProbsList)).rvs(size=1)] # rv_discrete(values=(l,[0,0,.8,0.05,0.05,0,0.1])).rvs(size=100)
			winner = winCandidate.surfaceForm
		return winner, winCandidate

	def compareObsPred(self,theory,w, t=None, decayRate=None, decayType=None):
		if theory =='batchGD':
			pass
			# Have to do some kind of vector comparison for batch gradient descent
		if theory =='MaxEnt':
			self.predictProbs(w, t, decayRate, decayType)
			obs, obsCandidate=self.getObsWinner(theory) # Sample from observed distribution
			pred, predCandidate=self.getPredWinner(theory) # Sample from predicted distribution
			error = (0 if obs==pred else 1)
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

	def addUR(self,ur):
		self.urList.append(ur)
		self.urIDlist.append(ur.ur)
		self.urProbsList.append(ur.prob)

	def initializeWeights(self):
		self.w = [0]*len(self.constraints)

	def sample(self):
		sampleVector = [i/sum(self.urProbsList) for i in self.urProbsList]
		#print sampleVector
		th = rv_discrete(values=(range(0,len(sampleVector)),sampleVector)).rvs(size=1)
		#print th
		theUR = self.urList[th] # rv_discrete(values=(l,[0,0,.8,0.05,0.05,0,0.1])).rvs(size=100)
		return theUR


	def update(self,theory,learnRate,lexCstartW, decayRate=None, decayType=None):
		# Sample an input form
		theUR = self.sample()
		#print theUR.ur

		e, o, p = theUR.compareObsPred(theory,self.w, self.t, decayRate, decayType)
		if e: # on error
			# update general constraints with perceptron update
			self.w = perceptronUpdate(p.violations, o.violations, self.w, learnRate)
			self.w=[i if i>0 else 0 for i in self.w]

			# update lexC's
			existsLexC = False
			for c in theUR.lexC:
				# Check if lex C favors the observed form
				if c[0]==o.surfaceForm:
					c[1]+=learnRate # increment lexC by learning rate
					existsLexC = True
				else:
					c[1]-=learnRate #else, decrement by learning rate
			if not existsLexC: # if there's no lexical constraint for the observed output, make one
				theUR.lexC.append([o.surfaceForm,lexCstartW])
		self.t+=1
		theUR.lastSeen = self.t
		theUR.nSeen += 1
		return theUR, e #return the UR and whether or not there was an error

	def epoch(self,theory,iterations,learnRate,lexCstartW, decayRate=None, decayType=None):
		errRate = 0
		for i in range(0,int(iterations)):
			UR, err = self.update(theory,learnRate,lexCstartW,decayRate,decayType)
			errRate += err
		errRate = float(errRate)/float(iterations)
		#print errRate

		sse = self.SSE()

		# Lexically specific constraints and their weights
		lexCs  = []
		lexCws = []
		for i in self.urList:
			for j in i.lexC:
				lexCs.append((i.ur,j[0]))
				lexCws.append(j[1])
		return errRate, lexCs, lexCws

	def resetTime(self):
		''' Resets time.  Sets Tableaux.t to 0, and also goes through each UR and resets its last seen time (UR.lastSeen) to 0'''
		self.t = 0
		for x in self.urList:
			x.lastSeen = 0

	def learn(self, iterations, nEpochs,learnRate,lexCstartW, decayRate=None, decayType='linear',theory='MaxEnt'):
		self.initializeWeights()
		self.resetTime()
		print "learning..."
		if theory == 'MaxEnt':
			snapshots = []
			for n in range(0,nEpochs):
				x,y,z = self.epoch(theory,iterations,learnRate,lexCstartW,decayRate,decayType)
				snapshots.append([n,x,y,z])
		return snapshots

	def printTableau(self):
		print "urProb", "\t", "UR", "\t","Candidate", "\t", "H", "\t", "Observed", "\t", "Predicted", "\t", self.constraints
		for i in self.urList:
			for j in i.candidates:
				print i.prob, "\t", i.ur, "\t", j.c,"\t",j.harmony,"\t",j.observedProb,"\t",j.predictedProb, "\t", j.violations

	def printPrettyTableau(self):
		plot = [["urProb","UR","Candidate","H","Observed","Predicted"]]
		plot[0].extend(self.constraints)
		for i in self.urList:
			for j in i.candidates:
				row =[i.prob,i.ur,j.c,j.harmony,j.observedProb,j.predictedProb]
				row.extend(j.violations)
				plot.append(row)
		plt.table(cellText=plot,loc="center")
		plt.show(block=False)

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



def perceptronUpdate(error,target,weights,rate):
	"""error and target should be violation vectors, weights should be a vector of weights, and rate is the learning rate."""
	if len(error)!=len(target):
		sys.exit("It looks like you tried to update with two violation vectors that are different lengths.  Maybe check your input file?")
	if min(error+target)<0 and max(error+target)>0:
		sys.exit("It looks like your tableaux contain both positive and negative violations.  Check your input file.  If this isn't an error, consider using both positive and negative weights instead.")
	error=[abs(i) for i in error]
	target=[abs(i) for i in target]
	return [w+(e-t)*rate for w,e,t in zip(weights,error,target)]


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
		print lines
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
