import sys
import math
from scipy.stats import rv_discrete


def readGrammarPraat(file):
	"""Reads in a Praat grammar file, and converts it to a tableaux dictionary and a list for the constraints.  In the tableaux dictionary, keys are inputs, values are dictionaries, one for each candidate: candidates are keys; values are lists of violations"""
	f = open(file, "r")
	tableaux={}
	constraints=[]
	for line in f:
		line = line.split('\t')
		if len(line)<3:
			pass
		#If first member of the line is 'input' start a new input
		elif line[0]=='input':
			inpt=line[2].strip()
			tableaux[inpt]={}
		elif line[1]=='candidate':
			tableaux[inpt][line[3].strip()]=[float(x.strip('\r\n')) for x in line[4:]]
		elif line[0]=='constraint':
			constraints.append((line[2],line[3]))
	return tableaux, constraints

def readData(file):
	"""Reads in a tab-delimited file with the first column being inputs and the second column being their correct outputs, and the third column being their frequency of occurrence."""
	f=open(file,"r")
	data={}
	for line in f:
		line = line.split('\t')
		if len(line)>2:
			data[(line[0],line[1])]=float(line[2].strip('\n'))

	for k,f in data.items():
		data[k]=f/sum(data.values())
	return data

class candidate:
	def __init__(self,c,violations,observedProb,surfaceForm=None):
		self.c = c # This is the hidden structure, or the actual candidate that's got a violation profile to learn from
		self.violations = violations
		if surfaceForm is None:
			self.surfaceForm = c
		else:
			self.surfaceForm = surfaceForm # This is the output
		self.observedProb = observedProb
		self.harmony = 0
		self.predictedProb = 0
		self.checkViolationsSign()
	def checkViolationsSign(self): #convert all violations to negative numbers
		for i in range(0,len(self.violations)):
			if self.violations[i]>0:
				self.violations[i] = - self.violations[i]


class UR:
	def __init__(self,ur,prob=1,lexC=None):
		self.ur = ur
		self.prob = prob # Lexical frequency of this form
		self.candidates = []
		if lexC is not None:
			self.lexC = [] # This can later be filled with tuples
		# e.g. (output, weight)
		# it's important that output is a surface form, not a hidden structure
		self.probDenom = 1 # denominator for calculating MaxEnt prob of each candidate
						   # this is the sum of each exponentiated harmony score
		self.predProbsList = []
		self.lastSeen # Keep track of what timestep the learner last saw this word

	def addCandidate(self,cand):
		self.candidates.append(cand)
	
	def decayLexC(self,t,decayRate,decayType='linear'):
		if len(self.lexC>1):
			if decayType=='linear':
				for i in range(0,len(self.lexC)):
					self.lexC[i][1]-=(t-self.lastSeen)*decayRate
			if decayType=='logarithmic':
				for i in range(0,len(self.lexC)):
					self.lexC[i][1]-=pow(t-self.lastSeen,decayRate)
	
	def checkViolationLength(self): #check if all the violation vectors are the same length
		pass
	
	def calculateHarmony(self, w, t=None, decayRate=None, decayType=None): # Takes a vector of weights, equal in length to the violation vectors of the candidates
		if (decayRate is not None) and (t is None):
			sys.exit("You have to pass a time parameter to the calculateHarmony if you're using a decay rate")
		if decayRate is not None:
			self.decayLexC()
		for cand in self.candidates:
			# dot product
			cand.harmony = sum(viol*weight for viol,weight in zip(cand.violations,w))
			# Assuming the candidate's violations and weights are the correct sign, this'll be a negative number
			for i in self.lexC: # now add in stuff for the lexical constraint
				cand.harmony += i[1]*(0 if cand.surfaceForm==i[0] else 1)
			self.probDenom += pow(math.e,cand.harmony)

	def predictProbs(self,w):
		self.calculateHarmony(w)
		for cand in self.candidates:
			cand.predictedProb = pow(math.e,cand.harmony)/self.probDenom

	def getWinner(self,theory):
		if theory =='HG':
			pass # Figure out which candidate has the highest harmony
		if theory =='MaxEnt': # sample from distribution
			rv_discrete() # rv_discrete(values=(l,[0,0,.8,0.05,0.05,0,0.1])).rvs(size=100)

class Tableaux:
	def __init__(self,theory='MaxEnt'):
		self.urList = []
		self.urIDlist = []
		self.theory = theory

	def addUR(self,ur):
		self.urList.append(ur)
		self.urIDlist.append(ur.ur)






def readOTSoft(file):
	"""Reads in an OTSoft file type"""
	with open(file,"r") as f:
		firstLine = f.read().split('\r',1)[0]
		firstLine = firstLine.split('\t')
		if firstLine[1]=="":
			# OTSoft file? They have empty cells at the top
			pass

		if firstLine[0]=="input":
			# hgr file? They have the first three columns labeled
			inputType='hgr'
			# Constraints are in the first line too, so grab those
			# Headers of these files:
			# input output (hidden) probability (tab.prob) Constraint1 Constraint2 ...
			offset = (3 if firstLine[2]=='hidden' else 2)
			hidden = (True if firstLine[2]=='hidden' else False)
			offset = offset + (2 if firstLine[offset+1]=='tab.prob' else 1)
			tokenFrequency = (True if firstLine[offset+1]=='tab.prob' else False)
			constraints=firstLine[offset:]

		tableaux = Tableaux()
		for line in f:
			line=line.split('\t')
			if inputType=='hgr':
				ur = line[0]
				surfaceForm = (line[1] if hidden else None)
				c = (line[2] if hidden else line[1])
				observedProb = (line[3] if hidden else line[2])
				tokenProb = (line[offset-1] if tokenFrequency else 1)
				violations = line[offset:]

				if ur not in tableaux.urIDlist:
					tableaux.addUR(UR(ur,tokenProb))

				for i in tableaux.urList:
					if i.ur==ur:
						ur.addCandidate(c,violations,observedProb,surfaceForm)
						break

			else:
				print "I can't tell what type of file this is :("


def sample(probs,labels): #takes two lists: probabilities and corresponding labels. Samples from the probability distribution, returning a label
        #Strategy: generate a random number, then take the list of probabilities and 'stack' them into a set of sums
        import random
        probsum=0
        r=random.random()
        for i, j in zip(probs,labels):
                if r <= i+probsum: # run through the probs, summing them as you go.  When you get between one and the last
                        return j   # return the label corresponding to that probability
                probsum+=i

def perceptronUpdate(error,target,weights,rate):
	"""error and target should be violation vectors weights should be a vector of weights, and rate is the learning rate"""
	return [w+(e-t)*rate for w,e,t in zip(weights,error,target)]

def eval(inpt,tableaux,weights,theory="HG"):
	""" Takes a tableaux file, weights, and an input; returns the optimal output and it's harmony """
	H=[]
	candidates=[]
	#print tableaux[inpt]
	for cand,violations in tableaux[inpt].items():
		candidates.append(cand)
		#print weights
		H.append(sum([v*w for v, w in zip(violations, weights)]))
	return candidates[H.index(min(H))],min(H)

def accuracy(data,tableaux,weights):
	"""Calculates total accuracy on some data for some tableaux and set of weights """
	accuracies=[]
	for datum in data.keys():
		if eval(datum[0],tableaux,weights)[0]==datum[1]:
			accuracies.append(1.0)
		else:
			accuracies.append(0.0)
	return sum(accuracies)/len(accuracies)

def iteration(data,tableaux,constraints,rate):
	"""Performs a single iteration, sampling an input-output pair from 'data' and updating the weights of 'constraints'"""
	weights=constraints[1]
	datum=sample(data.values(),data.keys())
	predicted=eval(datum[0],tableaux,constraints[1],rate)[0]
	#print constraints[1]
	if predicted!=datum[1]:
		weights=perceptronUpdate(tableaux[datum[0]][predicted],tableaux[datum[0]][datum[1]],constraints[1],rate)
		print 'ERROR:'
		print 'data:	', datum[0], datum[1], 'OUTPUT 	', predicted
		print 'CURRENT GRAMMAR: ', weights
		print 'ACCURACY is ', accuracy(data,tableaux,weights)
	return weights

def initialize(data,tableaux,constraints,uni):
	"""Take in a grammar, a set of constraints, and an initial state (uni), which should be 0 for mark >> faith, and 1 for uniform starting weights"""
	constraintVectors=([],[])
	for c in constraints:
		constraintVectors[0].append(c[0])
		if uni:
			constraintVectors[1].append(9*float(c[1])+1)
		else:
			constraintVectors[1].append(1)
	return constraintVectors

	#Checks - does everything make sense?
	#Check that violation vectors are the same length as the constraints
	#Check that all the inputs and outputs in data are attested in the tableaux

def summarize(data,tableaux,constraints):
	for datum in data.keys():
		print "data: ", datum[0], datum[1], "OUTPUT:", eval(datum[0],tableaux,constraints[1])[0]

def learn(data,tableaux,constraints,uni,rate,itr):
	constraints=initialize(data,tableaux,constraints,uni)

	print constraints
	print "CONSTRAINTS:", constraints[0]
	print "STARTING GRAMMAR:", constraints[1]
	summarize(data,tableaux,constraints)

	for i in range(itr):
		if i%100==0:
			print "ITERATION: " + str(i) + "\n"
			print "CURRENT GRAMMAR:", constraints[1]
			summarize(data,tableaux,constraints)
			print "ACCURACY is ", accuracy(data,tableaux,constraints[1])

		constraints=(constraints[0],iteration(data,tableaux,constraints,rate))

gramFile=sys.argv[1]
dataFile=sys.argv[2]
uni=int(sys.argv[3])
itr=int(sys.argv[4])
rate=float(sys.argv[5])

tableaux, constraints = readGrammarPraat(gramFile)
data = readData(dataFile)
learn(data,tableaux,constraints,uni,rate,itr)


